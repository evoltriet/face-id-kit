from __future__ import annotations

import copy
import json
import sqlite3
import threading
from dataclasses import asdict, replace
from pathlib import Path
from .matching import check_model
from .protocols import EmbeddingCipher
from .types import EnrollmentSample, Identity, ModelSpec
import numpy as np


def _validated(sample: EnrollmentSample) -> EnrollmentSample:
    if not sample.id or not sample.identity_id or not np.isfinite(sample.quality) or not 0 <= sample.quality <= 1:
        raise ValueError("Sample IDs and quality in [0, 1] are required")
    return replace(sample, embedding=check_model(sample.model, sample.model, sample.embedding).copy())


class InMemoryStore:
    def __init__(self):
        self._revision = 0
        self._identities: dict[str, Identity] = {}
        self._samples: dict[str, EnrollmentSample] = {}
        self._lock = threading.RLock()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def snapshot(self):
        with self._lock:
            return self._revision, copy.deepcopy(list(self._samples.values()))

    def identities(self):
        with self._lock:
            return copy.deepcopy(list(self._identities.values()))

    def put_identity(self, identity: Identity):
        if not identity.id:
            raise ValueError("Identity ID is required")
        with self._lock:
            self._identities[identity.id] = copy.deepcopy(identity)
            self._revision += 1

    def put_sample(self, sample: EnrollmentSample):
        sample = _validated(sample)
        with self._lock:
            if sample.identity_id not in self._identities:
                raise KeyError(sample.identity_id)
            self._samples[sample.id] = sample
            self._revision += 1

    def delete_sample(self, sample_id: str):
        with self._lock:
            self._samples.pop(sample_id, None)
            self._revision += 1

    def delete_identity(self, identity_id: str):
        with self._lock:
            self._identities.pop(identity_id, None)
            self._samples = {k: v for k, v in self._samples.items() if v.identity_id != identity_id}
            self._revision += 1


class SQLiteStore:
    """Model/sample metadata are plaintext; embedding bytes require an explicit cipher."""
    def __init__(self, path: Path, *, cipher: EmbeddingCipher):
        if cipher is None:
            raise ValueError("Persistent storage requires an explicit embedding cipher")
        self.path, self.cipher = Path(path), cipher
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS kit_meta(version INTEGER NOT NULL, revision INTEGER NOT NULL);
                INSERT INTO kit_meta SELECT 1,0 WHERE NOT EXISTS(SELECT 1 FROM kit_meta);
                CREATE TABLE IF NOT EXISTS identities(id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS samples(
                    id TEXT PRIMARY KEY, identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
                    embedding BLOB NOT NULL, model TEXT NOT NULL, quality REAL NOT NULL, source_id TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS sample_insert AFTER INSERT ON samples BEGIN UPDATE kit_meta SET revision=revision+1; END;
                CREATE TRIGGER IF NOT EXISTS sample_update AFTER UPDATE ON samples BEGIN UPDATE kit_meta SET revision=revision+1; END;
                CREATE TRIGGER IF NOT EXISTS sample_delete AFTER DELETE ON samples BEGIN UPDATE kit_meta SET revision=revision+1; END;
                CREATE TRIGGER IF NOT EXISTS identity_update AFTER UPDATE ON identities BEGIN UPDATE kit_meta SET revision=revision+1; END;
                CREATE TRIGGER IF NOT EXISTS identity_insert AFTER INSERT ON identities BEGIN UPDATE kit_meta SET revision=revision+1; END;
                CREATE TRIGGER IF NOT EXISTS identity_delete AFTER DELETE ON identities BEGIN UPDATE kit_meta SET revision=revision+1; END;
            """)
            if db.execute("SELECT version FROM kit_meta").fetchone()[0] != 1:
                raise ValueError("Unsupported storage schema version")

    def _connect(self):
        # The connection context manager commits but does not close; use closing wrapper.
        from contextlib import contextmanager
        @contextmanager
        def connection():
            db = sqlite3.connect(self.path, timeout=30)
            try:
                db.execute("PRAGMA foreign_keys=ON")
                db.execute("PRAGMA secure_delete=ON")
                with db:
                    yield db
            finally:
                db.close()
        return connection()

    @property
    def revision(self):
        with self._connect() as db:
            return db.execute("SELECT revision FROM kit_meta").fetchone()[0]

    def snapshot(self):
        with self._connect() as db:
            db.execute("BEGIN")
            revision = db.execute("SELECT revision FROM kit_meta").fetchone()[0]
            rows = db.execute("SELECT id,identity_id,embedding,model,quality,source_id FROM samples ORDER BY id").fetchall()
        samples = []
        for sid, iid, blob, model, quality, source in rows:
            spec = ModelSpec(**json.loads(model))
            raw = self.cipher.unprotect(blob)
            if len(raw) != spec.dimension * 4:
                raise ValueError("Corrupt embedding length")
            samples.append(_validated(EnrollmentSample(sid, iid, np.frombuffer(raw, dtype="<f4").copy(), spec, quality, source)))
        return revision, samples

    def identities(self):
        with self._connect() as db:
            return [Identity(iid, json.loads(meta)) for iid, meta in db.execute("SELECT id,metadata FROM identities ORDER BY id")]

    def put_identity(self, identity: Identity):
        if not identity.id:
            raise ValueError("Identity ID is required")
        with self._connect() as db:
            db.execute("INSERT INTO identities VALUES(?,?) ON CONFLICT(id) DO UPDATE SET metadata=excluded.metadata",
                       (identity.id, json.dumps(identity.metadata)))

    def put_sample(self, sample: EnrollmentSample):
        sample = _validated(sample)
        blob = self.cipher.protect(sample.embedding.astype("<f4").tobytes())
        with self._connect() as db:
            db.execute("""INSERT INTO samples VALUES(?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET identity_id=excluded.identity_id, embedding=excluded.embedding,
                model=excluded.model, quality=excluded.quality, source_id=excluded.source_id""",
                (sample.id, sample.identity_id, blob, json.dumps(asdict(sample.model)), sample.quality, sample.source_id))

    def delete_sample(self, sample_id: str):
        with self._connect() as db:
            db.execute("DELETE FROM samples WHERE id=?", (sample_id,))

    def delete_identity(self, identity_id: str):
        with self._connect() as db:
            db.execute("DELETE FROM identities WHERE id=?", (identity_id,))
