import os
from dataclasses import replace
import numpy as np
import pytest

from face_id_kit import (Candidate, EnrollmentSample, Identity, Identifier, InMemoryStore,
                         MatchPolicy, ModelSpec, SQLiteStore)
from face_id_kit.crypto import AESGCMCipher, DPAPICipher
from face_id_kit.matching import normalize_embedding, resolve_unique
from face_id_kit.live import TrackConsensus

MODEL = ModelSpec("synthetic-v1", 3, "test-l2")
A = np.array([1., 0., 0.])
B = np.array([0., 1., 0.])


def sample(sid="one", iid="a", vector=A):
    return EnrollmentSample(sid, iid, vector, MODEL)


@pytest.mark.parametrize("bad", [np.array([]), np.zeros(3), np.array([np.nan]), np.ones((2, 2))])
def test_invalid_vectors_rejected(bad):
    with pytest.raises(ValueError):
        normalize_embedding(bad)


def test_gallery_edit_reassign_replace_delete_and_rename():
    store = InMemoryStore()
    store.put_identity(Identity("a", {"name": "First name"}))
    store.put_identity(Identity("b"))
    store.put_sample(sample())
    identifier = Identifier(store, MODEL)
    assert identifier.identify(A, MODEL).identity_id == "a"
    store.put_identity(Identity("a", {"name": "New name"}))
    assert identifier.identify(A, MODEL).identity_id == "a"
    store.put_sample(sample(iid="b"))  # same sample count and ID, changed identity
    assert identifier.identify(A, MODEL).identity_id == "b"
    store.put_sample(sample(iid="b", vector=B))
    assert identifier.identify(A, MODEL).identity_id is None
    store.delete_identity("b")
    assert store.snapshot()[1] == []
    store.put_sample(sample())
    store.delete_sample("one")
    assert identifier.identify(A, MODEL).reason == "empty_gallery"


def test_snapshot_does_not_allow_untracked_mutation():
    store = InMemoryStore()
    store.put_identity(Identity("a"))
    store.put_sample(sample())
    store.snapshot()[1][0].embedding[:] = B
    assert np.allclose(store.snapshot()[1][0].embedding, A)


def test_model_version_and_dimension_mismatches():
    store = InMemoryStore()
    store.put_identity(Identity("a"))
    store.put_sample(sample())
    identifier = Identifier(store, MODEL)
    for spec in [replace(MODEL, fingerprint="new"), replace(MODEL, preprocessing="new")]:
        with pytest.raises(ValueError):
            identifier.identify(A, spec)
    with pytest.raises(ValueError):
        store.put_sample(sample(vector=np.ones(4)))


def test_unknown_ambiguous_and_conflicts_never_promote_runner_up():
    lists = [[Candidate("a", .9), Candidate("b", .7)],
             [Candidate("a", .86), Candidate("b", .80)],
             [Candidate("a", .4), Candidate("b", .2)]]
    results = resolve_unique(lists, MatchPolicy())
    assert [r.identity_id for r in results] == ["a", None, None]
    assert results[1].reason == "identity_conflict"
    assert resolve_unique([[Candidate("a", .7), Candidate("b", .68)]], MatchPolicy())[0].reason == "ambiguous"


def test_tracking_survives_order_changes_and_resets_ambiguous_crossing():
    tracker = TrackConsensus(required=2)
    left, right = (0, 0, 10, 10), (30, 0, 10, 10)
    initial = tracker.update([left, right], ["a", "b"], now=0)
    reordered = tracker.update([right, left], ["b", "a"], now=.2)
    assert reordered == [(initial[1][0], True), (initial[0][0], True)]
    ambiguous = tracker.update([left, left], ["a", "b"], now=.4)
    assert not any(stable for _, stable in ambiguous)
    expired = tracker.update([left], ["a"], now=2)
    assert not expired[0][1]


def test_missed_and_changed_identities_do_not_leak_stability():
    tracker = TrackConsensus(required=2)
    box = (0, 0, 10, 10)
    tracker.update([box], ["a"], now=0)
    assert tracker.update([box], ["a"], now=.1)[0][1]
    assert not tracker.update([box], ["b"], now=.2)[0][1]
    for time in [.3, .4, .5, .6, .7, .8]:
        tracker.update([], [], now=time)
    assert not tracker.update([box], ["b"], now=.9)[0][1]


def test_encrypted_sqlite_reload_revision_and_delete(tmp_path):
    key = os.urandom(32)
    path = tmp_path / "gallery.db"
    store = SQLiteStore(path, cipher=AESGCMCipher(key))
    store.put_identity(Identity("a", {"name": "Test"}))
    store.put_sample(sample())
    assert sample().embedding.astype("<f4").tobytes() not in path.read_bytes()
    reopened = SQLiteStore(path, cipher=AESGCMCipher(key))
    identifier = Identifier(reopened, MODEL)
    assert identifier.identify(A, MODEL).identity_id == "a"
    revision = reopened.revision
    store.put_sample(sample(vector=B))
    assert reopened.revision > revision
    assert identifier.identify(A, MODEL).identity_id is None
    with pytest.raises(Exception):
        SQLiteStore(path, cipher=AESGCMCipher(os.urandom(32))).snapshot()
    store.delete_identity("a")
    assert reopened.snapshot()[1] == []


def test_aes_tampering_and_explicit_cipher_requirement(tmp_path):
    cipher = AESGCMCipher(os.urandom(32))
    blob = cipher.protect(b"private embedding")
    assert cipher.unprotect(blob) == b"private embedding"
    with pytest.raises(Exception):
        cipher.unprotect(blob[:-1] + bytes([blob[-1] ^ 1]))
    with pytest.raises(ValueError):
        SQLiteStore(tmp_path / "bad.db", cipher=None)


@pytest.mark.skipif(os.name != "nt", reason="DPAPI is Windows-only")
def test_dpapi_roundtrip_and_plaintext_rejection():
    cipher = DPAPICipher()
    blob = cipher.protect(b"existing-format")
    assert blob.startswith(b"DPAPI1\x00")
    assert cipher.unprotect(blob) == b"existing-format"
    with pytest.raises(ValueError):
        cipher.unprotect(b"PLAIN1\x00data")
