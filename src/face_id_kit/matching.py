from __future__ import annotations

import threading
import uuid
import numpy as np
from .protocols import FaceBackend, GalleryStore
from .types import Candidate, Detection, EnrollmentSample, Match, MatchPolicy, ModelSpec


SCORING_VERSION = "cosine-top3-0.7max-0.3mean/unique-top-only-v1"


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("Embedding must be a finite, nonempty vector")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding norm must be positive and finite")
    return array / norm


def check_model(expected: ModelSpec, actual: ModelSpec, vector: np.ndarray) -> np.ndarray:
    if expected != actual:
        raise ValueError("Incompatible embedding model, dimension or preprocessing")
    result = normalize_embedding(vector)
    if result.size != expected.dimension:
        raise ValueError("Embedding dimension does not match its model")
    return result


def gallery_score(vector: np.ndarray, gallery: np.ndarray) -> float:
    similarities = np.sort(gallery @ normalize_embedding(vector))[::-1][:3]
    if not similarities.size:
        raise ValueError("Cannot score an empty identity gallery")
    return float(np.clip(0.7 * similarities[0] + 0.3 * similarities.mean(), -1, 1))


def resolve_unique(candidate_lists: list[list[Candidate]], policy: MatchPolicy) -> list[Match]:
    results = []
    for candidates in candidate_lists:
        ordered = sorted(candidates, key=lambda c: (-c.score, c.identity_id))
        if not ordered:
            results.append(Match(None, 0, 0, reason="empty_gallery"))
            continue
        best = ordered[0]
        margin = best.score - ordered[1].score if len(ordered) > 1 else 2.0
        reason = ("below_threshold" if best.score < policy.threshold else
                  "ambiguous" if margin < policy.margin else "matched")
        results.append(Match(best.identity_id if reason == "matched" else None,
                             best.score, margin, tuple(ordered), reason))
    used = set()
    for index in sorted(range(len(results)), key=lambda i: -results[i].score):
        result = results[index]
        if result.identity_id is None:
            continue
        if result.identity_id in used:
            # Never promote a runner-up by deleting the strongest competitor.
            results[index] = Match(None, result.score, result.margin,
                                   result.candidates, "identity_conflict")
        else:
            used.add(result.identity_id)
    return results


class Identifier:
    def __init__(self, store: GalleryStore, model: ModelSpec,
                 policy: MatchPolicy = MatchPolicy()):
        self.store, self.model, self.policy = store, model, policy
        self._revision = None
        self._gallery: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()

    def _refresh(self):
        if self._revision == self.store.revision:
            return
        revision, samples = self.store.snapshot()
        groups: dict[str, list[np.ndarray]] = {}
        for sample in samples:
            vector = check_model(self.model, sample.model, sample.embedding)
            groups.setdefault(sample.identity_id, []).append(vector)
        self._gallery = {key: np.stack(values) for key, values in groups.items()}
        self._revision = revision

    def candidates(self, embedding: np.ndarray, model: ModelSpec) -> list[Candidate]:
        vector = check_model(self.model, model, embedding)
        with self._lock:
            self._refresh()
            return sorted((Candidate(key, gallery_score(vector, values))
                           for key, values in self._gallery.items()),
                          key=lambda item: (-item.score, item.identity_id))

    def identify(self, embedding: np.ndarray, model: ModelSpec) -> Match:
        return self.identify_many([(embedding, model)])[0]

    def identify_many(self, queries: list[tuple[np.ndarray, ModelSpec]]) -> list[Match]:
        with self._lock:
            return resolve_unique([self.candidates(vector, model) for vector, model in queries], self.policy)


class FaceIdentifier:
    def __init__(self, backend: FaceBackend, store: GalleryStore,
                 policy: MatchPolicy = MatchPolicy()):
        self.backend, self.store = backend, store
        self.identifier = Identifier(store, backend.model, policy)

    def detect(self, image_bgr: np.ndarray, **options) -> list[Detection]:
        return self.backend.detect(image_bgr, **options)

    def enroll(self, identity_id: str, face: Detection, *, sample_id: str | None = None,
               source_id: str = "") -> EnrollmentSample:
        vector = check_model(self.backend.model, face.model, face.embedding)
        sample = EnrollmentSample(sample_id or str(uuid.uuid4()), identity_id, vector,
                                  face.model, face.quality, source_id)
        self.store.put_sample(sample)
        return sample

    def identify(self, image_bgr: np.ndarray, **options) -> list[tuple[Detection, Match]]:
        faces = self.detect(image_bgr, **options)
        matches = self.identifier.identify_many([(face.embedding, face.model) for face in faces])
        return list(zip(faces, matches))
