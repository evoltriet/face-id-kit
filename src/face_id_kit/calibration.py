from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from .matching import Identifier, MatchPolicy, SCORING_VERSION, normalize_embedding
from .stores import InMemoryStore
from .types import EnrollmentSample, Identity, ModelSpec, ValidationSample


@dataclass(frozen=True)
class Evaluation:
    known_accuracy: float
    known_tests: int
    unknown_tests: int
    false_accepts: int


@dataclass(frozen=True)
class CalibrationReport:
    policy: MatchPolicy
    tuning: Evaluation
    evaluation: Evaluation
    passed: bool
    reason: str
    scoring_version: str = SCORING_VERSION


def _keys(sample):
    # Reject reused IDs, source-photo groups and exact duplicate embeddings across splits.
    vector = normalize_embedding(sample.embedding).astype("<f4")
    keys = {("id", sample.id), ("embedding", hashlib.sha256(vector.tobytes()).hexdigest())}
    if sample.source_id:
        keys.add(("source", sample.source_id))
    return keys


def _check_disjoint(groups):
    previous = set()
    for group in groups:
        current = set()
        for sample in group:
            if _keys(sample) & previous:
                raise ValueError("Enrollment, tuning and evaluation must use disjoint samples and sources")
            current.update(_keys(sample))
        previous.update(current)


def calibrate(enrollment: list[EnrollmentSample], tuning: list[ValidationSample],
              evaluation: list[ValidationSample], model: ModelSpec, *,
              min_accuracy: float = .95, min_known: int = 10, min_unknown: int = 5,
              thresholds=None, margins=None) -> CalibrationReport:
    if min_known < 1 or min_unknown < 1 or not 0 < min_accuracy <= 1:
        raise ValueError("Calibration requires positive known/unknown sample minimums and accuracy in (0, 1]")
    _check_disjoint((enrollment, tuning, evaluation))
    store = InMemoryStore()
    for iid in sorted({s.identity_id for s in enrollment}):
        store.put_identity(Identity(iid))
    for sample in enrollment:
        store.put_sample(sample)
    identifier = Identifier(store, model)
    identities = {identity.id for identity in store.identities()}
    for sample in [*tuning, *evaluation]:
        if sample.identity_id is not None and sample.identity_id not in identities:
            raise ValueError("Known validation identity has no enrollment")
    # Scoring is shared with live identification; final evaluation never chooses parameters.
    def score_samples(samples):
        return [(sample, identifier.candidates(sample.embedding, sample.model)) for sample in samples]
    from .matching import resolve_unique
    def measure(scored, policy):
        known = unknown = correct = false = 0
        for sample, candidates in scored:
            match = resolve_unique([candidates], policy)[0]
            if sample.identity_id is None:
                unknown += 1
                false += match.identity_id is not None
            else:
                known += 1
                correct += match.identity_id == sample.identity_id
        return Evaluation(correct / known if known else 0., known, unknown, false)
    tuned, held_out = score_samples(tuning), score_samples(evaluation)
    best = None
    for threshold in thresholds if thresholds is not None else np.arange(.35, .751, .01):
        for margin in margins if margins is not None else np.arange(.02, .151, .01):
            policy = MatchPolicy(round(float(threshold), 4), round(float(margin), 4))
            metrics = measure(tuned, policy)
            rank = (-metrics.false_accepts, metrics.known_accuracy, policy.threshold, policy.margin)
            if best is None or rank > best[0]:
                best = (rank, policy, metrics)
    if best is None:
        raise ValueError("Threshold and margin grids must not be empty")
    _, policy, tuning_metrics = best
    final_metrics = measure(held_out, policy)
    sufficient = (len(identities) >= 2 and
                  all(m.known_tests >= min_known and m.unknown_tests >= min_unknown for m in (tuning_metrics, final_metrics)))
    passed = sufficient and all(m.known_accuracy >= min_accuracy and m.false_accepts == 0 for m in (tuning_metrics, final_metrics))
    reason = ("Calibration passed on separate tuning and evaluation sets." if passed else
              "Insufficient known/unknown validation samples in separate tuning and evaluation sets." if not sufficient else
              "Held-out accuracy or unknown rejection failed; add representative samples and recalibrate.")
    return CalibrationReport(policy, tuning_metrics, final_metrics, passed, reason)
