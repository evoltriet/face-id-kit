from dataclasses import replace
import numpy as np
import pytest
from face_id_kit import EnrollmentSample, ModelSpec, ValidationSample
from face_id_kit.calibration import calibrate

MODEL = ModelSpec("synthetic-calibration", 16, "l2")


def dataset():
    rng = np.random.default_rng(13)
    def vector(axis):
        v = rng.normal(0, .01, 16)
        v[axis] += 1
        return v
    enrollment = [EnrollmentSample(f"enroll-{i}", f"id-{i % 2}", vector(i % 2), MODEL, source_id=f"enroll-photo-{i}") for i in range(6)]
    def validation(prefix):
        return [ValidationSample(f"{prefix}-{i}", f"id-{i % 2}" if i < 12 else None,
                                 vector(i % 2 if i < 12 else 2), MODEL, f"{prefix}-photo-{i}") for i in range(18)]
    return enrollment, validation("tuning"), validation("evaluation")


def test_separate_sets_pass_with_same_live_scoring():
    enroll, tune, evaluate = dataset()
    report = calibrate(enroll, tune, evaluate, MODEL)
    assert report.passed
    assert report.evaluation.known_accuracy == 1
    assert report.evaluation.false_accepts == 0


def test_empty_unknown_data_cannot_pass():
    enroll, tune, evaluate = dataset()
    report = calibrate(enroll, [s for s in tune if s.identity_id], [s for s in evaluate if s.identity_id], MODEL)
    assert not report.passed
    assert "Insufficient" in report.reason


@pytest.mark.parametrize("overlap", ["id", "source", "embedding"])
def test_leakage_is_rejected(overlap):
    enroll, tune, evaluate = dataset()
    changes = {"id": enroll[0].id} if overlap == "id" else {"source_id": enroll[0].source_id} if overlap == "source" else {"embedding": enroll[0].embedding}
    evaluate[0] = replace(evaluate[0], **changes)
    with pytest.raises(ValueError, match="disjoint"):
        calibrate(enroll, tune, evaluate, MODEL)


def test_evaluation_does_not_tune_thresholds():
    enroll, tune, evaluate = dataset()
    before = calibrate(enroll, tune, evaluate, MODEL)
    changed = [replace(s, identity_id="id-1" if s.identity_id == "id-0" else "id-0") if s.identity_id else s for s in evaluate]
    after = calibrate(enroll, tune, changed, MODEL)
    assert before.policy == after.policy
    assert not after.passed
