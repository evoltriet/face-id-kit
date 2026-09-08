import json
from pathlib import Path
import numpy as np
import pytest
from face_id_kit import ModelSpec, Identity, EnrollmentSample, InMemoryStore, Identifier
from face_id_kit.live import TrackConsensus
from face_id_kit.calibration import calibrate
from face_id_kit.types import ValidationSample

DATA = json.loads((Path(__file__).resolve().parents[1] / "fixtures/workflow_parity.json").read_text())
MODEL = ModelSpec("synthetic-v1", 4, "test-l2")

def test_shared_tracking():
    tracker = TrackConsensus()
    for row in DATA["tracking"]:
        result = tracker.update(row["boxes"], row["identities"], now=row["now"])
        assert [r[0] for r in result] == row["tracks"]
        assert [r[1] for r in result] == row["stable"]

def test_shared_scores_revisions_and_cache():
    fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/parity.json").read_text())
    store = InMemoryStore()
    for s in fixture["samples"]:
        store.put_identity(Identity(s["identityId"]))
        store.put_sample(EnrollmentSample(s["id"], s["identityId"], np.array(s["embedding"]), MODEL))
    engine = Identifier(store, MODEL)
    for case, scores in zip(fixture["cases"], DATA["scores"]):
        results = engine.identify_many([(np.array(v), MODEL) for v in case["queries"]])
        for result, expected in zip(results, scores):
            assert (result.score, result.margin) == pytest.approx(expected, abs=1e-6)
    fresh = InMemoryStore()
    revisions = [fresh.revision]
    fresh.put_identity(Identity("a")); revisions.append(fresh.revision)
    fresh.put_identity(Identity("b")); revisions.append(fresh.revision)
    vector = np.array([1.,0.,0.,0.])
    fresh.put_sample(EnrollmentSample("s","a",vector,MODEL)); revisions.append(fresh.revision)
    matching = Identifier(fresh, MODEL)
    assert matching.identify(vector, MODEL).identity_id == "a"
    fresh.put_sample(EnrollmentSample("s","b",vector,MODEL)); revisions.append(fresh.revision)
    assert matching.identify(vector, MODEL).identity_id == "b"
    fresh.delete_sample("s"); revisions.append(fresh.revision)
    assert matching.identify(vector, MODEL).identity_id is None
    fresh.delete_identity("b"); revisions.append(fresh.revision)
    assert revisions == DATA["revisions"]

def test_shared_calibration_and_separation():
    def vector(i):
        result = np.zeros(4, dtype=np.float32); result[i]=1; return result
    enrollment = [EnrollmentSample("enroll-"+id,id,vector(i),MODEL,source_id="source-"+id) for i,id in enumerate(("a","b"))]
    def samples(prefix,offset):
        result=[]
        for i in range(15):
            v=vector(i%2 if i < 10 else 2); v[3]=offset+i*.001
            result.append(ValidationSample(f"{prefix}-{i}", ("a" if i%2==0 else "b") if i < 10 else None,v,MODEL,source_id=f"{prefix}-source-{i}"))
        return result
    tuning, evaluation = samples("t",.01), samples("e",.04)
    report=calibrate(enrollment,tuning,evaluation,MODEL)
    expected=DATA["calibration"]
    assert report.passed == expected["passed"]
    assert (report.policy.threshold,report.policy.margin) == (expected["threshold"],expected["margin"])
    assert (report.evaluation.known_accuracy,report.evaluation.known_tests,report.evaluation.unknown_tests,report.evaluation.false_accepts) == (1,10,5,0)
    assert not calibrate(enrollment,tuning,[v for v in evaluation if v.identity_id is not None],MODEL).passed
    with pytest.raises(ValueError):
        calibrate(enrollment,tuning,tuning,MODEL)

