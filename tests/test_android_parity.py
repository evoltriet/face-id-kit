import json
from pathlib import Path
import numpy as np
from face_id_kit import ModelSpec, Identity, EnrollmentSample, InMemoryStore, Identifier


def test_shared_android_decision_fixtures():
    fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/parity.json").read_text())
    model = ModelSpec(**fixture["model"])
    store = InMemoryStore()
    for sample in fixture["samples"]:
        store.put_identity(Identity(sample["identityId"]))
        store.put_sample(EnrollmentSample(sample["id"], sample["identityId"], np.array(sample["embedding"]), model))
    engine = Identifier(store, model)
    for case in fixture["cases"]:
        results = engine.identify_many([(np.array(v), model) for v in case["queries"]])
        assert [r.identity_id for r in results] == case["identities"]
        assert [r.reason for r in results] == case["reasons"]
