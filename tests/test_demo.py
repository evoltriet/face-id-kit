import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from face_id_kit import Detection, FaceIdentifier, InMemoryStore, ModelSpec
from face_id_kit.demo.app import create_app


class SyntheticBackend:
    model = ModelSpec("demo-test", 3, "test")
    def detect(self, image, **options):
        return [Detection((10., 10., 40., 40.), np.array([1., 0., 0.]),
                          self.model, .99, .9, np.zeros((40, 40, 3), dtype=np.uint8))]


def test_explicit_enrollment_photo_live_unknowns_and_delete():
    store = InMemoryStore()
    client = TestClient(create_app(FaceIdentifier(SyntheticBackend(), store)))
    stream = io.BytesIO()
    Image.new("RGB", (100, 100), "navy").save(stream, format="JPEG")
    data = stream.getvalue()
    photo = client.post("/api/photo", content=data, headers={"Content-Type": "image/jpeg"}).json()
    assert photo["faces"][0]["name"] is None
    assert not store.snapshot()[1]
    enrolled = client.post("/api/enroll", json={"token": photo["token"], "face_index": 0, "name": "Demo person"})
    assert enrolled.status_code == 200
    iid = enrolled.json()["identity_id"]
    assert client.post("/api/photo", content=data).json()["faces"][0]["name"] == "Demo person"
    count = len(store.snapshot()[1])
    with client.websocket_connect("/api/live") as ws:
        ws.send_bytes(b"invalid")
        assert ws.receive_json()["faces"] == []
        for _ in range(4):
            ws.send_bytes(data)
            result = ws.receive_json()
        assert result["faces"][0]["stable"]
        assert result["faces"][0]["name"] == "Demo person"
    assert len(store.snapshot()[1]) == count
    assert client.delete("/api/identities/" + iid).status_code == 200
    assert not store.snapshot()[1]


def test_local_origin_enforcement_and_invalid_capture():
    client = TestClient(create_app(FaceIdentifier(SyntheticBackend(), InMemoryStore())))
    assert client.get("/", headers={"Origin": "https://unrelated.example"}).status_code == 403
    assert client.post("/api/enroll", json={"token": "expired", "face_index": 0, "name": "A"}).status_code == 410
    assert client.post("/api/photo", content=b"bad").status_code == 422
    assert client.post("/api/photo", content=b"a" * 8_000_001).status_code == 413
