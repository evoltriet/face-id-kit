"""Real localhost transport smoke test; images and identities are synthetic."""
import io
import socket
import threading
import time
import httpx
import uvicorn
from PIL import Image
from websockets.sync.client import connect
from face_id_kit import FaceIdentifier, InMemoryStore
from face_id_kit.demo.app import create_app
from test_demo import SyntheticBackend


def test_http_and_websocket_server():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(FaceIdentifier(SyntheticBackend(), InMemoryStore())),
                                          log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(.02)
        assert server.started
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
            assert "Face ID Kit" in client.get("/").text
            stream = io.BytesIO()
            Image.new("RGB", (100, 100), "navy").save(stream, format="JPEG")
            data = stream.getvalue()
            photo = client.post("/api/photo", content=data).json()
            assert client.post("/api/enroll", json={"token": photo["token"], "face_index": 0,
                                                   "name": "Synthetic identity"}).status_code == 200
        import json
        with connect(f"ws://127.0.0.1:{port}/api/live", proxy=None) as ws:
            for _ in range(4):
                ws.send(data)
                result = json.loads(ws.recv(timeout=5))
            assert result["faces"][0]["name"] == "Synthetic identity"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive()
