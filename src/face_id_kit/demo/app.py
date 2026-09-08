from __future__ import annotations

import asyncio
import base64
import hashlib
import io
from pathlib import Path
import threading
import time
import uuid
from dataclasses import asdict

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageOps

from ..matching import FaceIdentifier
from ..live import LiveSession
from ..types import Identity

MAX_IMAGE_BYTES = 8_000_000


def decode(data: bytes):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Choose an image smaller than 8 MB")
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > 20_000_000:
            raise ValueError("Image exceeds 20 megapixels; resize it before upload")
        oriented = ImageOps.exif_transpose(source).convert("RGB")
        oriented.thumbnail((1600, 1600))
        return np.asarray(oriented)[:, :, ::-1].copy()


class EnrollmentRequest(BaseModel):
    token: str
    face_index: int = Field(ge=0)
    name: str = Field(default="", max_length=120)
    identity_id: str | None = None


def create_app(faces: FaceIdentifier) -> FastAPI:
    app = FastAPI(title="Face ID Kit demo")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])
    staged: dict[str, tuple[float, str, list]] = {}
    stage_lock = threading.Lock()

    def origin_allowed(headers):
        origin = headers.get("origin")
        return (not origin or origin in (f"http://{headers.get('host')}", f"https://{headers.get('host')}")) and headers.get("sec-fetch-site") != "cross-site"

    @app.middleware("http")
    async def local_requests(request, call_next):
        if not origin_allowed(request.headers):
            return JSONResponse({"detail": "Use the demo from its localhost page"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).with_name("index.html"))

    @app.get("/api/identities")
    def identities():
        return [{"id": i.id, "name": i.metadata.get("name", i.id)} for i in faces.store.identities()]

    @app.delete("/api/identities/{identity_id}")
    def remove(identity_id: str):
        faces.store.delete_identity(identity_id)
        return {"deleted": True}

    @app.post("/api/photo")
    async def photo(request: Request):
        # Raw image body: multipart UploadFile can spool photographs to disk.
        buffer = bytearray()
        async for chunk in request.stream():
            if len(buffer) + len(chunk) > MAX_IMAGE_BYTES:
                raise HTTPException(413, "Choose an image smaller than 8 MB")
            buffer.extend(chunk)
        data = bytes(buffer)
        try:
            image = await asyncio.to_thread(decode, data)
            detections = await asyncio.to_thread(faces.detect, image, max_faces=8)
            matches = await asyncio.to_thread(faces.identifier.identify_many, [(f.embedding, f.model) for f in detections])
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error
        token, now = str(uuid.uuid4()), time.monotonic()
        with stage_lock:
            for key in list(staged):
                if now - staged[key][0] > 180:
                    staged.pop(key)
            while len(staged) >= 32:
                staged.pop(next(iter(staged)))
            staged[token] = (now, hashlib.sha256(data).hexdigest(), detections)
        names = {i.id: i.metadata.get("name", i.id) for i in faces.store.identities()}
        results = []
        for face, match in zip(detections, matches):
            crop = io.BytesIO()
            if face.aligned is not None:
                Image.fromarray(face.aligned[:, :, ::-1]).save(crop, "JPEG")
            results.append({"bbox": face.bbox, "quality": face.quality,
                            "match": asdict(match), "name": names.get(match.identity_id),
                            "crop": "data:image/jpeg;base64," + base64.b64encode(crop.getvalue()).decode()})
        return {"token": token, "faces": results}

    @app.post("/api/enroll")
    def enroll(request: EnrollmentRequest):
        with stage_lock:
            entry = staged.get(request.token)
            if not entry or time.monotonic() - entry[0] > 180:
                raise HTTPException(410, "Capture expired; take another photo")
            _, source, detections = entry
            if request.face_index >= len(detections):
                raise HTTPException(422, "Select a detected face")
            face = detections[request.face_index]
        iid = request.identity_id
        if iid is None:
            name = request.name.strip()
            if not name:
                raise HTTPException(422, "Enter a name")
            iid = str(uuid.uuid4())
            faces.store.put_identity(Identity(iid, {"name": name}))
        elif iid not in {i.id for i in faces.store.identities()}:
            raise HTTPException(404, "Identity no longer exists")
        faces.enroll(iid, face, sample_id=f"{iid}:{source}:{request.face_index}", source_id=source)
        return {"identity_id": iid, "enrolled": True}

    @app.websocket("/api/live")
    async def live(socket: WebSocket):
        if not origin_allowed(socket.headers):
            await socket.close(code=1008)
            return
        await socket.accept()
        session = LiveSession(faces)
        try:
            while True:
                data = await socket.receive_bytes()
                try:
                    image = await asyncio.to_thread(decode, data)
                    results = await asyncio.to_thread(session.update, image)
                    names = {i.id: i.metadata.get("name", i.id) for i in faces.store.identities()}
                    await socket.send_json({"width": image.shape[1], "height": image.shape[0], "faces": [
                        {"track_id": r.track_id, "bbox": r.detection.bbox, "stable": r.stable,
                         "identity_id": r.match.identity_id if r.stable else None,
                         "name": names.get(r.match.identity_id) if r.stable else None,
                         "score": r.match.score, "reason": r.match.reason} for r in results]})
                except (ValueError, OSError) as error:
                    session.tracker.reset()
                    await socket.send_json({"faces": [], "error": str(error)})
        except WebSocketDisconnect:
            pass
    return app
