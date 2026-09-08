from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request
from .types import ModelSpec

MODEL_FILES = {
    "face_detection_yunet_2023mar.onnx": (
        "face_detection_yunet", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_recognition_sface_2021dec.onnx": (
        "face_recognition_sface", "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}
SFACE_MODEL = ModelSpec(
    fingerprint="yunet:" + MODEL_FILES["face_detection_yunet_2023mar.onnx"][1] +
                "/sface:" + MODEL_FILES["face_recognition_sface_2021dec.onnx"][1],
    dimension=128,
    preprocessing="opencv-sface-alignCrop-bgr-l2-v1",
)


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_models(directory: Path):
    for name, (_, expected) in MODEL_FILES.items():
        path = Path(directory) / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}; run face-id-kit download-models --directory <path>")
        if sha256(path) != expected:
            raise ValueError(f"Model checksum mismatch: {name}")


def download_models(directory: Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (folder, expected) in MODEL_FILES.items():
        destination = directory / name
        if destination.exists():
            if sha256(destination) == expected:
                continue
            raise ValueError(f"Existing model checksum mismatch: {destination}; move it aside before retrying")
        url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{folder}/{name}"
        fd, temporary = tempfile.mkstemp(prefix="model-", suffix=".download", dir=directory)
        os.close(fd)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if sha256(Path(temporary)) != expected:
                raise ValueError(f"Downloaded checksum mismatch for {name}")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
