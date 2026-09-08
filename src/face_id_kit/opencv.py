from __future__ import annotations

import math
import threading
from pathlib import Path
import numpy as np
from .matching import normalize_embedding
from .models import MODEL_FILES, SFACE_MODEL, verify_models
from .types import Detection


class OpenCVBackend:
    model = SFACE_MODEL

    def __init__(self, models_dir: Path, *, detector_threshold: float = 0.85,
                 min_face_size: int = 42):
        self.models_dir = Path(models_dir)
        self.detector_threshold, self.min_face_size = detector_threshold, min_face_size
        self._detector = self._recognizer = None
        self._lock = threading.RLock()

    @property
    def available(self):
        return all((self.models_dir / name).is_file() for name in MODEL_FILES)

    def ensure_loaded(self):
        import cv2
        with self._lock:
            if self._detector is not None:
                return
            verify_models(self.models_dir)
            detector = cv2.FaceDetectorYN.create(
                str(self.models_dir / "face_detection_yunet_2023mar.onnx"),
                "", (320, 320), self.detector_threshold, 0.3, 5000)
            recognizer = cv2.FaceRecognizerSF.create(
                str(self.models_dir / "face_recognition_sface_2021dec.onnx"), "")
            self._detector, self._recognizer = detector, recognizer

    def detect(self, image_bgr: np.ndarray, *, max_faces: int | None = None,
               central_only: bool = False) -> list[Detection]:
        import cv2
        if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
            raise ValueError("Supply a nonempty uint8 BGR image")
        if max_faces is not None and max_faces < 1:
            raise ValueError("max_faces must be positive or None")
        self.ensure_loaded()
        height, width = image_bgr.shape[:2]
        with self._lock:
            self._detector.setInputSize((width, height))
            _, rows = self._detector.detect(image_bgr)
            if rows is None:
                return []
            eligible = []
            for row in rows:
                x, y, w, h = map(float, row[:4])
                if min(w, h) < self.min_face_size:
                    continue
                if central_only and not (width * .12 <= x + w / 2 <= width * .88 and height * .08 <= y + h / 2 <= height * .92):
                    continue
                eligible.append(row)
            eligible.sort(key=lambda row: float(row[2] * row[3]), reverse=True)
            if max_faces is not None:
                eligible = eligible[:max_faces]
            results = []
            for row in eligible:
                x, y, w, h = map(float, row[:4])
                aligned = self._recognizer.alignCrop(image_bgr, row)
                vector = normalize_embedding(self._recognizer.feature(aligned).reshape(-1))
                area = min(1., w * h / max(1., width * height * .08))
                blur = float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                clarity = min(1., math.log1p(max(0., blur)) / math.log1p(250.))
                confidence = float(row[14])
                results.append(Detection((x, y, w, h), vector, self.model, confidence,
                                         min(1., confidence * (.5 + .25 * area + .25 * clarity)), aligned))
        return sorted(results, key=lambda face: face.bbox[0] + face.bbox[2] / 2)
