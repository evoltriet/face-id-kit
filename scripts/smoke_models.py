import sys
from pathlib import Path
import numpy as np
from face_id_kit.opencv import OpenCVBackend

backend = OpenCVBackend(Path(sys.argv[1]))
backend.ensure_loaded()
assert backend.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []
# Exercise alignment/embedding operators using an artificial geometric input.
import cv2
image = np.random.default_rng(4).integers(0, 255, (112, 112, 3), dtype=np.uint8)
landmarks = np.array([0, 0, 112, 112, 35, 40, 77, 40, 56, 62, 40, 82, 72, 82, .99], dtype=np.float32)
aligned = backend._recognizer.alignCrop(image, landmarks)
vector = backend._recognizer.feature(aligned).reshape(-1)
assert vector.size == 128 and np.isfinite(vector).all()
import json
reference = json.loads((Path(__file__).resolve().parents[1] / "fixtures/sface_synthetic_reference.json").read_text())
synthetic = ((np.arange(112 * 112 * 3) * 73 + 19) % 256).astype(np.uint8).reshape(112, 112, 3)
feature = backend._recognizer.feature(synthetic).reshape(-1)
np.testing.assert_allclose(feature / np.linalg.norm(feature), reference["embedding"], atol=reference["tolerance"], rtol=0)
print("Pinned YuNet detection and SFace alignment/embedding smoke tests passed.")
