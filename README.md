# Face ID Kit

A local-first framework for 1:N face identification. Enroll identities, recognize faces in photos and live video, and integrate your own models, storage, and application logic.

**Contract:** an input face + your enrolled identity gallery → an identity ID or unknown, with similarity, rejection reason, and live stability information.

Version 0.1 is a Python SDK with an OpenCV backend and a localhost browser demo. Native mobile and glasses SDKs are future work. No cloud service, vector database, model training, or API key is required for recognition.

## Quick start

Python 3.11 or newer:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install ".[demo]"
face-id-kit download-models --directory .local-data/models
face-id-kit demo --models .local-data/models
```

Open http://127.0.0.1:8766. Capture or upload a photo, select a detected face, enter a name, and enroll it. Recognize a different photo or enable live recognition. Add several varied photos to the same identity.

The default demo gallery is in memory and disappears on exit. Windows persistent storage:

```sh
face-id-kit demo --models .local-data/models --database .local-data/gallery.db --cipher dpapi
```

For other platforms use `--cipher aesgcm` and supply `FACE_ID_KIT_KEY` as a base64-encoded random 32-byte key through your own secret manager/environment. Keep that same key to reopen the database. The SDK never generates or saves it for you.

Only explicit enrollments are persisted. Enrollment captures are temporarily held in RAM for three minutes; live frames are never written to disk. SQLite embedding blobs are encrypted; identity metadata and model descriptors are plaintext. The demo binds to loopback and is intended for a local operator.

## Python API

```python
from pathlib import Path
import cv2
from face_id_kit import FaceIdentifier, Identity, InMemoryStore
from face_id_kit.opencv import OpenCVBackend

store = InMemoryStore()
store.put_identity(Identity("app-person-123", {"name": "Alex"}))
faces = FaceIdentifier(OpenCVBackend(Path(".local-data/models")), store)

detected = faces.detect(cv2.imread("enrollment.jpg"))
# Your UI must let the user choose which detected face to enroll.
faces.enroll("app-person-123", detected[0], source_id="enrollment-photo-1")

for face, match in faces.identify(cv2.imread("another-photo.jpg")):
    print(match.identity_id, match.score, match.reason)
```

`FaceIdentifier.enroll` accepts `sample_id` to replace an existing sample. Stores expose `delete_sample` and `delete_identity`. Renaming identity metadata does not change its ID. Store revisions refresh cached galleries after edits.

Use `LiveSession(faces).update(bgr_frame)` from `face_id_kit.live` for tracked results. Defaults: at most four faces, no central-frame restriction, four agreeing observations in a six-observation window, and one-second track expiry. Ambiguous geometric associations start new tracks. Applications should display names only for stable accepted matches.

## What is reusable

| Layer | Responsibility |
| --- | --- |
| Core | Typed records, normalized embeddings, exact cosine lookup, score/margin rejection, unique assignments |
| Live | Spatial tracking and temporal consensus |
| Calibration | Separate enrollment, threshold-tuning, and final evaluation data |
| Backends | YuNet/SFace inference; replaceable through FaceBackend |
| Storage | In-memory or encrypted SQLite; replaceable through GalleryStore and EmbeddingCipher |
| Optional clustering | DBSCAN helper for human-reviewed enrollment groups |
| Demo | Camera/photo transport and display-name UI outside the SDK core |

The framework treats identities as opaque strings. Guest lists, contacts, household relationships, collages, permissions, and product-specific decisions belong to applications.

## Matching and calibration

Similarity scores are **not probabilities**. The default cosine threshold 0.45 and margin 0.05 are uncalibrated starting values, not accuracy claims. Each identity is scored using 70% best similarity plus 30% mean of its top three similarities. When faces compete for one identity, the weaker face becomes unknown; removing a competitor never turns an ambiguous runner-up into a match.

See [calibration and architecture](docs/architecture.md) for data splits and adapter contracts. Never treat a benchmark on a different dataset as an application accuracy guarantee.

Model descriptors include the weights' SHA-256 fingerprints, embedding dimension, and preprocessing version. Incompatible comparisons fail explicitly. Changing backend, preprocessing, gallery policy, or enrolled samples requires new validation.

## Installation extras and tests

- Base: `pip install .` (NumPy only)
- OpenCV: `pip install ".[opencv]"`
- AES-GCM storage: `pip install ".[storage]"` (SQLite is in Python's standard library)
- Clustering: `pip install ".[clustering]"`
- Demo: `pip install ".[demo]"`
- Development: `pip install ".[demo,clustering,test]"`; run `python -m pytest`

Core and demo tests use synthetic inputs. CI runs on Windows and Ubuntu. Model smoke tests run separately with checksum-verified downloads. Once dependencies and models are installed, inference/enrollment/lookup operate offline.

## License and model attribution

Framework source: [MIT](LICENSE). Upstream model notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). YuNet and SFace remain third-party models under their own terms. Model weights and personal datasets are not committed.

This is an identification toolkit for voluntarily enrolled identities. It does not provide liveness detection or authentication assurance.
