# Architecture and integration

The dependency direction is application → optional adapters → SDK core. Importing `face_id_kit` does not load OpenCV, scikit-learn, FastAPI, or an encryption library.

## Contracts

- `FaceBackend`: model descriptor and `detect(uint8_bgr_image, max_faces, central_only)`. Each detection contains pixel bbox, aligned crop when available, confidence, quality, and embedding.
- `GalleryStore`: identity/sample mutation, atomic `snapshot() -> (revision, samples)`, and a monotonic revision. Return independent snapshots; an update or reassignment of an existing sample must advance revision.
- `EmbeddingCipher`: `protect(bytes)` and `unprotect(bytes)`. SQLite requires one explicitly. No production plaintext implementation is supplied.
- `Identifier`: embedding-only lookup; `FaceIdentifier`: inference plus enrollment/lookup; `LiveSession`: per-stream tracking state.
- `Match`: identity or null, similarity, runner-up margin, ordered candidates and rejection reason. No candidate is an accepted identity unless `identity_id` is non-null.
- `Identity.metadata`: application-owned JSON values; it does not influence matching.

Core defaults serve small galleries (hundreds of identities). Exact search has linear cost in enrolled samples. Larger galleries and alternate indexing are adapter work; no large-scale performance claims are made.

## Model identity

`ModelSpec` equality includes weights fingerprint, dimension and preprocessing identifier. Enrollment validates finite nonzero vectors and dimensions. Lookup rejects a store containing incompatible samples, even if vector lengths happen to agree.

The initial backend uses checksum-pinned YuNet 2023mar and SFace 2021dec, BGR alignment via OpenCV alignCrop, and L2-normalized 128-dimensional embeddings. Downloading is an explicit command; imports and inference do not access the network.

## Calibration

```python
from face_id_kit.calibration import calibrate

report = calibrate(
    enrollment=enrollment_samples,   # EnrollmentSample records
    tuning=tuning_samples,            # ValidationSample records
    evaluation=evaluation_samples,    # ValidationSample records
    model=model_spec,
)
if report.passed:
    identifier.policy = report.policy
```

Known validation samples carry their true identity ID; unknown samples carry null. Provide at least two enrolled identities, ten known examples and five unknown examples in **each** validation partition by default. Insufficient data yields a failed report, including when there are zero unknown examples.

Threshold and margin selection uses tuning data only. Final evaluation is measured once with the selected policy. Pass requires at least 95% known accuracy and zero false acceptances in both partitions. These are empirical checks, not guarantees outside that validation set.

Sample IDs, exact duplicate embeddings and source IDs cannot overlap partitions. Set `source_id` to a capture session or source-photo group so related crops cannot leak across partitions. The caller must also group near-duplicates and adjacent video frames; the SDK cannot infer their provenance. Maintain a separate untouched evaluation set when iterating on models or collection practices.

Live and calibration use the same scoring implementation and enrollment bank. Do not add validation samples to that bank and then keep citing the old report.

## Tracking

Association uses bbox overlap, independent of detector output ordering. Ambiguous many-to-one/one-to-many overlaps discard those track histories; a changed identity resets history. Missing observations decay agreement and tracks expire after a configurable timeout. This conservative tracker may reset when faces cross or move rapidly; it is not an occlusion/reidentification model.

## Persistence

SQLite schema version 1 stores application metadata, model descriptors, source groups and encrypted float32 embeddings. Triggers increment revision for all gallery edits. Snapshots read data and revision in one transaction. Both AES-GCM (caller-owned 256-bit key) and Windows DPAPI are provided. Lost keys/accounts prevent decryption.

DPAPI uses the `DPAPI1\0` wire prefix and reads existing blobs protected for the current user. It rejects plaintext records. Embedding encryption does not encrypt identity metadata or guarantee removal from filesystem backups.

## Demo endpoints

- `POST /api/photo`: raw image bytes (not multipart), transient token and detected face crops/matches. The raw-body handler avoids multipart temporary-file spooling.
- `POST /api/enroll`: token, face index, new name or existing identity ID.
- `GET/DELETE /api/identities[/id]`: list/remove demo identities and their samples.
- `WS /api/live`: binary compressed images in; dimensions and tracked face results out.

These are demo interfaces, not required SDK transports. Device contacts and native camera APIs can call equivalent SDK contracts through their own adapters.

## Existing application adoption

Keep application tables and IDs in place. Implement GalleryStore over existing rows, attach an explicit descriptor to legacy embeddings, and preserve the existing encryption context. Use a revision token that changes on relabeling, deactivation and replacement, not just row counts. Pin the SDK to an immutable Git revision or package release.

Existing recognition thresholds must be reviewed if matching/tracking/gallery policy changes. Camera UI timing, household logic, message generation and media selection remain application behavior.
