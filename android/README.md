# Face ID Kit Android
Separately buildable Kotlin modules, under the repository's MIT license. No Python runtime, app UI, names, contacts, HTTP, or wedding logic.

- `core`: pure JVM typed contracts, normalized embeddings, revisioned gallery, exact cosine matching, unique top-candidate assignment, IoU tracking, separate tuning/evaluation.
- `opencv`: Android YuNet/SFace adapter, pinned OpenCV 4.12.0 and model fingerprints. Largest usable faces, optional central filter. Caller supplies upright, unmirrored BGR bytes.
- `storage`: Room gallery and Android Keystore AES-256-GCM cipher. Explicit cipher required; no plaintext fallback. Synchronous methods require an app-owned background executor.

## Build and consume
JDK17 / Gradle8.13. Android adapters require a licensed Android SDK, platform36/build-tools35.0.0. Minimum Android29. Core alone: `gradle -p android -PcoreOnly :core:test`.

Pin this repository at an immutable commit in a consumer's Git submodule, then `includeBuild("vendor/face-id-kit/android")` in Gradle settings. Dependencies: `io.github.evoltriet.faceidkit:core:0.2.0`, `opencv:0.2.0`, `storage:0.2.0`. Maven publication is deferred.

Run the existing explicit `face-id-kit download-models --directory <assets-model-directory>` command during setup (Python needed for this helper only), or use an equivalent checksum-verified downloader. Bundle the weights for first-launch offline use and include `licenses/YUNET.txt` and `licenses/SFACE.txt`; model licenses are separate from MIT.

## SDK operations
```kotlin
val store = InMemoryStore()
val backend = OpenCvBackend(modelDirectory)
val sdk = FaceIdentifier(backend, store)
store.putIdentity(Identity(opaqueId))
val detections = sdk.detect(uprightBgrImage)
sdk.enroll(opaqueId, detections[chosenFace], sourceId = opaqueSourceGroup)
val results = sdk.identify(anotherImage)
val session = LiveSession(sdk, maxFaces = 4, centralOnly = false)
val tracks = session.update(cameraFrame)
store.deleteSample(sampleId)
```

For camera images with large close-up faces, applications can opt into a detector
pyramid without changing embedding models or the original-resolution default:

```kotlin
val cameraBackend = OpenCvBackend(modelDirectory,
    scalePolicy = DetectorScalePolicy(listOf(640, 320)))
cameraBackend.warmUp() // Call on a worker before enabling capture/recognition.
```

Each requested long edge preserves the image aspect ratio and never upscales.
Boxes and all five landmarks are mapped back to the original image, duplicate
detections are merged, and alignment/quality/embedding inference uses original
pixels. `stats` exposes non-identifying detector/eligible counts for diagnostics.
Warmup executes both detector and SFace graphs using blank, memory-only inputs.
These settings can change matching behavior even though the sample format is
compatible: require calibration review and physical camera verification. They
are not a guarantee of recognition accuracy or latency. Android tests include
an explicitly synthetic portrait at ordinary and close-up sizes; provenance and
the generation prompt are in `opencv/src/androidTest/assets/SYNTHETIC_FACE.md`.

Persistent use: `RoomGalleryStore(context, KeystoreCipher(applicationOwnedAlias))`. Apps may encrypt arbitrary metadata/crops with metadata/extra APIs; the SDK does not interpret names or contact payloads. Room revision changes invalidate galleries and temporal consensus. Apps must invalidate their calibrated policy after gallery changes.

Score = 0.7 × best cosine + 0.3 × mean of top three confirmed examples. Threshold .45 and runner-up margin .05 are **uncalibrated defaults**, not probabilities. Unique assignment only accepts top choices; identity conflicts become unknown instead of assigning a weaker runner-up.

TrackConsensus defaults to 4/6, IoU .3, and 1-second expiry. Detection list reordering does not change track ownership. Ambiguous geometric associations reset history. This is conservative tracking, not a learned multi-object tracker; fast motion may require fresh consensus.

All samples carry the same YuNet/SFace fingerprint, dimension, and preprocessing string as the Python adapter. Incompatible comparisons throw. Kotlin uses double accumulation over float32 vectors; synthetic parity tolerates 1e-6 rather than claiming bitwise equality. Changing models or preprocessing requires new enrollment/calibration review.

Calibration rejects direct overlap by ID, source group, and normalized embedding hash across enrollment/tuning/final evaluation. Supply source groups from independent capture sessions to avoid leakage; identical recaptures or adjacent frames are not valid independent evidence. Empty unknown sets never pass. Policy selection uses tuning only, live scoring is reused, and final evaluation cannot tune policy.

## Verification
### Application-owned processing boundaries

`OpenCvBackend.locate(image, regions)` runs detection only on supplied upright,
unmirrored rectangles. An empty list runs no model; null retains legacy whole-image
behavior. `embed(image, locations)` is a separate stage, bound to the exact source
image and dimensions. Alignment uses original resolution with out-of-region pixels
masked, so interpolation cannot pick up excluded pixels. Face locations include
sensitive geometry: applications must obtain appropriate permission before even
calling `locate`, and discard locations when their authorization changes.

`LiveSession.updateDetections` accepts region-scoped detections with the same
matching, unique assignment, and consensus behavior as the default pipeline.
Applications own region approvals, tracking-loss rules, and consent—not the SDK.

`RoomGalleryStore.snapshot(identityIds)` selects rows in SQL before decrypting.
An empty set decrypts no samples; null retains the default complete gallery.
`sampleDescriptors()` reads IDs and quality without decrypting embeddings.
Eligibility adapters must expose a changed revision and call `invalidateRevision`
when eligibility changes, including expiry. Metadata-only writes can explicitly
set `affectsGallery=false`; existing calls retain their original revision behavior.

Shared `fixtures/parity.json` and `fixtures/workflow_parity.json` cover decisions, numeric scores, identity conflicts, revision/cache changes, reordering/crossings/expiry, and calibration policy/separation. Python CI runs Windows/Linux; Kotlin core can run without Android. Android instrumentation checks actual model execution and encrypted Room/Keystore persistence using synthetic data. No personal images or enrollment databases are committed.

`bash android/ci-emulator.sh <test command>` requires a pre-licensed SDK and never automatically accepts terms. It creates a disposable AVD in RUNNER_TEMP (or /tmp), bounds startup, and cleans up its emulator process. Camera performance and native alignment must also be verified by each application on its target hardware.
