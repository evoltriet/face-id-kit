package io.github.evoltriet.faceidkit

data class ModelSpec(val fingerprint: String, val dimension: Int, val preprocessing: String) {
    init { require(fingerprint.isNotBlank() && dimension > 0 && preprocessing.isNotBlank()) }
}
data class Identity(val id: String)
data class EnrollmentSample(val id: String, val identityId: String, val embedding: FloatArray,
    val model: ModelSpec, val quality: Double = 1.0, val sourceId: String = "")
data class Box(val x: Double, val y: Double, val width: Double, val height: Double) {
    fun iou(other: Box): Double {
        val area = (minOf(x + width, other.x + other.width) - maxOf(x, other.x)).coerceAtLeast(0.0) *
            (minOf(y + height, other.y + other.height) - maxOf(y, other.y)).coerceAtLeast(0.0)
        val union = width * height + other.width * other.height - area
        return if (union > 0) area / union else 0.0
    }
}
/** Images are caller-owned, upright, unmirrored BGR bytes; no Android dependency in core. */
data class BgrImage(val width: Int, val height: Int, val pixels: ByteArray) {
    init { require(width > 0 && height > 0 && width.toLong() * height * 3 == pixels.size.toLong()) }
}
data class Detection(val box: Box, val embedding: FloatArray, val model: ModelSpec,
    val confidence: Double, val quality: Double, val aligned: BgrImage? = null)
data class Candidate(val identityId: String, val score: Double)
data class Match(val identityId: String?, val score: Double, val margin: Double,
    val candidates: List<Candidate> = emptyList(), val reason: String = "unknown")
data class MatchPolicy(val threshold: Double = .45, val margin: Double = .05) {
    init { require(threshold in -1.0..1.0 && margin in 0.0..2.0) }
}
data class ConsensusDiagnostics(val agreeing: Int = 0, val observations: Int = 0,
    val required: Int = 4, val window: Int = 6, val resetReason: String? = null)
data class LiveResult(val trackId: Long, val detection: Detection, val match: Match, val stable: Boolean,
    val consensus: ConsensusDiagnostics = ConsensusDiagnostics())
data class GallerySnapshot(val revision: Long, val samples: List<EnrollmentSample>)
interface FaceBackend { val model: ModelSpec; fun detect(image: BgrImage, maxFaces: Int? = null, centralOnly: Boolean = false): List<Detection> }
interface EmbeddingCipher { fun protect(data: ByteArray): ByteArray; fun unprotect(data: ByteArray): ByteArray }
interface GalleryStore {
    val revision: Long
    fun snapshot(): GallerySnapshot
    fun identities(): List<Identity>
    fun putIdentity(identity: Identity)
    fun putSample(sample: EnrollmentSample)
    fun deleteSample(sampleId: String)
    fun deleteIdentity(identityId: String)
}
object SFace {
    const val DETECTOR = "face_detection_yunet_2023mar.onnx"
    const val RECOGNIZER = "face_recognition_sface_2021dec.onnx"
    const val DETECTOR_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    const val RECOGNIZER_SHA = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
    val model = ModelSpec("yunet:$DETECTOR_SHA/sface:$RECOGNIZER_SHA", 128, "opencv-sface-alignCrop-bgr-l2-v1")
}
