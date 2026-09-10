package io.github.evoltriet.faceidkit.opencv

import io.github.evoltriet.faceidkit.*
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.FaceDetectorYN
import org.opencv.objdetect.FaceRecognizerSF
import java.io.File
import java.security.MessageDigest
import kotlin.math.ln1p

/** Empty edges preserves the legacy original-resolution detector. Alignment always uses original pixels. */
data class DetectorScalePolicy(val longEdges: List<Int> = emptyList()) {
    init { require(longEdges.size <= 4 && longEdges.all { it in 128..4096 }) }
}
data class DetectorStats(val passes: Int = 0, val detected: Int = 0, val eligible: Int = 0,
    val detectionMs: Double = 0.0, val embeddingMs: Double = 0.0, val totalMs: Double = 0.0)

fun remapFaceRow(row: FloatArray, scaleX: Double, scaleY: Double): FloatArray {
    require(row.size == 15 && scaleX > 0 && scaleY > 0)
    return row.copyOf().apply { for (i in 0..13) this[i] = (this[i] / if (i % 2 == 0) scaleX else scaleY).toFloat() }
}
fun mergeFaceRows(rows: List<FloatArray>, iou: Double = .3): List<FloatArray> {
    val kept = mutableListOf<FloatArray>()
    fun box(f: FloatArray) = Box(f[0].toDouble(), f[1].toDouble(), f[2].toDouble(), f[3].toDouble())
    rows.sortedByDescending { it[14] }.forEach { row -> if (kept.none { box(it).iou(box(row)) > iou }) kept.add(row) }
    return kept
}

class OpenCvBackend(directory: File, threshold: Float = .85f, private val minFaceSize: Int = 42,
    private val scalePolicy: DetectorScalePolicy = DetectorScalePolicy()) : FaceBackend {
    override val model = SFace.model
    private val detector: FaceDetectorYN
    private val recognizer: FaceRecognizerSF
    @Volatile var stats = DetectorStats(); private set
    init {
        check(OpenCVLoader.initLocal()) { "OpenCV could not load on this device" }
        mapOf(SFace.DETECTOR to SFace.DETECTOR_SHA, SFace.RECOGNIZER to SFace.RECOGNIZER_SHA).forEach { (name, hash) ->
            val digest = MessageDigest.getInstance("SHA-256")
            File(directory, name).inputStream().use { input ->
                val buffer = ByteArray(65536); var count = input.read(buffer)
                while (count > 0) { digest.update(buffer, 0, count); count = input.read(buffer) }
            }
            check(digest.digest().joinToString("") { "%02x".format(it) } == hash) { "Model checksum mismatch" }
        }
        detector = FaceDetectorYN.create(File(directory, SFace.DETECTOR).path, "", Size(320.0, 320.0), threshold, .3f, 5000)
        recognizer = FaceRecognizerSF.create(File(directory, SFace.RECOGNIZER).path, "")
    }
    /** Initialize both DNN execution paths without retaining a camera frame or enrollment. */
    @Synchronized fun warmUp() {
        detect(BgrImage(640,360,ByteArray(640*360*3)),4,false)
        val aligned = Mat.zeros(112,112,CvType.CV_8UC3); val feature = Mat()
        try { recognizer.feature(aligned,feature) }
        finally { aligned.release(); feature.release() }
    }
    @Synchronized override fun detect(image: BgrImage, maxFaces: Int?, centralOnly: Boolean): List<Detection> {
        require(maxFaces == null || maxFaces > 0)
        val begun = System.nanoTime()
        stats = DetectorStats()
        val source = Mat(image.height, image.width, CvType.CV_8UC3)
        try {
            source.put(0, 0, image.pixels)
            val sizes = (scalePolicy.longEdges.ifEmpty { listOf(maxOf(image.width, image.height)) }).map { edge ->
                val scale = minOf(1.0, edge.toDouble() / maxOf(image.width, image.height))
                maxOf(1, (image.width * scale).toInt()) to maxOf(1, (image.height * scale).toInt())
            }.distinct()
            val candidates = mutableListOf<FloatArray>()
            sizes.forEach { (width, height) ->
                val resized = Mat(); val found = Mat()
                try {
                    val input = if (width == image.width && height == image.height) source else {
                        Imgproc.resize(source, resized, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA); resized
                    }
                    detector.setInputSize(input.size()); detector.detect(input, found)
                    for (i in 0 until found.rows()) {
                        val f = FloatArray(15); found.get(i, 0, f)
                        if (f.all { it.isFinite() } && f[2] > 0 && f[3] > 0)
                            candidates.add(remapFaceRow(f, width.toDouble()/image.width, height.toDouble()/image.height))
                    }
                } finally { resized.release(); found.release() }
            }
            val eligible = (if (sizes.size == 1) candidates else mergeFaceRows(candidates)).filter { f ->
                minOf(f[2], f[3]) >= minFaceSize && (!centralOnly ||
                    ((f[0] + f[2] / 2) in image.width * .12..image.width * .88 &&
                     (f[1] + f[3] / 2) in image.height * .08..image.height * .92))
            }.sortedByDescending { it[2] * it[3] }.let { if (maxFaces == null) it else it.take(maxFaces) }
            val detectedAt = System.nanoTime()
            var embeddingNs = 0L
            val results = eligible.map { f ->
                val row = Mat(1, 15, CvType.CV_32F).apply { put(0, 0, f) }; val aligned = Mat(); val feature = Mat(); val gray = Mat(); val lap = Mat()
                val mean = MatOfDouble(); val deviation = MatOfDouble()
                try {
                    val embeddingStart = System.nanoTime()
                    recognizer.alignCrop(source, row, aligned); recognizer.feature(aligned, feature)
                    embeddingNs += System.nanoTime() - embeddingStart
                    val embedding = FloatArray(128); feature.get(0, 0, embedding)
                    Imgproc.cvtColor(aligned, gray, Imgproc.COLOR_BGR2GRAY); Imgproc.Laplacian(gray, lap, CvType.CV_64F)
                    Core.meanStdDev(lap, mean, deviation)
                    val blur = deviation.toArray()[0].let { it * it }
                    val area = minOf(1.0, f[2].toDouble() * f[3] / maxOf(1.0, image.width.toDouble() * image.height * .08))
                    val clarity = minOf(1.0, ln1p(maxOf(0.0, blur)) / ln1p(250.0))
                    val pixels = ByteArray(aligned.rows() * aligned.cols() * 3); aligned.get(0, 0, pixels)
                    Detection(Box(f[0].toDouble(), f[1].toDouble(), f[2].toDouble(), f[3].toDouble()),
                        normalizeEmbedding(embedding), model, f[14].toDouble(),
                        minOf(1.0, f[14] * (.5 + .25 * area + .25 * clarity)), BgrImage(aligned.cols(), aligned.rows(), pixels))
                } finally { row.release(); aligned.release(); feature.release(); gray.release(); lap.release(); mean.release(); deviation.release() }
            }.sortedBy { it.box.x + it.box.width / 2 }
            stats = DetectorStats(sizes.size, candidates.size, eligible.size, (detectedAt-begun)/1e6,
                embeddingNs/1e6, (System.nanoTime()-begun)/1e6)
            return results
        } finally { source.release() }
    }
}
