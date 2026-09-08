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

class OpenCvBackend(directory: File, threshold: Float = .85f, private val minFaceSize: Int = 42) : FaceBackend {
    override val model = SFace.model
    private val detector: FaceDetectorYN
    private val recognizer: FaceRecognizerSF
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
    @Synchronized override fun detect(image: BgrImage, maxFaces: Int?, centralOnly: Boolean): List<Detection> {
        require(maxFaces == null || maxFaces > 0)
        val source = Mat(image.height, image.width, CvType.CV_8UC3)
        val faces = Mat()
        try {
            source.put(0, 0, image.pixels)
            detector.setInputSize(Size(image.width.toDouble(), image.height.toDouble()))
            detector.detect(source, faces)
            val eligible = (0 until faces.rows()).map { row ->
                val values = FloatArray(15); faces.get(row, 0, values); row to values
            }.filter { (_, f) ->
                minOf(f[2], f[3]) >= minFaceSize && (!centralOnly ||
                    ((f[0] + f[2] / 2) in image.width * .12..image.width * .88 &&
                     (f[1] + f[3] / 2) in image.height * .08..image.height * .92))
            }.sortedByDescending { it.second[2] * it.second[3] }.let { if (maxFaces == null) it else it.take(maxFaces) }
            return eligible.map { (index, f) ->
                val row = faces.row(index); val aligned = Mat(); val feature = Mat(); val gray = Mat(); val lap = Mat()
                val mean = MatOfDouble(); val deviation = MatOfDouble()
                try {
                    recognizer.alignCrop(source, row, aligned); recognizer.feature(aligned, feature)
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
        } finally { source.release(); faces.release() }
    }
}
