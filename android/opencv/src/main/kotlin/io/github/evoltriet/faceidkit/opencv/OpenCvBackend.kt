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

/** Empty edges preserves the legacy original-resolution detector. */
data class DetectorScalePolicy(val longEdges: List<Int> = emptyList()) {
    init { require(longEdges.size <= 4 && longEdges.all { it in 128..4096 }) }
}
data class DetectorStats(val passes: Int = 0, val detected: Int = 0, val eligible: Int = 0,
    val detectionMs: Double = 0.0, val embeddingMs: Double = 0.0, val totalMs: Double = 0.0)

/** Facial geometry is sensitive, even without an identity or embedding. A location
 * is bound to exactly one image, not reusable across camera frames. */
class FaceLocation internal constructor(internal val row: FloatArray, internal val source: String,
    val regionIndex: Int, val imageWidth: Int, val imageHeight: Int, internal val region: Box?) {
    val box get() = Box(row[0].toDouble(), row[1].toDouble(), row[2].toDouble(), row[3].toDouble())
    val confidence get() = row[14].toDouble()
}
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
    @Synchronized fun warmUp() {
        detect(BgrImage(640,360,ByteArray(640*360*3)),4,false)
        val aligned = Mat.zeros(112,112,CvType.CV_8UC3); val feature = Mat()
        try { recognizer.feature(aligned,feature) } finally { aligned.release(); feature.release() }
    }
    private fun fingerprint(image: BgrImage) = MessageDigest.getInstance("SHA-256").digest(image.pixels)
        .joinToString("") { "%02x".format(it) }

    /** Null regions keeps whole-image behavior; empty regions performs NO inference.
     * Detector passes receive only the caller-selected rectangular pixel regions. */
    @Synchronized fun locate(image: BgrImage, regions: List<Box>? = null, maxFaces: Int? = null,
        centralOnly: Boolean = false): List<FaceLocation> {
        require(maxFaces == null || maxFaces > 0)
        val begun = System.nanoTime(); stats = DetectorStats()
        if(regions?.isEmpty() == true) return emptyList()
        val source = Mat(image.height,image.width,CvType.CV_8UC3)
        try {
            source.put(0,0,image.pixels)
            val chosen = regions ?: listOf(Box(0.0,0.0,image.width.toDouble(),image.height.toDouble()))
            require(chosen.size <= 1000)
            var passes = 0; var rawCount = 0
            val rows = mutableListOf<Pair<Int,FloatArray>>()
            chosen.forEachIndexed { regionIndex, box ->
                require(listOf(box.x,box.y,box.width,box.height).all { it.isFinite() } && box.width > 0 && box.height > 0)
                require(box.x >= 0 && box.y >= 0 && box.x+box.width <= image.width && box.y+box.height <= image.height) {"Region is outside the image"}
                val x = box.x.toInt().coerceIn(0,image.width-1); val y = box.y.toInt().coerceIn(0,image.height-1)
                val right = (box.x+box.width).toInt().coerceIn(x+1,image.width)
                val bottom = (box.y+box.height).toInt().coerceIn(y+1,image.height)
                val crop = source.submat(Rect(x,y,right-x,bottom-y))
                try {
                    val sizes = scalePolicy.longEdges.ifEmpty { listOf(maxOf(crop.cols(),crop.rows())) }.map { edge ->
                        val scale = minOf(1.0,edge.toDouble()/maxOf(crop.cols(),crop.rows()))
                        maxOf(1,(crop.cols()*scale).toInt()) to maxOf(1,(crop.rows()*scale).toInt())
                    }.distinct()
                    val candidates = mutableListOf<FloatArray>()
                    sizes.forEach { (width,height) ->
                        val resized = Mat(); val found = Mat()
                        try {
                            val input = if(width == crop.cols() && height == crop.rows()) crop else {
                                Imgproc.resize(crop,resized,Size(width.toDouble(),height.toDouble()),0.0,0.0,Imgproc.INTER_AREA); resized
                            }
                            detector.setInputSize(input.size()); detector.detect(input,found); passes++
                            for(i in 0 until found.rows()) {
                                val f = FloatArray(15); found.get(i,0,f)
                                if(f.all { it.isFinite() } && f[2] > 0 && f[3] > 0) {
                                    val row = remapFaceRow(f,width.toDouble()/crop.cols(),height.toDouble()/crop.rows())
                                    row[0]+=x; row[1]+=y
                                    for(j in 4..12 step 2) { row[j]+=x; row[j+1]+=y }
                                    candidates.add(row)
                                }
                            }
                        } finally { resized.release(); found.release() }
                    }
                    rawCount += candidates.size
                    (if(sizes.size == 1) candidates else mergeFaceRows(candidates)).forEach { rows.add(regionIndex to it) }
                } finally { crop.release() }
            }
            val eligible = rows.filter { (_,f) -> minOf(f[2],f[3]) >= minFaceSize && (!centralOnly ||
                ((f[0]+f[2]/2) in image.width*.12..image.width*.88 && (f[1]+f[3]/2) in image.height*.08..image.height*.92))
            }.sortedByDescending { it.second[2]*it.second[3] }.let { if(maxFaces == null) it else it.take(maxFaces) }
            val hash = fingerprint(image)
            val elapsed=(System.nanoTime()-begun)/1e6
            stats = DetectorStats(passes,rawCount,eligible.size,elapsed,totalMs=elapsed)
            return eligible.map { (region,row) -> FaceLocation(row,hash,region,image.width,image.height,regions?.get(region)) }
        } finally { source.release() }
    }

    /** Alignment uses original upright pixels, preserving SFace preprocessing. */
    @Synchronized fun embed(image: BgrImage, locations: List<FaceLocation>): List<Detection> {
        if(locations.isEmpty()) return emptyList()
        val hash = fingerprint(image)
        require(locations.all { it.source == hash && it.imageWidth == image.width && it.imageHeight == image.height }) { "Locations belong to a different image" }
        val begun = System.nanoTime(); val source = Mat(image.height,image.width,CvType.CV_8UC3)
        try {
            source.put(0,0,image.pixels)
            val results = locations.map { location ->
                val f=location.row
                val row=Mat(1,15,CvType.CV_32F).apply { put(0,0,f) }; val aligned=Mat(); val feature=Mat(); val gray=Mat(); val lap=Mat()
                val mean=MatOfDouble(); val deviation=MatOfDouble(); var restricted: Mat?=null
                try {
                    // Keep original upright coordinates and resolution, but never let
                    // alignment interpolation read pixels outside an approved region.
                    val input=location.region?.let {box ->
                        Mat.zeros(image.height,image.width,CvType.CV_8UC3).also {masked ->
                            restricted=masked
                            val x=box.x.toInt();val y=box.y.toInt()
                            val rect=Rect(x,y,(box.x+box.width).toInt()-x,(box.y+box.height).toInt()-y)
                            val from=source.submat(rect);val to=masked.submat(rect)
                            try {from.copyTo(to)} finally {from.release();to.release()}
                        }
                    } ?: source
                    recognizer.alignCrop(input,row,aligned); recognizer.feature(aligned,feature)
                    val embedding=FloatArray(128); feature.get(0,0,embedding)
                    Imgproc.cvtColor(aligned,gray,Imgproc.COLOR_BGR2GRAY); Imgproc.Laplacian(gray,lap,CvType.CV_64F)
                    Core.meanStdDev(lap,mean,deviation)
                    val blur=deviation.toArray()[0].let { it*it }
                    val area=minOf(1.0,f[2].toDouble()*f[3]/maxOf(1.0,image.width.toDouble()*image.height*.08))
                    val clarity=minOf(1.0,ln1p(maxOf(0.0,blur))/ln1p(250.0))
                    val pixels=ByteArray(aligned.rows()*aligned.cols()*3); aligned.get(0,0,pixels)
                    Detection(location.box,normalizeEmbedding(embedding),model,location.confidence,
                        minOf(1.0,f[14]*(.5+.25*area+.25*clarity)),BgrImage(aligned.cols(),aligned.rows(),pixels))
                } finally { restricted?.release(); row.release(); aligned.release(); feature.release(); gray.release(); lap.release(); mean.release(); deviation.release() }
            }
            val ms=(System.nanoTime()-begun)/1e6; stats=stats.copy(embeddingMs=ms,totalMs=stats.detectionMs+ms)
            return results.sortedBy { it.box.x+it.box.width/2 }
        } finally { source.release() }
    }
    @Synchronized override fun detect(image: BgrImage, maxFaces: Int?, centralOnly: Boolean) =
        embed(image,locate(image,maxFaces=maxFaces,centralOnly=centralOnly))
}
