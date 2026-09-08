package io.github.evoltriet.faceidkit.opencv
import androidx.test.platform.app.InstrumentationRegistry
import io.github.evoltriet.faceidkit.*
import org.junit.Test
import org.junit.Assert.*
import org.opencv.core.*
import org.opencv.objdetect.FaceRecognizerSF
import java.io.File
class ModelTest {
    @Test fun realOnnxModelsProduceFiniteFeatures() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val directory = File(instrumentation.targetContext.cacheDir, "models").apply { mkdirs() }
        for (name in listOf(SFace.DETECTOR, SFace.RECOGNIZER))
            instrumentation.context.assets.open("models/$name").use { input -> File(directory, name).outputStream().use(input::copyTo) }
        val backend = OpenCvBackend(directory)
        assertTrue(backend.detect(BgrImage(320, 320, ByteArray(320 * 320 * 3)), 4, false).isEmpty())
        val source = Mat(112, 112, CvType.CV_8UC3); val output = Mat()
        try {
            source.put(0, 0, ByteArray(112 * 112 * 3) { ((it * 73 + 19) % 256).toByte() })
            FaceRecognizerSF.create(File(directory, SFace.RECOGNIZER).path, "").feature(source, output)
            val vector = FloatArray(128); output.get(0, 0, vector)
            val normalized = checkModel(SFace.model, backend.model, vector)
            assertEquals(128, normalized.size)
            assertTrue(normalized.all { it.isFinite() })
            assertEquals(1.0, normalized.sumOf { it.toDouble() * it }, .00001)
        } finally { source.release(); output.release(); directory.listFiles()?.forEach { it.delete() }; directory.delete() }
    }
}
