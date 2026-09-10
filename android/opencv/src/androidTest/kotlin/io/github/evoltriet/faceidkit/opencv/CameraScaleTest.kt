package io.github.evoltriet.faceidkit.opencv

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import io.github.evoltriet.faceidkit.*
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class CameraScaleTest {
    @Test fun geometryAndDuplicateSuppression() {
        val f = floatArrayOf(10f,20f,50f,60f,20f,30f,40f,30f,30f,40f,20f,50f,40f,50f,.9f)
        val mapped = remapFaceRow(f, .5, .25)
        assertEquals(20f, mapped[0]); assertEquals(80f,mapped[1]); assertEquals(160f,mapped[12]); assertEquals(200f,mapped[13])
        assertEquals(.9f,mapped[14]); assertEquals(1,mergeFaceRows(listOf(f,f.copyOf())).size)
        val distant = f.copyOf().apply { this[0] += 200 }
        assertEquals(2,mergeFaceRows(listOf(f,distant)).size)
    }
    @Test fun syntheticFaceAtOrdinaryAndCloseCameraSizes() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val directory = File(instrumentation.targetContext.cacheDir,"camera-scale-models").apply { mkdirs() }
        for (name in listOf(SFace.DETECTOR,SFace.RECOGNIZER))
            instrumentation.context.assets.open("models/$name").use { input -> File(directory,name).outputStream().use(input::copyTo) }
        val original = instrumentation.context.assets.open("synthetic-face.png").use { BitmapFactory.decodeStream(it) }
        val backend = OpenCvBackend(directory, scalePolicy = DetectorScalePolicy(listOf(640,320)))
        try {
            for (edge in listOf(320,1280)) {
                val bitmap = Bitmap.createScaledBitmap(original,edge,edge,true)
                val argb = IntArray(edge*edge); bitmap.getPixels(argb,0,edge,0,0,edge,edge)
                val bgr = ByteArray(edge*edge*3)
                argb.forEachIndexed { i,c -> bgr[i*3]=c.toByte(); bgr[i*3+1]=(c shr 8).toByte(); bgr[i*3+2]=(c shr 16).toByte() }
                val faces = backend.detect(BgrImage(edge,edge,bgr),4,false)
                assertEquals("Synthetic frontal face at $edge",1,faces.size)
                val face = faces.single()
                assertEquals(128,face.embedding.size); assertEquals(SFace.model,face.model)
                assertTrue(face.box.x >= 0 && face.box.x + face.box.width <= edge)
                assertTrue(face.box.width > edge*.2 && face.box.height > edge*.3)
                assertEquals(112,face.aligned!!.width)
                bitmap.recycle()
            }
        } finally { original.recycle() }
    }
}
