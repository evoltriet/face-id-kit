package io.github.evoltriet.faceidkit.storage
import androidx.test.platform.app.InstrumentationRegistry
import io.github.evoltriet.faceidkit.*
import org.junit.Test
import org.junit.Assert.*
import java.util.UUID

class StorageTest {
    @Test fun encryptionReloadDeletionAndTampering() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = "test-" + UUID.randomUUID() + ".db"; val cipher = KeystoreCipher(name)
        val model = ModelSpec("test", 3, "test")
        val encrypted = cipher.protect("private".toByteArray())
        assertFalse(String(encrypted).contains("private"))
        encrypted[encrypted.lastIndex] = (encrypted.last().toInt() xor 1).toByte()
        try { cipher.unprotect(encrypted); fail("Tampering accepted") } catch (_: javax.crypto.AEADBadTagException) {}
        RoomGalleryStore(context, cipher, name).use { store ->
            store.putIdentity(Identity("a")); store.putMetadata("a", "Secret name".toByteArray())
            store.putSample(EnrollmentSample("s", "a", floatArrayOf(1f, 0f, 0f), model))
            store.putExtra("thumb:s", "crop".toByteArray())
        }
        RoomGalleryStore(context, cipher, name).use { store ->
            assertEquals("Secret name", String(store.metadata("a")!!)); assertEquals(1, store.snapshot().samples.size)
            val revision = store.revision; store.deleteIdentity("a")
            assertTrue(store.revision > revision); assertTrue(store.snapshot().samples.isEmpty()); assertNull(store.extra("thumb:s"))
        }
        context.deleteDatabase(name); cipher.deleteKey()
    }
}
