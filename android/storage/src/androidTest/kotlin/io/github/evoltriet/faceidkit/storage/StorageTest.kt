package io.github.evoltriet.faceidkit.storage
import androidx.test.platform.app.InstrumentationRegistry
import io.github.evoltriet.faceidkit.*
import org.junit.Test
import org.junit.Assert.*
import java.util.UUID

class StorageTest {
    @Test fun filteredRowsAreNotDecryptedAndPresentationDoesNotInvalidate() {
        val context=InstrumentationRegistry.getInstrumentation().targetContext
        val name="filtered-${UUID.randomUUID()}.db";val real=KeystoreCipher(name)
        var opened=0
        val counting=object : EmbeddingCipher {
            override fun protect(plain: ByteArray)=real.protect(plain)
            override fun unprotect(encrypted: ByteArray): ByteArray {opened++;return real.unprotect(encrypted)}
        }
        try {RoomGalleryStore(context,counting,name).use {store ->
            val model=ModelSpec("test",3,"test")
            listOf("active","inactive").forEach {id->store.putIdentity(Identity(id));store.putSample(EnrollmentSample(id,id,floatArrayOf(1f,0f,0f),model))}
            opened=0;assertEquals(2,store.sampleDescriptors().size);assertEquals(0,opened)
            assertTrue(store.snapshot(emptySet()).samples.isEmpty());assertEquals(0,opened)
            assertEquals("active",store.snapshot(setOf("active")).samples.single().identityId);assertEquals(1,opened)
            val before=store.revision;store.putMetadata("active","display".toByteArray(),false);assertEquals(before,store.revision)
            store.invalidateRevision();assertTrue(store.revision>before)
        }} finally {context.deleteDatabase(name);real.deleteKey()}
    }
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
