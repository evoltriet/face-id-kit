package io.github.evoltriet.faceidkit.storage

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import io.github.evoltriet.faceidkit.EmbeddingCipher
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

class KeystoreCipher(private val alias: String) : EmbeddingCipher {
    private fun key(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val existing = store.getKey(alias, null)
        if (existing != null) return existing as SecretKey
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256).build())
        return generator.generateKey()
    }
    @Synchronized override fun protect(data: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, key()); cipher.updateAAD("face-id-kit/android/v1".toByteArray())
        return byteArrayOf(1) + cipher.iv + cipher.doFinal(data)
    }
    @Synchronized override fun unprotect(data: ByteArray): ByteArray {
        require(data.size >= 29 && data[0] == 1.toByte()) { "Invalid encrypted record" }
        // Never generate a replacement key while attempting to read existing data.
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val secret = store.getKey(alias, null) as? SecretKey ?: error("Device encryption key is unavailable")
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, secret, GCMParameterSpec(128, data.copyOfRange(1, 13)))
        cipher.updateAAD("face-id-kit/android/v1".toByteArray())
        return cipher.doFinal(data.copyOfRange(13, data.size))
    }
    fun deleteKey() { KeyStore.getInstance("AndroidKeyStore").apply { load(null); deleteEntry(alias) } }
}
