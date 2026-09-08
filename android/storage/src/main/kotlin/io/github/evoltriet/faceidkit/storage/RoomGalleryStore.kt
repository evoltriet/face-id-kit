package io.github.evoltriet.faceidkit.storage

import android.content.Context
import androidx.room.*
import io.github.evoltriet.faceidkit.*
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

@Entity(tableName = "identities") data class IdentityRow(@PrimaryKey val id: String, val payload: ByteArray)
@Entity(tableName = "samples", foreignKeys = [ForeignKey(entity = IdentityRow::class, parentColumns = ["id"], childColumns = ["identityId"], onDelete = ForeignKey.CASCADE)], indices = [Index("identityId")])
data class SampleRow(@PrimaryKey val id: String, val identityId: String, val encrypted: ByteArray, val fingerprint: String,
    val dimension: Int, val preprocessing: String, val quality: Double, val sourceId: String)
@Entity(tableName = "extras") data class ExtraRow(@PrimaryKey val id: String, val encrypted: ByteArray)
@Entity(tableName = "meta") data class MetaRow(@PrimaryKey val id: Int = 0, val revision: Long = 0)

@Dao interface GalleryDao {
    @Query("SELECT * FROM identities ORDER BY id") fun identities(): List<IdentityRow>
    @Insert(onConflict = OnConflictStrategy.IGNORE) fun insertIdentity(row: IdentityRow)
    @Query("UPDATE identities SET payload=:payload WHERE id=:id") fun updateIdentity(id: String, payload: ByteArray)
    @Query("SELECT * FROM samples ORDER BY id") fun samples(): List<SampleRow>
    @Upsert fun putSample(row: SampleRow)
    @Query("DELETE FROM samples WHERE id=:id") fun deleteSample(id: String)
    @Query("DELETE FROM identities WHERE id=:id") fun deleteIdentity(id: String)
    @Query("SELECT revision FROM meta WHERE id=0") fun revision(): Long?
    @Insert(onConflict = OnConflictStrategy.IGNORE) fun initMeta(row: MetaRow)
    @Query("UPDATE meta SET revision=revision+1 WHERE id=0") fun bump()
    @Upsert fun putExtra(row: ExtraRow)
    @Query("SELECT encrypted FROM extras WHERE id=:id") fun extra(id: String): ByteArray?
    @Query("DELETE FROM extras WHERE id=:id") fun deleteExtra(id: String)
    @Query("DELETE FROM extras") fun deleteExtras()
    @Query("DELETE FROM identities") fun deleteIdentities()
}
@Database(entities = [IdentityRow::class, SampleRow::class, ExtraRow::class, MetaRow::class], version = 1, exportSchema = false)
abstract class GalleryDatabase : RoomDatabase() { abstract fun gallery(): GalleryDao }

/** All operations are synchronous: call from an application-owned background executor. */
class RoomGalleryStore(context: Context, private val cipher: EmbeddingCipher, filename: String = "face-gallery.db") : GalleryStore, Closeable {
    private val db = Room.databaseBuilder(context.applicationContext, GalleryDatabase::class.java, filename)
        .setJournalMode(RoomDatabase.JournalMode.TRUNCATE).addCallback(object : RoomDatabase.Callback() {
            override fun onOpen(db: androidx.sqlite.db.SupportSQLiteDatabase) { db.query("PRAGMA secure_delete=ON").use { it.moveToFirst() } }
        }).build()
    private val dao = db.gallery()
    init { dao.initMeta(MetaRow()) }
    private fun seal(kind: String, id: String, payload: ByteArray): ByteArray {
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { it.writeUTF(kind); it.writeUTF(id); it.write(payload) }
        return cipher.protect(bytes.toByteArray())
    }
    private fun open(kind: String, id: String, bytes: ByteArray): ByteArray =
        DataInputStream(ByteArrayInputStream(cipher.unprotect(bytes))).use {
            require(it.readUTF() == kind && it.readUTF() == id) { "Encrypted record binding mismatch" }; it.readBytes()
        }
    override val revision get() = dao.revision() ?: 0
    override fun identities() = dao.identities().map { Identity(it.id) }
    override fun snapshot(): GallerySnapshot {
        var result: GallerySnapshot? = null
        db.runInTransaction {
            result = GallerySnapshot(revision, dao.samples().map { row ->
                val bytes = open("sample", row.id + ":" + row.identityId, row.encrypted)
                require(bytes.size == row.dimension * 4)
                val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                val spec = ModelSpec(row.fingerprint, row.dimension, row.preprocessing)
                EnrollmentSample(row.id, row.identityId, checkModel(spec, spec, FloatArray(row.dimension) { buffer.float }),
                    spec, row.quality, row.sourceId)
            })
        }
        return requireNotNull(result)
    }
    override fun putIdentity(identity: Identity) {
        require(identity.id.isNotBlank())
        db.runInTransaction { dao.insertIdentity(IdentityRow(identity.id, seal("identity", identity.id, byteArrayOf()))); dao.bump() }
    }
    fun metadata(id: String): ByteArray? = dao.identities().find { it.id == id }?.let { open("identity", id, it.payload) }
    fun putMetadata(id: String, value: ByteArray) {
        db.runInTransaction { require(dao.identities().any { it.id == id }); dao.updateIdentity(id, seal("identity", id, value)); dao.bump() }
    }
    override fun putSample(sample: EnrollmentSample) {
        require(sample.id.isNotBlank() && sample.quality in 0.0..1.0)
        val vector = checkModel(sample.model, sample.model, sample.embedding)
        val bytes = ByteBuffer.allocate(vector.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        vector.forEach { bytes.putFloat(it) }
        db.runInTransaction {
            dao.putSample(SampleRow(sample.id, sample.identityId, seal("sample", sample.id + ":" + sample.identityId, bytes.array()),
                sample.model.fingerprint, sample.model.dimension, sample.model.preprocessing, sample.quality, sample.sourceId))
            dao.bump()
        }
    }
    fun putExtra(id: String, value: ByteArray) { dao.putExtra(ExtraRow(id, seal("extra", id, value))) }
    fun extra(id: String): ByteArray? = dao.extra(id)?.let { open("extra", id, it) }
    fun deleteExtra(id: String) = dao.deleteExtra(id)
    /** Combines enrollment, app metadata, and thumbnail writes into one transaction. */
    fun transaction(action: () -> Unit) = db.runInTransaction(action)
    override fun deleteSample(sampleId: String) {
        db.runInTransaction { dao.deleteSample(sampleId); dao.deleteExtra("thumb:$sampleId"); dao.bump() }
    }
    override fun deleteIdentity(identityId: String) {
        db.runInTransaction {
            dao.samples().filter { it.identityId == identityId }.forEach { dao.deleteExtra("thumb:" + it.id) }
            dao.deleteIdentity(identityId); dao.bump()
        }
    }
    fun deleteAll() { db.runInTransaction { dao.deleteIdentities(); dao.deleteExtras(); dao.bump() } }
    override fun close() = db.close()
}
