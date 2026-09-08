package io.github.evoltriet.faceidkit

import kotlin.math.sqrt
import java.util.UUID

const val SCORING_VERSION = "cosine-top3-0.7max-0.3mean/unique-top-only-v1"
fun normalizeEmbedding(vector: FloatArray): FloatArray {
    require(vector.isNotEmpty() && vector.all { it.isFinite() }) { "Embedding must be finite and nonempty" }
    val norm = sqrt(vector.sumOf { it.toDouble() * it })
    require(norm > 0 && norm.isFinite()) { "Embedding norm must be positive" }
    return FloatArray(vector.size) { (vector[it] / norm).toFloat() }
}
fun checkModel(expected: ModelSpec, actual: ModelSpec, vector: FloatArray): FloatArray {
    require(expected == actual && vector.size == expected.dimension) { "Incompatible model, dimension, or preprocessing" }
    return normalizeEmbedding(vector)
}
fun galleryScore(vector: FloatArray, gallery: List<FloatArray>): Double {
    require(gallery.isNotEmpty())
    val normalized = normalizeEmbedding(vector)
    val scores = gallery.map { sample ->
        require(sample.size == normalized.size)
        sample.indices.sumOf { sample[it].toDouble() * normalized[it] }
    }.sortedDescending().take(3)
    return (.7 * scores.first() + .3 * scores.average()).coerceIn(-1.0, 1.0)
}
fun resolveUnique(groups: List<List<Candidate>>, policy: MatchPolicy): List<Match> {
    val results = groups.map { group ->
        val ordered = group.sortedWith(compareByDescending<Candidate> { it.score }.thenBy { it.identityId })
        if (ordered.isEmpty()) Match(null, 0.0, 0.0, reason = "empty_gallery")
        else {
            val best = ordered.first()
            val margin = if (ordered.size > 1) best.score - ordered[1].score else 2.0
            val reason = when { best.score < policy.threshold -> "below_threshold"
                margin < policy.margin -> "ambiguous"; else -> "matched" }
            Match(if (reason == "matched") best.identityId else null, best.score, margin, ordered, reason)
        }
    }.toMutableList()
    val used = mutableSetOf<String>()
    results.indices.sortedByDescending { results[it].score }.forEach { i ->
        val match = results[i]
        if (match.identityId != null && !used.add(match.identityId))
            results[i] = match.copy(identityId = null, reason = "identity_conflict")
    }
    return results
}
class InMemoryStore : GalleryStore {
    private var rev = 0L
    private val people = linkedMapOf<String, Identity>()
    private val samples = linkedMapOf<String, EnrollmentSample>()
    override val revision get() = synchronized(this) { rev }
    @Synchronized override fun snapshot() = GallerySnapshot(rev, samples.values.map { it.copy(embedding = it.embedding.copyOf()) })
    @Synchronized override fun identities() = people.values.toList()
    @Synchronized override fun putIdentity(identity: Identity) { require(identity.id.isNotBlank()); people[identity.id] = identity; rev++ }
    @Synchronized override fun putSample(sample: EnrollmentSample) {
        require(sample.id.isNotBlank() && sample.identityId in people && sample.quality in 0.0..1.0)
        samples[sample.id] = sample.copy(embedding = checkModel(sample.model, sample.model, sample.embedding)); rev++
    }
    @Synchronized override fun deleteSample(sampleId: String) { samples.remove(sampleId); rev++ }
    @Synchronized override fun deleteIdentity(identityId: String) {
        people.remove(identityId); samples.entries.removeAll { it.value.identityId == identityId }; rev++
    }
}
class Identifier(val store: GalleryStore, val model: ModelSpec, var policy: MatchPolicy = MatchPolicy()) {
    private var loadedRevision = -1L
    private var gallery = emptyMap<String, List<FloatArray>>()
    private fun refresh() {
        if (loadedRevision == store.revision) return
        val snapshot = store.snapshot()
        gallery = snapshot.samples.groupBy { it.identityId }.mapValues { (_, samples) ->
            samples.map { checkModel(model, it.model, it.embedding) }
        }
        loadedRevision = snapshot.revision
    }
    @Synchronized fun candidates(vector: FloatArray, spec: ModelSpec): List<Candidate> {
        val query = checkModel(model, spec, vector); refresh()
        return gallery.map { Candidate(it.key, galleryScore(query, it.value)) }
            .sortedWith(compareByDescending<Candidate> { it.score }.thenBy { it.identityId })
    }
    @Synchronized fun identifyMany(queries: List<Pair<FloatArray, ModelSpec>>) =
        resolveUnique(queries.map { candidates(it.first, it.second) }, policy)
    fun identify(vector: FloatArray, spec: ModelSpec) = identifyMany(listOf(vector to spec)).first()
}
class FaceIdentifier(val backend: FaceBackend, val store: GalleryStore, policy: MatchPolicy = MatchPolicy()) {
    val identifier = Identifier(store, backend.model, policy)
    fun detect(image: BgrImage, maxFaces: Int? = null, centralOnly: Boolean = false) = backend.detect(image, maxFaces, centralOnly)
    fun enroll(identityId: String, face: Detection, sampleId: String = UUID.randomUUID().toString(), sourceId: String = ""): EnrollmentSample {
        val sample = EnrollmentSample(sampleId, identityId, checkModel(backend.model, face.model, face.embedding), face.model, face.quality, sourceId)
        store.putSample(sample); return sample
    }
    fun identify(image: BgrImage, maxFaces: Int? = null, centralOnly: Boolean = false): List<Pair<Detection, Match>> {
        val faces = detect(image, maxFaces, centralOnly)
        return faces.zip(identifier.identifyMany(faces.map { it.embedding to it.model }))
    }
}
