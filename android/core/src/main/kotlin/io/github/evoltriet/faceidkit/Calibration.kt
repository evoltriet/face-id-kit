package io.github.evoltriet.faceidkit

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
data class ValidationSample(val id: String, val identityId: String?, val embedding: FloatArray, val model: ModelSpec, val sourceId: String = "")
data class Evaluation(val knownAccuracy: Double, val knownTests: Int, val unknownTests: Int, val falseAccepts: Int)
data class CalibrationReport(val policy: MatchPolicy, val tuning: Evaluation, val evaluation: Evaluation,
    val passed: Boolean, val reason: String, val scoringVersion: String = SCORING_VERSION)
private fun keys(id: String, source: String, embedding: FloatArray): Set<String> {
    val bytes = ByteBuffer.allocate(embedding.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    normalizeEmbedding(embedding).forEach { bytes.putFloat(it) }
    val hash = MessageDigest.getInstance("SHA-256").digest(bytes.array()).joinToString("") { "%02x".format(it) }
    return setOf("id:$id", "embedding:$hash") + if (source.isNotEmpty()) setOf("source:$source") else emptySet()
}
fun calibrate(enrollment: List<EnrollmentSample>, tuning: List<ValidationSample>, evaluation: List<ValidationSample>,
    model: ModelSpec, minAccuracy: Double = .95, minKnown: Int = 10, minUnknown: Int = 5,
    thresholds: List<Double> = (35..75).map { it / 100.0 }, margins: List<Double> = (2..15).map { it / 100.0 }): CalibrationReport {
    require(minKnown > 0 && minUnknown > 0 && minAccuracy > 0 && minAccuracy <= 1)
    val partitions = listOf(enrollment.map { keys(it.id, it.sourceId, it.embedding) },
        tuning.map { keys(it.id, it.sourceId, it.embedding) }, evaluation.map { keys(it.id, it.sourceId, it.embedding) })
    val prior = mutableSetOf<String>()
    partitions.forEach { group ->
        require(group.none { it.any { key -> key in prior } }) { "Enrollment, tuning and evaluation must use disjoint samples and sources" }
        group.forEach { prior.addAll(it) }
    }
    val store = InMemoryStore()
    enrollment.map { it.identityId }.distinct().forEach { store.putIdentity(Identity(it)) }
    enrollment.forEach(store::putSample)
    val ids = store.identities().map { it.id }
    require((tuning + evaluation).all { it.identityId == null || it.identityId in ids }) { "Known validation identity has no enrollment" }
    val identifier = Identifier(store, model)
    fun scored(samples: List<ValidationSample>) = samples.map { it to identifier.candidates(it.embedding, it.model) }
    fun measure(samples: List<Pair<ValidationSample, List<Candidate>>>, policy: MatchPolicy): Evaluation {
        var known = 0; var unknown = 0; var correct = 0; var falseAccepts = 0
        samples.forEach { (sample, candidates) ->
            val match = resolveUnique(listOf(candidates), policy).first()
            if (sample.identityId == null) { unknown++; if (match.identityId != null) falseAccepts++ }
            else { known++; if (match.identityId == sample.identityId) correct++ }
        }
        return Evaluation(if (known > 0) correct.toDouble() / known else 0.0, known, unknown, falseAccepts)
    }
    val tuned = scored(tuning); val heldOut = scored(evaluation)
    require(thresholds.isNotEmpty() && margins.isNotEmpty())
    val best = thresholds.flatMap { t -> margins.map { m -> val p = MatchPolicy(t, m); p to measure(tuned, p) } }
        .maxWith(compareBy<Pair<MatchPolicy, Evaluation>> { -it.second.falseAccepts }.thenBy { it.second.knownAccuracy }
            .thenBy { it.first.threshold }.thenBy { it.first.margin })
    val final = measure(heldOut, best.first)
    val sufficient = ids.size >= 2 && listOf(best.second, final).all { it.knownTests >= minKnown && it.unknownTests >= minUnknown }
    val passed = sufficient && listOf(best.second, final).all { it.knownAccuracy >= minAccuracy && it.falseAccepts == 0 }
    return CalibrationReport(best.first, best.second, final, passed, when {
        passed -> "Calibration passed on separate tuning and evaluation sets."
        !sufficient -> "Insufficient known/unknown validation samples in separate tuning and evaluation sets."
        else -> "Held-out accuracy or unknown rejection failed; add representative samples and recalibrate."
    })
}
