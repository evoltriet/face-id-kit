package io.github.evoltriet.faceidkit

import com.google.gson.Gson
import java.io.File
import kotlin.test.*

private data class SampleFixture(val id: String, val identityId: String, val embedding: FloatArray)
private data class CaseFixture(val queries: List<FloatArray>, val identities: List<String?>, val reasons: List<String>)
private data class Fixture(val model: ModelSpec, val samples: List<SampleFixture>, val cases: List<CaseFixture>)

class CoreTest {
    private val model = ModelSpec("synthetic-v1", 4, "test-l2")
    private fun vector(i: Int) = FloatArray(4) { if (it == i) 1f else 0f }
    @Test fun sharedPythonFixtures() {
        val fixture = Gson().fromJson(File("fixtures/parity.json").readText(), Fixture::class.java)
        val store = InMemoryStore()
        fixture.samples.forEach { store.putIdentity(Identity(it.identityId)); store.putSample(EnrollmentSample(it.id, it.identityId, it.embedding, fixture.model)) }
        val identifier = Identifier(store, fixture.model)
        fixture.cases.forEach { case ->
            val results = identifier.identifyMany(case.queries.map { it to fixture.model })
            assertEquals(case.identities, results.map { it.identityId })
            assertEquals(case.reasons, results.map { it.reason })
        }
        assertEquals(1.0, galleryScore(vector(0), listOf(vector(0))), .000001)
    }
    @Test fun updatesInvalidateAndCopiesAreIndependent() {
        val store = InMemoryStore(); store.putIdentity(Identity("a")); store.putIdentity(Identity("b"))
        val sample = EnrollmentSample("s", "a", vector(0), model)
        store.putSample(sample); val engine = Identifier(store, model)
        assertEquals("a", engine.identify(vector(0), model).identityId)
        store.snapshot().samples[0].embedding[0] = 0f
        assertEquals("a", engine.identify(vector(0), model).identityId)
        store.putSample(sample.copy(identityId = "b"))
        assertEquals("b", engine.identify(vector(0), model).identityId)
        store.deleteSample("s"); assertNull(engine.identify(vector(0), model).identityId)
        store.putSample(sample); store.deleteIdentity("a"); assertTrue(store.snapshot().samples.isEmpty())
    }
    @Test fun invalidVectorsAndModels() {
        listOf(floatArrayOf(), floatArrayOf(0f), floatArrayOf(Float.NaN)).forEach { assertFailsWith<IllegalArgumentException> { normalizeEmbedding(it) } }
        assertFailsWith<IllegalArgumentException> { checkModel(model, model.copy(preprocessing = "different"), vector(0)) }
        assertFailsWith<IllegalArgumentException> { checkModel(model, model, floatArrayOf(1f)) }
    }
    @Test fun trackingOrderingCrossingExpiry() {
        val left = Box(0.0, 0.0, 10.0, 10.0); val right = Box(30.0, 0.0, 10.0, 10.0)
        val tracker = TrackConsensus(required = 2)
        val first = tracker.update(listOf(left, right), listOf("a", "b"), 0.0)
        val second = tracker.update(listOf(right, left), listOf("b", "a"), .1)
        assertEquals(first.map { it.first }.reversed(), second.map { it.first }); assertTrue(second.all { it.second })
        assertFalse(tracker.update(listOf(left, left), listOf("a", "b"), .2).any { it.second })
        assertFalse(tracker.update(listOf(left), listOf("a"), 2.0).single().second)
    }
    private fun dataset(): Triple<List<EnrollmentSample>, List<ValidationSample>, List<ValidationSample>> {
        val enrollment = listOf(EnrollmentSample("a", "a", vector(0), model, sourceId = "enroll-a"),
            EnrollmentSample("b", "b", vector(1), model, sourceId = "enroll-b"))
        fun validation(prefix: String, offset: Float): List<ValidationSample> = (0..14).map { i ->
            val v = if (i < 10) vector(i % 2) else vector(2)
            v[3] = offset + i * .001f
            ValidationSample("$prefix-$i", if (i < 10) if (i % 2 == 0) "a" else "b" else null, v, model, "$prefix-source-$i")
        }
        return Triple(enrollment, validation("tune", .01f), validation("eval", .04f))
    }
    @Test fun calibrationPassAndInsufficientUnknowns() {
        val (e, t, v) = dataset()
        assertTrue(calibrate(e, t, v, model).passed)
        assertFalse(calibrate(e, t.filter { it.identityId != null }, v, model).passed)
        val passedPolicy = calibrate(e, t, v, model).policy
        assertEquals(passedPolicy, calibrate(e, t, v.map { it.copy(identityId = null) }, model).policy)
    }
    @Test fun calibrationRejectsOverlap() {
        val (e, t, v) = dataset()
        assertFailsWith<IllegalArgumentException> { calibrate(e, t, t, model) }
        assertFailsWith<IllegalArgumentException> { calibrate(e, t.map { it.copy(sourceId = "enroll-a") }, v, model) }
        assertFailsWith<IllegalArgumentException> { calibrate(e, t.map { it.copy(embedding = vector(0)) }, v, model) }
    }
}
