package io.github.evoltriet.faceidkit
import com.google.gson.JsonParser
import java.io.File
import kotlin.test.*

class ParityWorkflowTest {
    private val data = JsonParser.parseString(File("fixtures/workflow_parity.json").readText()).asJsonObject
    private val model = ModelSpec("synthetic-v1", 4, "test-l2")
    @Test fun sharedTracking() {
        val tracker = TrackConsensus()
        data["tracking"].asJsonArray.forEach { node ->
            val row = node.asJsonObject
            val boxes = row["boxes"].asJsonArray.map { it.asJsonArray.map { n -> n.asDouble }.let { v -> Box(v[0],v[1],v[2],v[3]) } }
            val results = tracker.update(boxes, row["identities"].asJsonArray.map { if(it.isJsonNull) null else it.asString }, row["now"].asDouble)
            assertEquals(row["tracks"].asJsonArray.map { it.asLong }, results.map { it.first })
            assertEquals(row["stable"].asJsonArray.map { it.asBoolean }, results.map { it.second })
        }
    }
    @Test fun sharedScoringAndRevision() {
        val fixture = JsonParser.parseString(File("fixtures/parity.json").readText()).asJsonObject
        val store = InMemoryStore()
        fixture["samples"].asJsonArray.forEach {
            val s = it.asJsonObject
            store.putIdentity(Identity(s["identityId"].asString))
            store.putSample(EnrollmentSample(s["id"].asString, s["identityId"].asString, s["embedding"].asJsonArray.map { it.asFloat }.toFloatArray(), model))
        }
        val engine = Identifier(store, model)
        fixture["cases"].asJsonArray.forEachIndexed { i, case ->
            val queries = case.asJsonObject["queries"].asJsonArray.map { it.asJsonArray.map { n -> n.asFloat }.toFloatArray() to model }
            engine.identifyMany(queries).forEachIndexed { j, result ->
                val expected = data["scores"].asJsonArray[i].asJsonArray[j].asJsonArray
                assertEquals(expected[0].asDouble, result.score, 1e-6)
                assertEquals(expected[1].asDouble, result.margin, 1e-6)
            }
        }
        val fresh = InMemoryStore(); val revisions = mutableListOf(fresh.revision)
        fresh.putIdentity(Identity("a")); revisions.add(fresh.revision)
        fresh.putIdentity(Identity("b")); revisions.add(fresh.revision)
        val sample = EnrollmentSample("s","a",floatArrayOf(1f,0f,0f,0f),model)
        fresh.putSample(sample); revisions.add(fresh.revision)
        val matching = Identifier(fresh, model)
        assertEquals("a", matching.identify(sample.embedding,model).identityId)
        fresh.putSample(sample.copy(identityId="b")); revisions.add(fresh.revision)
        assertEquals("b", matching.identify(sample.embedding,model).identityId)
        fresh.deleteSample("s"); revisions.add(fresh.revision)
        assertNull(matching.identify(sample.embedding,model).identityId)
        fresh.deleteIdentity("b"); revisions.add(fresh.revision)
        assertEquals(data["revisions"].asJsonArray.map { it.asLong }, revisions)
    }
    @Test fun sharedCalibrationPolicyAndSeparation() {
        fun vector(i: Int) = FloatArray(4) { if(it == i) 1f else 0f }
        val enrollment = listOf("a","b").mapIndexed { i,id -> EnrollmentSample("enroll-$id",id,vector(i),model,sourceId="source-$id") }
        fun samples(prefix: String, offset: Float) = (0..14).map { i ->
            val v = vector(if(i < 10) i%2 else 2); v[3]=offset+i*.001f
            ValidationSample("$prefix-$i",if(i < 10) if(i%2==0) "a" else "b" else null,v,model,"$prefix-source-$i")
        }
        val tuning=samples("t",.01f); val evaluation=samples("e",.04f)
        val report=calibrate(enrollment,tuning,evaluation,model)
        val expected=data["calibration"].asJsonObject
        assertEquals(expected["threshold"].asDouble,report.policy.threshold)
        assertEquals(expected["margin"].asDouble,report.policy.margin)
        assertEquals(expected["passed"].asBoolean,report.passed)
        assertEquals(expected["knownAccuracy"].asDouble,report.evaluation.knownAccuracy)
        assertEquals(expected["knownTests"].asInt,report.evaluation.knownTests)
        assertEquals(expected["unknownTests"].asInt,report.evaluation.unknownTests)
        assertEquals(expected["falseAccepts"].asInt,report.evaluation.falseAccepts)
        assertFalse(calibrate(enrollment,tuning,evaluation.filter { it.identityId != null },model).passed)
        assertEquals(report.policy,calibrate(enrollment,tuning,evaluation.map { it.copy(identityId=null) },model).policy)
        assertFailsWith<IllegalArgumentException> { calibrate(enrollment,tuning,tuning,model) }
    }
}

