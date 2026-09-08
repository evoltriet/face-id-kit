package io.github.evoltriet.faceidkit

import com.google.gson.JsonParser
import org.junit.Test
import org.junit.Assert.*
import java.io.File

class ClusteringTest {
    private val model = ModelSpec("test", 3, "l2")
    @Test fun sharedParityAndSingletons() {
        val data = JsonParser.parseString(File("fixtures/clustering_parity.json").readText()).asJsonObject
        val vectors = data["vectors"].asJsonArray.map { row -> row.asJsonArray.map { it.asFloat }.toFloatArray() }
        assertArrayEquals(data["labels"].asJsonArray.map { it.asInt }.toIntArray(), clusterEmbeddings(vectors, model))
        assertArrayEquals(intArrayOf(-1), clusterEmbeddings(listOf(floatArrayOf(1f,0f,0f)), model))
        assertEquals(0, clusterEmbeddings(emptyList(), model).size)
    }
    @Test fun duplicatesAndInputValidation() {
        assertArrayEquals(intArrayOf(0,0), clusterEmbeddings(listOf(floatArrayOf(2f,0f,0f), floatArrayOf(1f,0f,0f)), model))
        assertThrows(IllegalArgumentException::class.java) { clusterEmbeddings(listOf(floatArrayOf(1f)), model) }
        assertThrows(IllegalStateException::class.java) { clusterEmbeddings(listOf(floatArrayOf(1f,0f,0f)), model, cancelled = { true }) }
    }
}
