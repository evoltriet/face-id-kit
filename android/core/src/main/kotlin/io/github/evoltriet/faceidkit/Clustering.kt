package io.github.evoltriet.faceidkit

/** Batch-enrollment suggestions only. A -1 label is an individual, not a discarded face.
 * Mirrors the Python normalized-Euclidean DBSCAN helper without an N-by-N matrix.
 * Input order defines deterministic border-point ownership; callers should sort opaque IDs.
 */
fun clusterEmbeddings(vectors: List<FloatArray>, model: ModelSpec, eps: Double = .72,
                      minSamples: Int = 2, cancelled: () -> Boolean = { false }): IntArray {
    require(eps > 0 && eps.isFinite() && minSamples > 0)
    val matrix = vectors.map { checkModel(model, model, it) }
    val labels = IntArray(matrix.size) { -2 }
    fun neighbors(i: Int): List<Int> {
        check(!cancelled()) { "Clustering cancelled" }
        return matrix.indices.filter { j ->
            var distance = 0.0
            for (k in matrix[i].indices) {
                val d = matrix[i][k].toDouble() - matrix[j][k]
                distance += d * d
            }
            distance <= eps * eps
        }
    }
    var cluster = 0
    for (i in matrix.indices) {
        if (labels[i] != -2) continue
        val near = neighbors(i)
        if (near.size < minSamples) { labels[i] = -1; continue }
        labels[i] = cluster
        val queue = ArrayDeque<Int>()
        val queued = BooleanArray(matrix.size)
        fun enqueue(j: Int) { if (!queued[j]) { queued[j] = true; queue.addLast(j) } }
        near.forEach(::enqueue)
        while (queue.isNotEmpty()) {
            val j = queue.removeFirst()
            if (labels[j] == -1) labels[j] = cluster
            if (labels[j] != -2) continue
            labels[j] = cluster
            val adjacent = neighbors(j)
            if (adjacent.size >= minSamples) adjacent.forEach(::enqueue)
        }
        cluster++
    }
    return labels
}
