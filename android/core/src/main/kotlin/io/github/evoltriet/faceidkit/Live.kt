package io.github.evoltriet.faceidkit

private data class Track(var box: Box, var seen: Double, val history: MutableList<String?> = mutableListOf())
class TrackConsensus(private val window: Int = 6, private val required: Int = 4,
    private val expirySeconds: Double = 1.0, private val minIou: Double = .3) {
    init { require(required in 1..window && expirySeconds > 0 && minIou > 0 && minIou <= 1) }
    private val tracks = linkedMapOf<Long, Track>()
    private var nextId = 0L
    fun reset() { tracks.clear() }
    private fun append(track: Track, id: String?) { track.history.add(id); while (track.history.size > window) track.history.removeAt(0) }
    fun update(boxes: List<Box>, identities: List<String?>, now: Double = System.nanoTime() / 1e9): List<Pair<Long, Boolean>> {
        require(boxes.size == identities.size)
        tracks.entries.removeAll { now - it.value.seen >= expirySeconds }
        val options = boxes.map { box -> tracks.filterValues { box.iou(it.box) >= minIou }.keys.toList() }
        val counts = tracks.keys.associateWith { id -> options.count { id in it } }
        val ambiguous = options.flatMap { choices -> choices.filter { choices.size != 1 || counts[it] != 1 } }.toSet()
        ambiguous.forEach { tracks.remove(it) }
        val used = mutableSetOf<Long>()
        val result = boxes.indices.map { i ->
            val choices = options[i]
            val id = if (choices.size == 1 && choices[0] in tracks && counts[choices[0]] == 1) choices[0] else nextId++
            val track = tracks.getOrPut(id) { Track(boxes[i], now) }
            val identity = identities[i]
            if (track.history.isNotEmpty() && identity != null && track.history.last() != null && track.history.last() != identity) track.history.clear()
            track.box = boxes[i]; track.seen = now; append(track, identity); used.add(id)
            id to (identity != null && track.history.count { it == identity } >= required)
        }
        tracks.filterKeys { it !in used }.values.forEach { append(it, null) }
        return result
    }
}
class LiveSession(private val faces: FaceIdentifier, private val maxFaces: Int = 4,
    private val centralOnly: Boolean = false, val tracker: TrackConsensus = TrackConsensus()) {
    private var revision = faces.store.revision
    @Synchronized fun reset() { tracker.reset() }
    @Synchronized fun update(image: BgrImage, now: Double = System.nanoTime() / 1e9): List<LiveResult> {
        val before = faces.store.revision
        if (before != revision) { tracker.reset(); revision = before }
        val results = faces.identify(image, maxFaces, centralOnly)
        if (before != faces.store.revision) { tracker.reset(); return emptyList() }
        val tracks = tracker.update(results.map { it.first.box }, results.map { it.second.identityId }, now)
        return results.zip(tracks).map { (r, track) -> LiveResult(track.first, r.first, r.second, track.second) }
    }
}
