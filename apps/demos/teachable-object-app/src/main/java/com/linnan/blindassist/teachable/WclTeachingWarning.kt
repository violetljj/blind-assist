package com.linnan.blindassist.teachable

/** Scores a pending clip against the saved WCL library without changing either. */
internal object WclTeachingWarning {
    // One fixed cutoff from normal, pending-filtered ORBIT Development clips (pooled 5% target).
    // Device scores and real teaching attempts have not been calibrated at this operating point.
    const val THRESHOLD = -0.15058053f

    data class Match(val suggested: String, val gap: Float, val retained: Int)

    fun compare(rows: List<FloatArray>, intended: String, saved: WclTeachingStore.Library): Match? {
        val destination = saved.names.indexOf(intended)
        if (destination < 0 || saved.names.size < 2) return null
        require(rows.isNotEmpty())

        val wholeClipScores = WclPipeline.cosine(prototype(rows, rows.indices), saved.prototypes)
        val inferred = wholeClipScores.indices.maxBy { wholeClipScores[it] }
        // Select frames using the visual match of the whole clip, independently of its intended name.
        val margins = rows.map { row ->
            val scores = WclPipeline.cosine(row, saved.prototypes)
            scores[inferred] - scores[bestOther(scores, inferred)]
        }
        val retained = (rows.size + 1) / 2
        val selected = rows.indices.sortedWith(compareByDescending<Int> { margins[it] }.thenBy { it })
            .take(retained).sorted()
        val scores = WclPipeline.cosine(prototype(rows, selected), saved.prototypes)
        val other = bestOther(scores, destination)
        return Match(saved.names[other], scores[other] - scores[destination], retained)
    }

    private fun prototype(rows: List<FloatArray>, indices: Iterable<Int>): FloatArray {
        val sum = FloatArray(WclPipeline.D)
        for (index in indices) {
            val row = rows[index]
            for (d in sum.indices) sum[d] += row[d]
        }
        return WclPipeline.prototypesOf(listOf(sum))
    }

    private fun bestOther(scores: FloatArray, excluded: Int): Int {
        var best = if (excluded == 0) 1 else 0
        for (i in scores.indices) {
            if (i != excluded && scores[i] > scores[best]) best = i
        }
        return best
    }
}
