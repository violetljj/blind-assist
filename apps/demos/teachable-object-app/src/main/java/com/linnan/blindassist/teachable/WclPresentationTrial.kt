package com.linnan.blindassist.teachable

/** A fresh instance per request. Called on the UI thread after frame scoring. */
internal class WclPresentationTrial(val startedNs: Long) {
    private val smoother = ConfidenceEma(ConfidenceEma.alphaOf(RATE))
    private var runTop = -1
    private var runLength = 0
    private var lastSpoken = -1

    val deadlineNs = startedNs + WINDOW_MS * 1_000_000L

    fun contains(timestampNs: Long): Boolean = timestampNs in startedNs until deadlineNs

    data class Decision(val top: Int, val announce: Boolean)

    fun onFrame(scores: FloatArray, capturedNs: Long, availableNs: Long): Decision? {
        if (!contains(capturedNs) || !contains(availableNs) || capturedNs > availableNs) return null
        val top = smoother.update(scores)
        if (top == runTop) {
            runLength++
        } else {
            runTop = top
            runLength = 1
        }
        val announce = runLength == STABLE_FRAMES && top != lastSpoken
        if (announce) lastSpoken = top
        return Decision(top, announce)
    }

    companion object {
        const val RATE = 3.0
        const val WINDOW_MS = 6_000L
        private const val STABLE_FRAMES = 3
    }
}
