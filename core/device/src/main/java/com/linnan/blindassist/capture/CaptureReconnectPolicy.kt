package com.linnan.blindassist.capture

/** Clock-driven retry state; the foreground owner's control executor performs the work. */
internal class CaptureReconnectPolicy(private val startedNs: Long) {
    private var failures = 0
    private var dueNs: Long? = startedNs
    private var lastProgressNs = startedNs
    private var awaitingData = false

    fun beginIfDue(nowNs: Long): Boolean {
        if (awaitingData && nowNs - lastProgressNs > STALE_NS) failed(nowNs)
        val due = dueNs ?: return false
        if (nowNs < due) return false
        dueNs = null
        awaitingData = true
        lastProgressNs = nowNs
        return true
    }

    fun received(nowNs: Long) {
        // Closing a replaced source drains buffered frames before invalidating
        // its callbacks. An old receipt must not age the new attempt or reset
        // its failure backoff; the frame can still be preserved by the writer.
        if (nowNs < lastProgressNs) return
        failures = 0
        dueNs = null
        awaitingData = true
        lastProgressNs = nowNs
    }

    fun failed(nowNs: Long) {
        if (dueNs != null) return
        awaitingData = false
        val delaySeconds = BACKOFF_SECONDS[failures.coerceAtMost(BACKOFF_SECONDS.lastIndex)]
        failures++
        dueNs = nowNs + delaySeconds * 1_000_000_000L
    }

    private companion object {
        const val STALE_NS = 8_000_000_000L
        val BACKOFF_SECONDS = longArrayOf(1, 2, 4, 8, 10)
    }
}
