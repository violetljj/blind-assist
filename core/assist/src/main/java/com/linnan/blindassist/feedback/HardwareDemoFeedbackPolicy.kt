package com.linnan.blindassist.feedback

enum class HardwareDemoFeedbackEvent(val spokenText: String) {
    OBSTACLE("前方可能有障碍"),
    CLOSER("前方测距变近，请留意"),
    REMINDER("前方仍可能有障碍"),
    CONNECTION_LOST("硬件数据已中断，暂时无法判断"),
    RECOVERED("硬件数据已恢复"),
    RECOVERED_OBSTACLE("硬件数据已恢复，前方可能有障碍"),
}

/** Event feedback for the demonstration, not an independent obstacle detector.
 * Call with a monotonic clock and fresh transport usability (not the risk verdict).
 * A non-alert verdict, including UNKNOWN, never generates a clear-space message.
 */
class HardwareDemoFeedbackPolicy {
    private var wasUsable = false
    private var lost = false
    private var obstacleActive = false
    private var releaseSince: Long? = null
    private var announcedAt = 0L
    private var announcedMm: Int? = null
    private var lastNow: Long? = null

    fun reset() {
        wasUsable = false
        lost = false
        obstacleActive = false
        releaseSince = null
        announcedAt = 0L
        announcedMm = null
        lastNow = null
    }

    fun update(
        nowMs: Long,
        live: Boolean,
        usable: Boolean,
        alert: Boolean,
        nearestMm: Int?,
    ): HardwareDemoFeedbackEvent? {
        if (!live) {
            reset()
            return null
        }
        if (lastNow?.let { nowMs < it } == true) reset()
        lastNow = nowMs
        if (!usable) {
            obstacleActive = false
            releaseSince = null
            if (wasUsable && !lost) {
                lost = true
                return HardwareDemoFeedbackEvent.CONNECTION_LOST
            }
            return null
        }
        wasUsable = true
        val recovering = lost
        lost = false
        val distance = nearestMm?.takeIf { it > 0 }
        if (!alert) {
            if (obstacleActive) {
                val since = releaseSince ?: nowMs.also { releaseSince = it }
                if (nowMs - since >= 1_000L) obstacleActive = false
            }
            return if (recovering) HardwareDemoFeedbackEvent.RECOVERED else null
        }
        // A return after a full release interval starts a new event, even when
        // there was no poll exactly at the release deadline.
        if (releaseSince?.let { nowMs - it >= 1_000L } == true) obstacleActive = false
        releaseSince = null
        val event = when {
            recovering -> HardwareDemoFeedbackEvent.RECOVERED_OBSTACLE
            !obstacleActive -> HardwareDemoFeedbackEvent.OBSTACLE
            nowMs - announcedAt >= 1_000L && distance != null &&
                announcedMm?.let { it - distance >= 300 } == true -> HardwareDemoFeedbackEvent.CLOSER
            nowMs - announcedAt >= 12_000L -> HardwareDemoFeedbackEvent.REMINDER
            else -> null
        }
        if (event != null) {
            obstacleActive = true
            announcedAt = nowMs
            announcedMm = distance
        }
        return event
    }
}
