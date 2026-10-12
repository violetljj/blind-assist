package com.linnan.blindassist.device.glasses

import com.linnan.blindassist.capture.CnhCaptureRecord
import com.linnan.blindassist.risk.DemoTofCell
import org.json.JSONObject

/** H3 ms is post-read device uptime, NOT an exposure stamp or a phone-clock mapping. */
internal class H3RangeAdapter {
    private var boot: String? = null
    private var previousSequence: Long? = null
    private var previousDeviceMs: Long? = null

    fun accept(record: CnhCaptureRecord, nowNs: Long): H3RangeSample? {
        if (record.kind == "boot" || record.kind == "transport_connection") {
            val nextBoot = runCatching { JSONObject(record.rawLine.toString(Charsets.UTF_8))
                .getString("boot_id").takeIf { it.isNotBlank() } }.getOrNull()
            if (nextBoot != boot) {
                boot = nextBoot; previousSequence = null; previousDeviceMs = null
            }
            return null
        }
        val frame = record.frame ?: return null
        val currentBoot = boot ?: return null
        if (nowNs < record.hostReceivedMonotonicNs ||
            nowNs - record.hostReceivedMonotonicNs > HardwareWifiFreshness.MAX_AGE_MS * 1_000_000L) return null
        if (record.issues.any { !it.startsWith("sequence_gap:") }) return null
        // Forward uint32 progression accepts rollover and gaps but never relabels a duplicate as fresh.
        fun advances(next: Long, previous: Long?) = previous == null ||
            ((next - previous) and 0xffffffffL) in 1L..0x7fffffffL
        if (!advances(frame.sequence, previousSequence) || !advances(frame.deviceMs, previousDeviceMs)) return null
        previousSequence = frame.sequence; previousDeviceMs = frame.deviceMs
        return H3RangeSample(currentBoot, frame.sequence,
            frame.distanceMm.indices.map { zone -> DemoTofCell(zone, frame.distanceMm[zone].toDouble(),
                frame.targetStatus[zone], frame.targetCount[zone], "KNOWN") },
            record.hostReceivedMonotonicNs / 1_000_000L)
    }
}

internal data class H3RangeSample(val boot: String, val sequence: Long,
    val cells: List<DemoTofCell>, val receivedMs: Long)
