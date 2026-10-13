package com.linnan.blindassist.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class CaptureCameraPerformanceTest {
    @Test
    fun separatesSourceCadenceAndroidReadAndSlowDurableCallback() {
        val performance = CaptureCameraPerformance()
        performance.received(frame(1, "a", 1_000, 31_000, 50_000_000), 10_000_000, 51_000_000)
        performance.callback(51_000_000, 171_000_000)
        performance.received(frame(4, "a", 37_000, 67_000, 210_000_000), 172_000_000, 211_000_000)
        performance.callback(211_000_000, 213_000_000)
        val receipt = performance.receipt()
        assertEquals(2, receipt.getInt("frames_received"))
        assertEquals(2, receipt.getInt("callbacks_completed"))
        assertEquals(2, receipt.getInt("source_sequence_gap_frames"))
        assertEquals(36.0, receipt.getJSONObject("device_capture_interval_ms").getDouble("recent_p50"), 0.0)
        assertEquals(160.0, receipt.getJSONObject("receipt_interval_ms").getDouble("recent_p50"), 0.0)
        assertEquals(120.0, receipt.getJSONObject("callback_ms").getDouble("all_max"), 0.0)
        assertEquals(30.0, receipt.getJSONObject("device_capture_to_ready_ms").getDouble("recent_p50"), 0.0)
        assertEquals(5.0, receipt.getJSONObject("jpeg_body_read_ms").getDouble("recent_p50"), 0.0)
        assertEquals(20, receipt.getInt("jpeg_bytes_total"))
    }

    @Test
    fun boundedRecentQuantilesRetainFullCountsAndResetDoesNotBridgeSessions() {
        val performance = CaptureCameraPerformance(2)
        for (ms in listOf(1L, 3L, 9L)) performance.callback(0, ms * 1_000_000)
        val metric = performance.receipt().getJSONObject("callback_ms")
        assertEquals(3, metric.getInt("count"))
        assertEquals(2, metric.getInt("retained_samples"))
        assertEquals(6.0, metric.getDouble("recent_p50"), 0.0)
        assertEquals(9.0, metric.getDouble("all_max"), 0.0)
        performance.received(frame(100, "a", 1_000, 2_000, 10_000_000), 0, 10_000_000)
        performance.reset()
        performance.received(frame(1, "a", 1_000, 2_000, 10_000_000), 0, 10_000_000)
        assertEquals(1, performance.receipt().getInt("frames_received"))
        assertEquals(0, performance.receipt().getInt("source_sequence_regressions"))
        assertFalse(performance.receipt().has("callback_ms"))
        assertFalse(performance.receipt().has("receipt_interval_ms"))
    }

    @Test
    fun bootChangeDoesNotTurnDeviceResetIntoLostFramesOrCrossClockDuration() {
        val performance = CaptureCameraPerformance()
        performance.received(frame(100, "a", 30_000, 31_000, 10_000_000), 0, 10_000_000)
        performance.received(frame(1, "b", 1_000, 2_000, 20_000_000), 11_000_000, 20_000_000)
        val receipt = performance.receipt()
        assertEquals(1, receipt.getInt("boot_changes"))
        assertEquals(0, receipt.getInt("source_sequence_gap_frames"))
        assertEquals(0, receipt.getInt("source_sequence_regressions"))
        assertFalse(receipt.has("device_capture_interval_ms"))
    }

    private fun frame(sequence: Long, boot: String, captureUs: Long, readyUs: Long, receiptNs: Long) =
        AtomCameraCaptureFrame(ByteArray(10), receiptNs, 640, 480, sequence, boot, captureUs, readyUs,
            emptyMap(), "fixture", AtomJpegReadTiming(receiptNs - 5_000_000, receiptNs - 4_000_000, 2, 4_000_000))
}
