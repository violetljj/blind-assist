package com.linnan.blindassist.capture

import java.net.ConnectException
import java.net.SocketTimeoutException
import org.junit.Assert.*
import org.junit.Test

class CaptureConnectionDiagnosticsTest {
    @Test fun connectedWithoutValidFramesIsReportedOnceAndRecoveryUsesElapsedClock() {
        val tracker = CaptureConnectionTracker()
        tracker.attempt("tof", "tcp:host", 1_000_000_000L)
        tracker.status("tof", CnhCaptureStatus(CnhConnectionState.CONNECTED, ""), 2_000_000_000L)
        tracker.check(4_000_000_000L)
        assertEquals(CaptureConnectionReason.TRANSPORT_CONNECTED, tracker.snapshot().latest.single().reason)
        tracker.check(4_000_000_001L)
        tracker.check(5_000_000_000L)
        assertEquals(1, tracker.snapshot().history.count { it.reason == CaptureConnectionReason.NO_VALID_FRAMES })
        tracker.frame("tof", 6_000_000_000L)
        val recovered = tracker.snapshot().latest.single()
        assertEquals(CaptureConnectionReason.RECOVERED, recovered.reason)
        assertEquals(5000L, recovered.recoveryElapsedMs)
        assertEquals(1, recovered.attempts)
    }

    @Test fun retryHistoryPreservesFailureAndBootEvidenceWithoutInventingTofReboot() {
        val tracker = CaptureConnectionTracker()
        tracker.attempt("camera", "http:host", 1L)
        tracker.frame("camera", 2L, "boot-a")
        tracker.status("camera", CnhCaptureStatus(CnhConnectionState.ERROR, "timeout",
            CaptureConnectionReason.SOCKET_TIMEOUT), 1_000_000_000L)
        tracker.status("camera", CnhCaptureStatus(CnhConnectionState.STOPPED, ""), 2_000_000_000L)
        assertEquals(CaptureConnectionReason.SOCKET_TIMEOUT, tracker.snapshot().latest.single().reason)
        tracker.attempt("camera", "http:host", 3_000_000_000L)
        tracker.status("camera", CnhCaptureStatus(CnhConnectionState.ERROR, "timeout"), 4_000_000_000L)
        tracker.attempt("camera", "http:host", 5_000_000_000L)
        tracker.frame("camera", 6_000_000_000L, "boot-b")
        assertEquals(1, tracker.snapshot().history.count { it.reason == CaptureConnectionReason.BOOT_ID_CHANGED })
        assertEquals(2, tracker.snapshot().latest.single().attempts)
        assertEquals(5000L, tracker.snapshot().latest.single().recoveryElapsedMs)
        tracker.attempt("tof", "tcp:host", 7_000_000_000L)
        tracker.frame("tof", 8_000_000_000L)
        assertFalse(tracker.snapshot().history.any { it.stream == "tof" && it.reason == CaptureConnectionReason.BOOT_ID_CHANGED })
    }

    @Test fun boundedSnapshotsDoNotChangeAndEndpointChangesDoNotImplyReboot() {
        val tracker = CaptureConnectionTracker()
        tracker.attempt("camera", "host-a", 1L)
        tracker.frame("camera", 2L, "a")
        val earlier = tracker.snapshot()
        tracker.attempt("camera", "host-b", 3L)
        tracker.frame("camera", 4L, "b")
        assertFalse(tracker.snapshot().history.any { it.reason == CaptureConnectionReason.BOOT_ID_CHANGED })
        repeat(100) { tracker.factoryFailed("camera", IllegalArgumentException("bad address"), 10L + it) }
        assertEquals(64, tracker.snapshot().history.size)
        assertEquals(2, earlier.history.size)
        assertEquals(CaptureConnectionReason.RECOVERED, earlier.latest.single().reason)
    }

    @Test fun explicitStopDoesNotProduceAStallAndFailuresStayObservational() {
        val tracker = CaptureConnectionTracker()
        tracker.attempt("imu", "host", 1L)
        tracker.status("imu", CnhCaptureStatus(CnhConnectionState.CONNECTED, ""), 2L)
        tracker.frame("imu", 3L, "a")
        tracker.status("imu", CnhCaptureStatus(CnhConnectionState.STOPPED, ""), 4L)
        tracker.check(10_000_000_000L)
        assertEquals(CaptureConnectionReason.STOPPED, tracker.snapshot().latest.single().reason)
        assertEquals(CaptureConnectionReason.CONNECTION_FAILED, captureFailureReason(ConnectException()))
        assertEquals(CaptureConnectionReason.SOCKET_TIMEOUT, captureFailureReason(SocketTimeoutException()))
        assertEquals(CaptureConnectionReason.SOURCE_ERROR, captureFailureReason(IllegalArgumentException()))
    }
    @Test fun reopenedHistoryDoesNotRestoreReadinessOrComparePreviousBootClock() {
        val old = CaptureConnectionEvent("tof", CaptureConnectionReason.SOCKET_ERROR,
            9_000_000_000L, "old", 2, wallClockMs = 100L)
        val tracker = CaptureConnectionTracker(listOf(old))
        tracker.check(1L)
        assertTrue(tracker.snapshot().latest.isEmpty())
        assertEquals(listOf(old), tracker.snapshot().history)
        tracker.attempt("tof", "host", 10L)
        tracker.frame("tof", 1_000_010L)
        assertEquals(1L, tracker.snapshot().latest.single().recoveryElapsedMs)
    }

    @Test fun intentionalStopClosesFailureEpisodeWithoutErasingHistory() {
        val tracker = CaptureConnectionTracker()
        tracker.attempt("tof", "host", 1L)
        tracker.status("tof", CnhCaptureStatus(CnhConnectionState.ERROR, "failed"), 2L)
        tracker.intentionalStop(3L)
        assertEquals(CaptureConnectionReason.STOPPED, tracker.snapshot().latest.single().reason)
        assertTrue(tracker.snapshot().history.any { it.reason == CaptureConnectionReason.SOURCE_ERROR })
        tracker.attempt("tof", "host", 10_000_000_000L)
        tracker.frame("tof", 10_100_000_000L)
        assertEquals(1, tracker.snapshot().latest.single().attempts)
        assertEquals(100L, tracker.snapshot().latest.single().recoveryElapsedMs)
    }

}
