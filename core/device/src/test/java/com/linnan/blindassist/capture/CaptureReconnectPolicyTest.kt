package com.linnan.blindassist.capture

import org.junit.Assert.*
import org.junit.Test

class CaptureReconnectPolicyTest {
    @Test fun failuresBackOffUntilCappedAndDoNotDuplicatePendingAttempt() {
        var now = 1_000_000_000L
        val policy = CaptureReconnectPolicy(now)
        assertTrue(policy.beginIfDue(now))
        for (seconds in listOf(1L, 2L, 4L, 8L, 10L, 10L)) {
            policy.failed(now)
            policy.failed(now + 10)
            now += seconds * 1_000_000_000L
            assertFalse(policy.beginIfDue(now - 1))
            assertTrue(policy.beginIfDue(now))
            assertFalse(policy.beginIfDue(now))
        }
    }

    @Test fun onlyDataResetsBackoffAndStaleConnectionsRetry() {
        val policy = CaptureReconnectPolicy(1L)
        assertTrue(policy.beginIfDue(1L))
        policy.failed(2L)
        assertTrue(policy.beginIfDue(1_000_000_002L))
        policy.received(2_000_000_000L)
        assertFalse(policy.beginIfDue(10_000_000_000L))
        assertFalse(policy.beginIfDue(10_000_000_001L))
        assertTrue(policy.beginIfDue(11_000_000_001L))
    }

    @Test fun drainingOldFramesCannotShortenNewAttemptFirstFrameWindow() {
        val policy = CaptureReconnectPolicy(1_000_000_000L)
        assertTrue(policy.beginIfDue(1_000_000_000L))
        policy.received(2_000_000_000L)
        assertFalse(policy.beginIfDue(11_000_000_000L))
        assertTrue(policy.beginIfDue(12_000_000_000L))

        // Starting a replacement closes the old source, which drains accepted
        // frames before its generation is invalidated. These receipts are old.
        policy.received(2_500_000_000L)
        assertFalse(policy.beginIfDue(12_200_000_000L))
        assertFalse(policy.beginIfDue(13_200_000_000L))
        assertFalse(policy.beginIfDue(20_000_000_000L))

        // The replacement still gets its first frame inside the normal window.
        policy.received(20_000_000_000L)
        assertFalse(policy.beginIfDue(21_200_000_000L))
    }

    @Test fun oldDrainedFramesDoNotResetReplacementFailureBackoff() {
        val policy = CaptureReconnectPolicy(1_000_000_000L)
        assertTrue(policy.beginIfDue(1_000_000_000L))
        policy.failed(2_000_000_000L)
        assertTrue(policy.beginIfDue(3_000_000_000L))
        policy.received(1_500_000_000L)
        policy.failed(4_000_000_000L)
        assertFalse(policy.beginIfDue(5_000_000_000L))
        assertTrue(policy.beginIfDue(6_000_000_000L))
    }
}
