package com.linnan.blindassist.runtime

import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class AssistRuntimeLifecycleGateTest {
    @Test
    fun stopSessionWaitsForInFlightFrameBeforeReturning() {
        val gate = AssistRuntimeLifecycleGate(idleWaitMs = 2_000L)
        gate.startSession()
        val lease = gate.tryEnterFrame()
        assertNotNull(lease)

        val stopStarted = CountDownLatch(1)
        val stopReturned = CountDownLatch(1)
        var stopped = false
        val worker = Thread {
            stopStarted.countDown()
            stopped = gate.stopSession()
            stopReturned.countDown()
        }

        worker.start()
        assertTrue(stopStarted.await(1, TimeUnit.SECONDS))
        assertFalse(stopReturned.await(150, TimeUnit.MILLISECONDS))

        lease?.close()
        assertTrue(stopReturned.await(1, TimeUnit.SECONDS))
        worker.join(1_000L)
        assertTrue(stopped)
        assertNull(gate.tryEnterFrame())
    }

    @Test
    fun shutdownRejectsFutureFrames() {
        val gate = AssistRuntimeLifecycleGate()
        gate.startSession()

        assertTrue(gate.shutdown())

        assertFalse(gate.isAcceptingFrames())
        assertNull(gate.tryEnterFrame())
    }
}
