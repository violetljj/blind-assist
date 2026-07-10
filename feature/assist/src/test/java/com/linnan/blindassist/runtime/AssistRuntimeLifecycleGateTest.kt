package com.linnan.blindassist.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class AssistRuntimeLifecycleGateTest {
    @Test
    fun restartInvalidatesOldLeaseAndAdvancesGeneration() {
        val gate = AssistRuntimeLifecycleGate()
        val firstToken = gate.startSession {}
        val firstLease = gate.tryEnterFrame()
        assertNotNull(firstLease)

        gate.stopSession {}
        assertNull(gate.tryEnterFrame())

        val secondToken = gate.startSession {}
        assertTrue(secondToken.generation > firstToken.generation)
        assertFalse(gate.isCurrent(firstToken))
        assertTrue(gate.isCurrent(secondToken))
        assertNull(gate.commitIfCurrent(requireNotNull(firstLease)) { "stale" })
        firstLease.close()
    }

    @Test
    fun stopWaitsForCurrentCommitBeforeResettingState() {
        val gate = AssistRuntimeLifecycleGate()
        gate.startSession {}
        val lease = requireNotNull(gate.tryEnterFrame())
        val commitStarted = CountDownLatch(1)
        val releaseCommit = CountDownLatch(1)
        val stopReturned = CountDownLatch(1)
        val events = mutableListOf<String>()

        val commitWorker = Thread {
            gate.commitIfCurrent(lease) {
                events += "commit-start"
                commitStarted.countDown()
                releaseCommit.await(2, TimeUnit.SECONDS)
                events += "commit-end"
            }
            lease.close()
        }
        commitWorker.start()
        assertTrue(commitStarted.await(1, TimeUnit.SECONDS))

        val stopWorker = Thread {
            gate.stopSession { events += "reset" }
            stopReturned.countDown()
        }
        stopWorker.start()
        assertFalse(stopReturned.await(150, TimeUnit.MILLISECONDS))
        releaseCommit.countDown()
        assertTrue(stopReturned.await(1, TimeUnit.SECONDS))
        commitWorker.join(1_000L)
        stopWorker.join(1_000L)
        assertEquals(listOf("commit-start", "commit-end", "reset"), events)
    }

    @Test
    fun shutdownRejectsFutureFramesAndRunsCleanupAfterLastLease() {
        val gate = AssistRuntimeLifecycleGate()
        gate.startSession {}
        val lease = requireNotNull(gate.tryEnterFrame())
        var cleanupCalls = 0

        gate.shutdown(resetState = {}, onIdle = { cleanupCalls += 1 })

        assertFalse(gate.isAcceptingFrames())
        assertNull(gate.tryEnterFrame())
        assertEquals(0, cleanupCalls)
        lease.close()
        lease.close()
        assertEquals(1, cleanupCalls)
    }
}
