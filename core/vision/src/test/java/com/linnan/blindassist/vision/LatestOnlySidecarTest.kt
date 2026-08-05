package com.linnan.blindassist.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class LatestOnlySidecarTest {
    @Test
    fun keepsOnlyNewestQueuedInputAndClosesEveryOwnedInput() {
        val executor = Executors.newSingleThreadExecutor()
        val firstStarted = CountDownLatch(1)
        val allowFirstToFinish = CountDownLatch(1)
        val completed = CountDownLatch(2)
        val delivered = mutableListOf<Int>()
        val first = TrackedInput(1)
        val replaced = TrackedInput(2)
        val newest = TrackedInput(3)
        val sidecar = LatestOnlySidecar<TrackedInput, Int>(
            executor = executor,
            maxResultAgeNanos = Long.MAX_VALUE,
            process = { input ->
                if (input.id == 1) {
                    firstStarted.countDown()
                    check(allowFirstToFinish.await(2, TimeUnit.SECONDS))
                }
                input.id
            },
            onFreshResult = { result -> synchronized(delivered) { delivered += result.value }; completed.countDown() }
        )
        try {
            assertTrue(sidecar.submit(first, capturedAtNanos = 1L))
            assertTrue(firstStarted.await(2, TimeUnit.SECONDS))
            assertTrue(sidecar.submit(replaced, capturedAtNanos = 2L))
            assertTrue(sidecar.submit(newest, capturedAtNanos = 3L))
            assertEquals(1, replaced.closeCount.get())

            allowFirstToFinish.countDown()
            assertTrue(completed.await(2, TimeUnit.SECONDS))
            assertEquals(listOf(1, 3), synchronized(delivered) { delivered.toList() })
            assertEquals(1, first.closeCount.get())
            assertEquals(1, newest.closeCount.get())
        } finally {
            sidecar.close()
            executor.shutdownNow()
        }
    }

    @Test
    fun dropsResultThatExceedsAgeLimit() {
        val delivered = AtomicInteger(0)
        val discarded = AtomicInteger(0)
        var now = 100L
        val sidecar = LatestOnlySidecar<TrackedInput, Int>(
            executor = DirectExecutor,
            maxResultAgeNanos = 5L,
            process = { input -> now = 106L; input.id },
            onFreshResult = { delivered.incrementAndGet() },
            onDiscardedResult = { discarded.incrementAndGet() },
            nowNanos = { now }
        )
        val input = TrackedInput(7)
        try {
            assertTrue(sidecar.submit(input, capturedAtNanos = 100L))
            assertEquals(0, delivered.get())
            assertEquals(1, discarded.get())
            assertEquals(1, input.closeCount.get())
        } finally {
            sidecar.close()
        }
    }

    @Test
    fun rejectsAndClosesInputAfterClose() {
        val input = TrackedInput(9)
        val sidecar = LatestOnlySidecar<TrackedInput, Int>(
            executor = DirectExecutor,
            maxResultAgeNanos = 1L,
            process = { it.id },
            onFreshResult = {}
        )
        sidecar.close()
        assertFalse(sidecar.submit(input, capturedAtNanos = 0L))
        assertEquals(1, input.closeCount.get())
    }

    @Test
    fun closeSuppressesAnInFlightResult() {
        val executor = Executors.newSingleThreadExecutor()
        val started = CountDownLatch(1)
        val allowFinish = CountDownLatch(1)
        val delivered = AtomicInteger(0)
        val discarded = AtomicInteger(0)
        val sidecar = LatestOnlySidecar<TrackedInput, Int>(
            executor = executor,
            maxResultAgeNanos = Long.MAX_VALUE,
            process = { input ->
                started.countDown()
                check(allowFinish.await(2, TimeUnit.SECONDS))
                input.id
            },
            onFreshResult = { delivered.incrementAndGet() },
            onDiscardedResult = { discarded.incrementAndGet() }
        )
        try {
            assertTrue(sidecar.submit(TrackedInput(11), capturedAtNanos = 0L))
            assertTrue(started.await(2, TimeUnit.SECONDS))
            sidecar.close()
            allowFinish.countDown()
            executor.shutdown()
            assertTrue(executor.awaitTermination(2, TimeUnit.SECONDS))
            assertEquals(0, delivered.get())
            assertEquals(1, discarded.get())
        } finally {
            sidecar.close()
            executor.shutdownNow()
        }
    }

    @Test
    fun slowResultCallbackDoesNotBlockNewSubmission() {
        val worker = Executors.newSingleThreadExecutor()
        val submitter = Executors.newSingleThreadExecutor()
        val callbackStarted = CountDownLatch(1)
        val allowCallbackToFinish = CountDownLatch(1)
        val callbackCount = AtomicInteger(0)
        val sidecar = LatestOnlySidecar<TrackedInput, Int>(
            executor = worker,
            maxResultAgeNanos = Long.MAX_VALUE,
            process = { it.id },
            onFreshResult = {
                if (callbackCount.incrementAndGet() == 1) {
                    callbackStarted.countDown()
                    check(allowCallbackToFinish.await(2, TimeUnit.SECONDS))
                }
            }
        )
        try {
            assertTrue(sidecar.submit(TrackedInput(12), capturedAtNanos = 0L))
            assertTrue(callbackStarted.await(2, TimeUnit.SECONDS))

            val accepted = submitter.submit<Boolean> {
                sidecar.submit(TrackedInput(13), capturedAtNanos = 1L)
            }
            assertTrue(accepted.get(500, TimeUnit.MILLISECONDS))
        } finally {
            allowCallbackToFinish.countDown()
            sidecar.close()
            worker.shutdownNow()
            submitter.shutdownNow()
        }
    }

    private class TrackedInput(val id: Int) : AutoCloseable {
        val closeCount = AtomicInteger(0)
        override fun close() { closeCount.incrementAndGet() }
    }

    private data object DirectExecutor : java.util.concurrent.Executor {
        override fun execute(command: Runnable) = command.run()
    }
}
