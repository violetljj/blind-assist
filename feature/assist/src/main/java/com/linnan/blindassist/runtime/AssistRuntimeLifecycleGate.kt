package com.linnan.blindassist.runtime

import java.util.concurrent.TimeUnit

internal class AssistRuntimeLifecycleGate(
    private val idleWaitMs: Long = DEFAULT_IDLE_WAIT_MS
) {
    private val lock = Object()
    private var acceptingFrames = false
    private var closed = false
    private var inFlightFrames = 0

    fun startSession() {
        synchronized(lock) {
            if (!closed) {
                acceptingFrames = true
            }
        }
    }

    fun stopSession(): Boolean {
        stopAcceptingFrames()
        return awaitIdle()
    }

    fun shutdown(): Boolean {
        synchronized(lock) {
            closed = true
            acceptingFrames = false
        }
        return awaitIdle()
    }

    fun tryEnterFrame(): FrameLease? {
        synchronized(lock) {
            if (!acceptingFrames || closed) return null
            inFlightFrames += 1
            return FrameLease(this)
        }
    }

    fun isAcceptingFrames(): Boolean {
        return synchronized(lock) { acceptingFrames && !closed }
    }

    private fun stopAcceptingFrames() {
        synchronized(lock) {
            acceptingFrames = false
        }
    }

    private fun awaitIdle(): Boolean {
        val deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(idleWaitMs)
        synchronized(lock) {
            while (inFlightFrames > 0) {
                val remainingNanos = deadlineNanos - System.nanoTime()
                if (remainingNanos <= 0L) return false
                val waitMs = TimeUnit.NANOSECONDS.toMillis(remainingNanos).coerceAtLeast(1L)
                try {
                    lock.wait(waitMs)
                } catch (error: InterruptedException) {
                    Thread.currentThread().interrupt()
                    return false
                }
            }
            return true
        }
    }

    private fun leaveFrame() {
        synchronized(lock) {
            if (inFlightFrames > 0) {
                inFlightFrames -= 1
            }
            if (inFlightFrames == 0) {
                lock.notifyAll()
            }
        }
    }

    class FrameLease internal constructor(
        private val gate: AssistRuntimeLifecycleGate
    ) : AutoCloseable {
        private var closed = false

        override fun close() {
            if (!closed) {
                closed = true
                gate.leaveFrame()
            }
        }
    }

    private companion object {
        const val DEFAULT_IDLE_WAIT_MS = 1_000L
    }
}
