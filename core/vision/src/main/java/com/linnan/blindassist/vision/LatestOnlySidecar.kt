package com.linnan.blindassist.vision

import java.util.concurrent.Executor

/**
 * A one-slot asynchronous worker for inference that must never block a
 * real-time caller. Inputs are owned by this class after [submit] returns.
 *
 * CameraX frames themselves must not be submitted: callers first make a small
 * owned copy (for example, a luma plane), then submit that copy. At most one
 * queued input exists while an item is being processed; a newer input replaces
 * and closes the queued one. Results older than [maxResultAgeNanos] are not
 * delivered. This component deliberately has no alert or model semantics.
 */
class LatestOnlySidecar<I : AutoCloseable, O>(
    private val executor: Executor,
    private val maxResultAgeNanos: Long,
    private val process: (I) -> O,
    private val onFreshResult: (Result<O>) -> Unit,
    private val onFailure: (Throwable) -> Unit = {},
    private val nowNanos: () -> Long = System::nanoTime
) : AutoCloseable {
    init {
        require(maxResultAgeNanos >= 0L) { "maxResultAgeNanos must be non-negative" }
    }

    /** Returns false only when already closed; in that case [input] is closed here. */
    fun submit(input: I, capturedAtNanos: Long): Boolean {
        var replaced: Entry<I>? = null
        var scheduleDrain = false
        synchronized(lock) {
            if (closed) {
                input.closeQuietly(onFailure)
                return false
            }
            replaced = pending
            pending = Entry(input, capturedAtNanos)
            if (!draining) {
                draining = true
                scheduleDrain = true
            }
        }
        replaced?.input?.closeQuietly(onFailure)
        if (scheduleDrain) {
            try {
                executor.execute(::drain)
            } catch (failure: Throwable) {
                synchronized(lock) {
                    draining = false
                    pending?.also { queued -> pending = null; queued.input.closeQuietly(onFailure) }
                }
                onFailure(failure)
            }
        }
        return true
    }

    override fun close() {
        val toClose: Entry<I>?
        synchronized(lock) {
            if (closed) return
            closed = true
            toClose = pending
            pending = null
        }
        toClose?.input?.closeQuietly(onFailure)
    }

    private fun drain() {
        while (true) {
            val entry = synchronized(lock) {
                pending?.also { pending = null } ?: run {
                    draining = false
                    return
                }
            }
            val result = try {
                val value = process(entry.input)
                Result(value = value, capturedAtNanos = entry.capturedAtNanos, completedAtNanos = nowNanos())
            } catch (failure: Throwable) {
                onFailure(failure)
                null
            } finally {
                entry.input.closeQuietly(onFailure)
            }
            result?.let(::deliverIfFresh)
        }
    }

    private fun deliverIfFresh(result: Result<O>) {
        val shouldDeliver = synchronized(lock) {
            !closed && result.ageNanos <= maxResultAgeNanos
        }
        if (!shouldDeliver) return
        try {
            onFreshResult(result)
        } catch (failure: Throwable) {
            onFailure(failure)
        }
    }

    private fun AutoCloseable.closeQuietly(report: (Throwable) -> Unit) {
        try {
            close()
        } catch (failure: Throwable) {
            report(failure)
        }
    }

    data class Result<O>(
        val value: O,
        val capturedAtNanos: Long,
        val completedAtNanos: Long
    ) {
        val ageNanos: Long get() = (completedAtNanos - capturedAtNanos).coerceAtLeast(0L)

    }

    private data class Entry<I : AutoCloseable>(val input: I, val capturedAtNanos: Long)

    private val lock = Any()
    private var pending: Entry<I>? = null
    private var draining = false
    private var closed = false
}
