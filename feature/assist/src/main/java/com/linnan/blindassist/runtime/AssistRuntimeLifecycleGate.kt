package com.linnan.blindassist.runtime

internal data class SessionToken(val generation: Long)

internal class AssistRuntimeLifecycleGate {
    private val lock = Any()
    private var generation = 0L
    private var acceptingFrames = false
    private var closed = false
    private var inFlightFrames = 0
    private var idleAction: (() -> Unit)? = null

    fun startSession(resetState: () -> Unit): SessionToken {
        return synchronized(lock) {
            check(!closed) { "Cannot start a closed runtime lifecycle" }
            acceptingFrames = false
            generation += 1L
            resetState()
            acceptingFrames = true
            SessionToken(generation)
        }
    }

    fun stopSession(resetState: () -> Unit) {
        synchronized(lock) {
            acceptingFrames = false
            generation += 1L
            resetState()
        }
    }

    fun shutdown(resetState: () -> Unit, onIdle: () -> Unit) {
        var runImmediately: (() -> Unit)? = null
        synchronized(lock) {
            if (closed) return
            closed = true
            acceptingFrames = false
            generation += 1L
            resetState()
            if (inFlightFrames == 0) {
                runImmediately = onIdle
            } else {
                idleAction = onIdle
            }
        }
        runImmediately?.invoke()
    }

    fun tryEnterFrame(): FrameLease? {
        return synchronized(lock) {
            if (!acceptingFrames || closed) return null
            inFlightFrames += 1
            FrameLease(this, SessionToken(generation))
        }
    }

    fun <T> commitIfCurrent(lease: FrameLease, block: () -> T): T? {
        return synchronized(lock) {
            if (!isCurrentLocked(lease.token)) return null
            block()
        }
    }

    fun isCurrent(token: SessionToken): Boolean {
        return synchronized(lock) { isCurrentLocked(token) }
    }

    fun isAcceptingFrames(): Boolean {
        return synchronized(lock) { acceptingFrames && !closed }
    }

    private fun isCurrentLocked(token: SessionToken): Boolean {
        return !closed && acceptingFrames && generation == token.generation
    }

    private fun leaveFrame() {
        var action: (() -> Unit)? = null
        synchronized(lock) {
            if (inFlightFrames > 0) {
                inFlightFrames -= 1
            }
            if (inFlightFrames == 0) {
                action = idleAction
                idleAction = null
            }
        }
        action?.invoke()
    }

    class FrameLease internal constructor(
        private val gate: AssistRuntimeLifecycleGate,
        val token: SessionToken
    ) : AutoCloseable {
        private var closed = false

        override fun close() {
            if (!closed) {
                closed = true
                gate.leaveFrame()
            }
        }
    }
}
