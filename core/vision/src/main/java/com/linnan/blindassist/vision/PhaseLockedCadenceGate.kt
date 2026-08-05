package com.linnan.blindassist.vision

import java.util.concurrent.atomic.AtomicLong

/**
 * Claims at most one item per fixed monotonic-time period without accumulating
 * arrival jitter. Missed deadlines are skipped rather than replayed.
 */
class PhaseLockedCadenceGate(private val periodNanos: Long) {
    init { require(periodNanos > 0L) }

    private val nextDeadlineNanos = AtomicLong(Long.MIN_VALUE)

    fun claim(nowNanos: Long): Boolean {
        while (true) {
            val deadline = nextDeadlineNanos.get()
            if (deadline == Long.MIN_VALUE) {
                if (nextDeadlineNanos.compareAndSet(deadline, nowNanos + periodNanos)) return true
                continue
            }
            if (nowNanos < deadline) return false
            val elapsedPeriods = (nowNanos - deadline) / periodNanos + 1L
            val nextDeadline = deadline + elapsedPeriods * periodNanos
            if (nextDeadlineNanos.compareAndSet(deadline, nextDeadline)) return true
        }
    }
}
