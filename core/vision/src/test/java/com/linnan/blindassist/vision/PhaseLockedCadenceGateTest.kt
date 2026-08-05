package com.linnan.blindassist.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PhaseLockedCadenceGateTest {
    @Test
    fun arrivalJitterDoesNotAccumulateIntoCadenceDrift() {
        val gate = PhaseLockedCadenceGate(periodNanos = 200L)
        val arrivals = (0L..1_005L step 67L).toList()
        val claimed = arrivals.filter(gate::claim)

        assertEquals(listOf(0L, 201L, 402L, 603L, 804L, 1_005L), claimed)
    }

    @Test
    fun missedDeadlinesAreSkippedWithoutCatchUpBurst() {
        val gate = PhaseLockedCadenceGate(periodNanos = 200L)

        assertTrue(gate.claim(0L))
        assertTrue(gate.claim(650L))
        assertFalse(gate.claim(651L))
        assertFalse(gate.claim(799L))
        assertTrue(gate.claim(800L))
    }

    @Test(expected = IllegalArgumentException::class)
    fun periodMustBePositive() {
        PhaseLockedCadenceGate(0L)
    }
}
