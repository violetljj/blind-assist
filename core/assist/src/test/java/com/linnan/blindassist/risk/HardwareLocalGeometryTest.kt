package com.linnan.blindassist.risk

import org.junit.Assert.*
import org.junit.Test

class HardwareLocalGeometryTest {
    private fun cells() = List(64) { DemoTofCell(it, 1000.0 + it, 5, 1, "KNOWN") }
    @Test fun rawOrientationAndInvalidRegionsArePreserved() {
        val input = cells().map { if (it.zone == 63) it.copy(status = 255) else it }
        val tokens = HardwareLocalGeometry.tokens(input)
        assertEquals(63, HardwareLocalGeometry.rawZone(0))
        assertEquals(0, HardwareLocalGeometry.rawZone(63))
        assertEquals(0.0, tokens[0][1], 0.0)
        assertEquals(63, tokens.count { it[1] == 1.0 })
        assertTrue(tokens[63][0] * 8 < input[0].rangeMm / 1000)
        assertTrue(tokens[0][2] < tokens[63][2])
        assertTrue(tokens[0][3] < tokens[63][3])
    }
    @Test fun missingAndOutOfDomainRangesCannotEnterModel() {
        val input = cells().mapIndexed { i, c -> when(i) {
            0 -> c.copy(rangeMm = Double.NaN)
            1 -> c.copy(rangeMm = 2.0)
            2 -> c.copy(rangeMm = 20000.0)
            3 -> c.copy(targets = 0)
            else -> c
        } }
        assertEquals(60, HardwareLocalGeometry.tokens(input).count { it[1] == 1.0 })
    }
    @Test(expected = IllegalArgumentException::class)
    fun duplicateZonesRejected() { HardwareLocalGeometry.tokens(cells().map { it.copy(zone = 0) }) }
}
