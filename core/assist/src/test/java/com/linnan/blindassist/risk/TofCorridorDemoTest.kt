package com.linnan.blindassist.risk

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TofCorridorDemoTest {
    private fun grid(side: Int = 4) = (0 until side * side).map {
        DemoTofCell(it, 0.0, 0, 0, "UNKNOWN")
    }

    private fun withReturn(zone: Int, mm: Double, side: Int = 4): List<DemoTofCell> =
        grid(side).map { if (it.zone == zone) it.copy(rangeMm = mm, status = 5, targets = 1, quality = "KNOWN") else it }

    @Test fun independentValidReturnSurvivesUnknownNeighbours() {
        val result = TofCorridorDemo.evaluate(4, 4, withReturn(5, 600.0), true)
        assertTrue(result.alert)
        assertEquals("ALERT_SUPPORTED", result.state)
        assertEquals(listOf(5), result.triggeringZones)
        assertEquals(600, result.nearestMm)
        assertEquals(1, result.validZones)
    }

    @Test fun malformedGridAndUnavailableInputCannotAlert() {
        val cells = withReturn(5, 600.0)
        listOf(
            TofCorridorDemo.evaluate(4, 4, cells, false),
            TofCorridorDemo.evaluate(2, 2, cells.take(4), true),
            TofCorridorDemo.evaluate(4, 8, cells, true),
            TofCorridorDemo.evaluate(4, 4, cells.dropLast(1), true),
            TofCorridorDemo.evaluate(4, 4, cells.dropLast(1) + cells[0], true),
            TofCorridorDemo.evaluate(4, 4, cells.dropLast(1) + cells.last().copy(zone = 16), true)
        ).forEach {
            assertFalse(it.alert)
            assertEquals("UNKNOWN", it.state)
            assertEquals(0, it.validZones)
        }
    }

    @Test fun invalidMeasurementsAreNotTrusted() {
        val cells = withReturn(5, 600.0)
        val original = cells[5]
        listOf(
            original.copy(rangeMm = Double.NaN), original.copy(rangeMm = Double.POSITIVE_INFINITY),
            original.copy(rangeMm = -1.0), original.copy(rangeMm = 0.0), original.copy(rangeMm = 3.0),
            original.copy(status = 9), original.copy(targets = 0), original.copy(quality = "SUSPECT")
        ).forEach { invalid ->
            val result = TofCorridorDemo.evaluate(4, 4, cells.map { if (it.zone == 5) invalid else it }, true)
            assertFalse(result.alert)
            assertEquals("UNKNOWN", result.state)
            assertEquals(0, result.validZones)
        }
    }

    @Test fun wholeFootprintCanIntersectWhenZoneCentreWouldMiss() {
        // Zone 0 centre at this range is laterally outside 0.3m, its inner edge is not.
        val result = TofCorridorDemo.evaluate(4, 4, withReturn(0, 1500.0), true)
        assertTrue(result.alert)
        assertEquals("ALERT_AMBIGUOUS", result.state)
    }

    @Test fun rawColumnsMapUpwardWithAsymmetricVerticalCorridor() {
        val down = TofCorridorDemo.evaluate(4, 4, withReturn(4, 1800.0), true)
        val up = TofCorridorDemo.evaluate(4, 4, withReturn(7, 1800.0), true)
        assertTrue(down.alert)
        assertFalse(up.alert)
        assertEquals("UNKNOWN", up.state)
    }

    @Test fun radialDistanceBeyondThreeMetresCanStillSupportAxialOverlap() {
        // The radial lower bound exceeds 3m; the footprint's axial lower bound does not.
        val result = TofCorridorDemo.evaluate(4, 4, withReturn(5, 3350.0), true)
        assertTrue(result.alert)
        assertEquals("ALERT_AMBIGUOUS", result.state)
    }

    @Test fun outsideOnlyIsUnknownRatherThanFreeSpace() {
        val result = TofCorridorDemo.evaluate(4, 4, withReturn(5, 5000.0), true)
        assertFalse(result.alert)
        assertEquals("UNKNOWN", result.state)
        assertEquals(1, result.validZones)
        assertEquals(null, result.nearestMm)
    }

    @Test fun eightByEightAndInputOrderingAreSupported() {
        val cells = withReturn(27, 600.0, 8).map {
            if (it.zone == 35) it.copy(rangeMm = 800.0, status = 5, targets = 1, quality = "KNOWN") else it
        }.reversed()
        val result = TofCorridorDemo.evaluate(8, 8, cells, true)
        assertTrue(result.alert)
        assertEquals(listOf(27, 35), result.triggeringZones)
        assertEquals(600, result.nearestMm)
        assertEquals(2, result.validZones)
    }
}
