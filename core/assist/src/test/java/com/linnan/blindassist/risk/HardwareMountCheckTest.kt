package com.linnan.blindassist.risk

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HardwareMountCheckTest {
    private fun capture(rows: IntRange? = null, status: Int = 5, count: Int = 5): HardwareMountCheck.Capture =
        HardwareMountCheck.Capture(List(count) {
            (0 until 64).map { zone -> DemoTofCell(zone,
                if (rows != null && zone / 8 in rows && zone % 8 in 3..4) 1000.0 else 2000.0,
                status, 1, "KNOWN") }
        })

    @Test fun expectedOrientationIsOnlyRoughlyConsistent() {
        val r = HardwareMountCheck.evaluate(capture(), capture(3..4), capture(6..7), capture(0..1))
        assertTrue(r.consistent)
        assertTrue(r.positions[1].row > r.positions[0].row)
        assertTrue(r.message.contains("未完成标定"))
    }
    @Test fun mirroredOrientationFails() {
        assertFalse(HardwareMountCheck.evaluate(capture(), capture(3..4), capture(0..1), capture(6..7)).consistent)
    }
    @Test fun constantSceneCannotPass() {
        assertFalse(HardwareMountCheck.evaluate(capture(), capture(), capture(), capture()).consistent)
    }
    @Test fun invalidOrInsufficientEvidenceCannotPass() {
        assertFalse(HardwareMountCheck.evaluate(capture(status = 0), capture(3..4), capture(6..7), capture(0..1)).consistent)
        assertFalse(HardwareMountCheck.evaluate(capture(count = 4), capture(3..4), capture(6..7), capture(0..1)).consistent)
        val bad = capture().copy(frames = List(5) { List(64) { DemoTofCell(0, 2000.0, 5, 1, "KNOWN") } })
        assertFalse(HardwareMountCheck.evaluate(bad, capture(3..4), capture(6..7), capture(0..1)).consistent)
    }
}
