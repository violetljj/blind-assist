package com.linnan.blindassist.device.glasses

import com.linnan.blindassist.risk.DemoTofCell
import com.linnan.blindassist.risk.TofCorridorDemo
import org.junit.Assert.*
import org.junit.Test

class HardwareWifiAvailabilityTest {
    private fun availability(cameraReceived: Long? = 1000, tofReceived: Long? = 1000,
        active: Boolean = true, cameraStatus: Boolean = true, now: Long = 1100) =
        HardwareWifiAvailability.evaluate(active, cameraStatus, now,
            cameraReceived, 50, tofReceived, 50)

    @Test fun fullPairMeansBothIndependentStreamsAreFresh() {
        val both = availability()
        assertTrue(both.liveUsable)
        assertTrue(both.cameraUsable)
        assertTrue(both.tofUsable)
    }

    @Test fun cameraLossRetainsTofButNeverFullPair() {
        for (state in listOf(availability(cameraReceived = null),
            availability(cameraReceived = 0), availability(cameraStatus = false))) {
            assertFalse(state.cameraUsable)
            assertFalse(state.liveUsable)
            assertTrue(state.tofUsable)
            val cells = (0 until 64).map { DemoTofCell(it, 1000.0, 5, 1, "KNOWN") }
            assertTrue(TofCorridorDemo.evaluate(8, 8, cells, state.tofUsable).alert)
        }
    }

    @Test fun tofLossRetainsCameraAndInvalidatesJudgment() {
        for (state in listOf(availability(tofReceived = null), availability(tofReceived = 0))) {
            assertTrue(state.cameraUsable)
            assertFalse(state.tofUsable)
            assertFalse(state.liveUsable)
            val cells = (0 until 64).map { DemoTofCell(it, 1000.0, 5, 1, "KNOWN") }
            val decision = TofCorridorDemo.evaluate(8, 8, cells, state.tofUsable)
            assertFalse(decision.alert)
            assertEquals("UNKNOWN", decision.state)
        }
    }

    @Test fun stoppedOrBothStaleClearsBothStreams() {
        for (state in listOf(availability(active = false), availability(now = 1751))) {
            assertFalse(state.cameraUsable)
            assertFalse(state.tofUsable)
            assertFalse(state.liveUsable)
        }
    }

    @Test fun freshnessBoundaryIsUnchangedAndRecoveryIsIndependent() {
        assertTrue(availability(now = 1700).liveUsable)
        val recovered = availability(cameraReceived = 1600, now = 1701)
        assertTrue(recovered.cameraUsable)
        assertFalse(recovered.tofUsable)
        assertTrue(availability(cameraReceived = 1600, tofReceived = 1700, now = 1701).liveUsable)
        assertFalse(availability(now = 999).liveUsable)
    }

    @Test fun freshTransportDoesNotPromiseValidZonesOrClearSpace() {
        val state = availability(cameraReceived = null)
        val invalid = (0 until 64).map { DemoTofCell(it, 0.0, 0, 0, "KNOWN") }
        val decision = TofCorridorDemo.evaluate(8, 8, invalid, state.tofUsable)
        assertTrue(state.tofUsable)
        assertFalse(decision.alert)
        assertEquals(0, decision.validZones)
        assertEquals("UNKNOWN", decision.state)
    }
}
