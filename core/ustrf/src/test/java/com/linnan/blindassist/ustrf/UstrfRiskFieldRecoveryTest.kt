package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfRiskFieldRecoveryTest {
    private val coordinate = UstrfGridCoordinate(0, 1)
    private val builder = UstrfRiskFieldBuilder(
        UstrfRiskFieldConfig(staticLifetimeNs = 100L, dynamicLifetimeNs = 50L)
    )

    @Test
    fun freshReliableObservationRecoversAgedUnknownWithoutReset() {
        builder.update(packet(100L, listOf(observation(.10f, "camera"))))
        val aged = builder.update(packet(180L, emptyList())).cellAt(coordinate)
        assertTrue(aged.isUnknown)

        val recovered = builder.update(packet(181L, listOf(observation(.10f, "camera")))).cellAt(coordinate)

        assertEquals(.10f, recovered.uncertainty, 0f)
        assertEquals(0L, recovered.ageNs)
        assertFalse(recovered.isUnknown)
    }

    @Test
    fun fullyExpiredCellIsPrunedAndCanBeRecreatedKnown() {
        builder.update(packet(100L, listOf(observation(.10f, "camera"))))
        val expired = builder.update(packet(200L, emptyList()))
        assertFalse(coordinate in expired.cells)
        assertTrue(expired.cellAt(coordinate).isUnknown)

        val recreated = builder.update(packet(201L, listOf(observation(.10f, "camera"))))
        assertTrue(coordinate in recreated.cells)
        assertFalse(recreated.cellAt(coordinate).isUnknown)
    }

    @Test
    fun currentSourcesAggregateConservativelyWithoutAgedUncertainty() {
        builder.update(packet(100L, listOf(observation(.10f, "old"))))
        builder.update(packet(180L, emptyList()))

        val recovered = builder.update(
            packet(
                181L,
                listOf(observation(.10f, "depth"), observation(.40f, "motion"))
            )
        ).cellAt(coordinate)

        assertEquals(.40f, recovered.uncertainty, 0f)
        assertTrue(recovered.sources.containsAll(setOf("old", "depth", "motion")))
    }

    private fun packet(timeNs: Long, observations: List<UstrfRiskObservation>) = UstrfPerceptionPacket(
        sourceFrame = UstrfFrameStamp(timeNs, timeNs, "user-local-v1"),
        producedAtNs = timeNs,
        validUntilNs = timeNs + 1_000L,
        observations = observations
    )

    private fun observation(uncertainty: Float, source: String) = UstrfRiskObservation(
        coordinate = coordinate,
        occupancy = .20f,
        traversability = .80f,
        dropRisk = 0f,
        headRisk = 0f,
        dynamicTtcMs = null,
        uncertainty = uncertainty,
        source = source,
        validUntilNs = 10_000L
    )
}
