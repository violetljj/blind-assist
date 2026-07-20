package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfOfflineSafetySimulationTest {
    @Test
    fun fiveCandidateCorridorsSweepTheSyntheticBodyEnvelopeInsteadOfOnlyTheirCenterLines() {
        val frame = UstrfFrameStamp(1L, 1_000L, "synthetic-local-v1")
        val cells = (-3..3).flatMap { lateral ->
            (1..4).map { forward ->
                val coordinate = UstrfGridCoordinate(lateral, forward)
                coordinate to UstrfRiskCell(
                    occupancy = if (coordinate == UstrfGridCoordinate(0, 2)) .95f else 0f,
                    traversability = 1f,
                    uncertainty = 0f,
                    ageNs = 0L,
                    sources = setOf("synthetic-fixture-v1")
                )
            }
        }.toMap()
        val plan = UstrfCorridorPlanner(
            capsuleHalfWidthCells = 1,
            fixedCandidateOffsets = listOf(-2, -1, 0, 1, 2)
        ).plan(UstrfRiskField(frame, cells), UstrfRouteIntent("synthetic-local-v1", 0, 1f, 2_000L))

        assertEquals(listOf(-2, -1, 0, 1, 2), plan.candidates.map { it.offsetCells })
        assertTrue(plan.candidates.filter { it.offsetCells in -1..1 }.none { it.hardSafe })
        assertEquals(-2, plan.selected?.offsetCells)
    }

    @Test
    fun labelledOfflineScenariosExerciseGeometryMotionCorridorAndFailClosedBoundaries() {
        val results = UstrfOfflineSafetyScenarioRunner().runAll()

        assertEquals(UstrfOfflineSafetyScenarioId.entries.toList(), results.map { it.id })
        results.forEach { result ->
            assertEquals(result.id.name, result.expectation.action, result.record.decision.action)
            assertTrue(
                "${result.id} missing ${result.expectation.requiredReasons}",
                result.record.decision.reasons.containsAll(result.expectation.requiredReasons)
            )
            assertEquals(result.id.name, result.expectation.expectedSelectedOffset, result.record.decision.experimentalCorridorOffsetCells)
            assertTrue(result.id.name, UstrfSafetyReason.SHADOW_ONLY in result.record.decision.reasons)
        }
    }

    @Test
    fun offlineScenarioReportsAreDeterministicAndSyntheticOnly() {
        val first = UstrfOfflineSafetyScenarioRunner().runAll()
        val second = UstrfOfflineSafetyScenarioRunner().runAll()

        assertEquals(first, second)
        assertEquals(first.map { it.traceDigest }, second.map { it.traceDigest })
        assertTrue(first.all { it.record.decision.validUntilNs > 0L })
        assertTrue(first.filter { it.record.field != null }.all { result ->
            result.record.field!!.cells.values.all { "synthetic-fixture-v1" in it.sources || "synthetic-motion" in it.sources }
        })
    }
}
