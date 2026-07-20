package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfRouteConditionedRiskTest {
    private val frame = UstrfFrameStamp(7L, 1_000L, "body-local-grid-v1")
    private val left = UstrfGridCoordinate(-1, 2)
    private val right = UstrfGridCoordinate(1, 2)
    private val interactor = UstrfRouteConditionedRiskInteractor()

    @Test
    fun identicalObjectAgnosticFieldChangesRiskOnlyWhenSelectedRouteChanges() {
        val field = UstrfRiskField(
            frame,
            mapOf(
                left to UstrfRiskCell(occupancy = .9f, uncertainty = .1f, sources = setOf("dense-object-agnostic")),
                right to UstrfRiskCell(occupancy = 0f, uncertainty = .1f, sources = setOf("dense-object-agnostic"))
            )
        )

        val leftEvidence = available(interactor.interact(field, route(left), 1_100L, runtime = true))
        val rightEvidence = available(interactor.interact(field, route(right), 1_100L, runtime = true))

        assertEquals(.9f, leftEvidence.routeIntrusionScore, .0001f)
        assertEquals(0f, rightEvidence.routeIntrusionScore, .0001f)
        assertEquals(0f, leftEvidence.routeUnknownFraction, .0001f)
        assertEquals(setOf("dense-object-agnostic"), leftEvidence.riskSources)
    }

    @Test
    fun missingRouteCellsRemainExplicitlyUnknownInsteadOfBecomingClear() {
        val field = UstrfRiskField(frame, emptyMap())
        val evidence = available(interactor.interact(field, route(left), 1_100L, runtime = true))
        assertEquals(0f, evidence.routeIntrusionScore, .0001f)
        assertEquals(1f, evidence.routeUnknownFraction, .0001f)
    }

    @Test
    fun futureRiskModelOrOfflineRuntimeRouteFailsClosedWithStableReason() {
        val field = UstrfRiskField(frame, mapOf(left to UstrfRiskCell(uncertainty = .1f)))
        val inferred = unavailable(interactor.interact(field, route(left, inferred = true), 1_100L, runtime = true))
        assertEquals(UstrfRouteConditionedRiskFailure.ROUTE_INFERRED_BY_RISK_MODEL, inferred.failure)
        assertEquals("route_unknown_or_invalid", inferred.abstainReason)

        val future = unavailable(interactor.interact(field, route(left, futureDerived = true), 1_100L, runtime = false))
        assertEquals(UstrfRouteConditionedRiskFailure.FUTURE_DERIVED_ROUTE_FORBIDDEN, future.failure)

        val offlineRuntime = unavailable(interactor.interact(
            field,
            route(left, providerType = UstrfRouteFieldProviderType.OFFLINE_HUMAN_TEACHER),
            1_100L,
            runtime = true
        ))
        assertEquals(UstrfRouteConditionedRiskFailure.PROVIDER_NOT_ALLOWED_AT_RUNTIME, offlineRuntime.failure)
    }

    @Test
    fun staleOrInvalidRouteNeverFallsBackToCenterCorridor() {
        val field = UstrfRiskField(frame, mapOf(left to UstrfRiskCell(occupancy = 1f, uncertainty = .1f)))
        val stale = unavailable(interactor.interact(field, route(left, validUntilNs = 1_050L), 1_100L, runtime = true))
        assertEquals(UstrfRouteConditionedRiskFailure.ROUTE_STALE, stale.failure)
        val invalid = unavailable(interactor.interact(field, route(left, routeValid = false), 1_100L, runtime = true))
        assertEquals(UstrfRouteConditionedRiskFailure.ROUTE_INVALID, invalid.failure)
    }

    private fun route(
        cell: UstrfGridCoordinate,
        providerType: UstrfRouteFieldProviderType = UstrfRouteFieldProviderType.NAVIGATION,
        inferred: Boolean = false,
        futureDerived: Boolean = false,
        routeValid: Boolean = true,
        validUntilNs: Long = 2_000L
    ) = UstrfRouteFieldReceipt(
        sourceFrame = frame,
        routeIntentId = "route-1",
        providerId = "provider-1",
        providerType = providerType,
        coordinateFrame = frame.coordinateFrame,
        issuedAtNs = 900L,
        validUntilNs = validUntilNs,
        confidence = .95f,
        routeValid = routeValid,
        inferredByRiskModel = inferred,
        derivedFromFutureFrames = futureDerived,
        weights = mapOf(cell to 1f)
    )

    private fun available(value: UstrfRouteConditionedRiskResolution): UstrfRouteIntrusionEvidence {
        assertTrue(value is UstrfRouteConditionedRiskResolution.Available)
        return (value as UstrfRouteConditionedRiskResolution.Available).evidence
    }

    private fun unavailable(value: UstrfRouteConditionedRiskResolution): UstrfRouteConditionedRiskResolution.Unavailable {
        assertTrue(value is UstrfRouteConditionedRiskResolution.Unavailable)
        return value as UstrfRouteConditionedRiskResolution.Unavailable
    }
}
