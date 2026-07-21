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

    @Test
    fun futureOrStaleRiskFieldFailsClosedEvenWhenRouteRemainsValid() {
        val field = UstrfRiskField(frame, mapOf(left to UstrfRiskCell(occupancy = 1f, uncertainty = .1f)))
        val strict = UstrfRouteConditionedRiskInteractor(maximumRiskFieldAgeNs = 50L)

        val future = unavailable(strict.interact(field, route(left), 999L, runtime = true))
        assertEquals(UstrfRouteConditionedRiskFailure.RISK_FIELD_FROM_FUTURE, future.failure)

        val stale = unavailable(strict.interact(field, route(left), 1_051L, runtime = true))
        assertEquals(UstrfRouteConditionedRiskFailure.RISK_FIELD_STALE, stale.failure)
        assertEquals("risk_field_unknown_or_invalid", stale.abstainReason)
    }

    @Test
    fun evidenceValidityCannotOutliveTheRiskFieldFreshnessWindow() {
        val field = UstrfRiskField(frame, mapOf(left to UstrfRiskCell(occupancy = .8f, uncertainty = .1f)))
        val strict = UstrfRouteConditionedRiskInteractor(maximumRiskFieldAgeNs = 200L)

        val evidence = available(strict.interact(field, route(left, validUntilNs = 2_000L), 1_100L, runtime = true))

        assertEquals(1_200L, evidence.validUntilNs)
    }

    @Test
    fun continuousRouteFieldEntersTheSafetySessionAndSelectsTheAlignedSafeEnvelope() {
        val observations = (-1..1).flatMap { lateral ->
            (1..4).map { forward ->
                UstrfRiskObservation(
                    UstrfGridCoordinate(lateral, forward),
                    0f, 1f, 0f, 0f, null, 0f, "metric-fixture", 2_000L
                )
            }
        }
        val routeField = UstrfRouteFieldReceipt(
            sourceFrame = frame,
            routeIntentId = "route-field-session",
            providerId = "navigation",
            providerType = UstrfRouteFieldProviderType.NAVIGATION,
            coordinateFrame = frame.coordinateFrame,
            issuedAtNs = 900L,
            validUntilNs = 2_000L,
            confidence = .95f,
            routeValid = true,
            inferredByRiskModel = false,
            derivedFromFutureFrames = false,
            weights = (1..4).associate { UstrfGridCoordinate(-1, it) to 1f }
        )
        val record = UstrfSafetySession().evaluate(
            UstrfSessionInput(
                frame = frame,
                health = UstrfHealth(
                    UstrfPoseState.TRACKING,
                    UstrfEvidenceState.VALID,
                    UstrfEvidenceState.VALID,
                    UstrfEvidenceState.VALID
                ),
                perception = UstrfPerceptionAssembly.Available(
                    UstrfPerceptionPacket(frame, 1_000L, 2_000L, observations)
                ),
                route = null,
                decisionAtNs = 1_100L,
                routeField = routeField
            )
        )

        assertEquals(-1, record.decision.experimentalCorridorOffsetCells)
        assertEquals("route-field-session", record.routeIntrusionEvidence?.routeIntentId)
        assertTrue(UstrfSafetyReason.SHADOW_ONLY in record.decision.reasons)
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
