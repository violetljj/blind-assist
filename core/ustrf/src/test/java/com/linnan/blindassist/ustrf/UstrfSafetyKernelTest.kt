package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfSafetyKernelTest {
    @Test
    fun identicalSafeReplayIsDeterministicAndStillShadowOnly() {
        val ticks = listOf(safeTick(1L, 1_000L), safeTick(2L, 2_000L))
        val first = UstrfReplayRunner().run(ticks)
        val second = UstrfReplayRunner().run(ticks)

        assertEquals(first, second)
        assertTrue(first.all { it.decision.action == UstrfSafetyAction.SLOW_DOWN })
        assertTrue(first.all { UstrfSafetyReason.SHADOW_ONLY in it.decision.reasons })
        assertTrue(first.all { it.decision.experimentalCorridorOffsetCells == 0 })
        assertEquals(UstrfTraceDigest.sha256(first), UstrfTraceDigest.sha256(second))
    }

    @Test
    fun futureTickMutationCannotChangeEarlierTracePrefix() {
        val baseline = UstrfReplayRunner().run(listOf(safeTick(1L, 1_000L), safeTick(2L, 2_000L)))
        val futureObstacle = UstrfReplayRunner().run(listOf(
            safeTick(1L, 1_000L),
            safeTick(2L, 2_000L, occupied = UstrfGridCoordinate(0, 2))
        ))

        assertEquals(UstrfTraceDigest.canonicalText(baseline.take(1)), UstrfTraceDigest.canonicalText(futureObstacle.take(1)))
        assertTrue(UstrfTraceDigest.sha256(baseline) != UstrfTraceDigest.sha256(futureObstacle))
    }

    @Test
    fun replayRejectsReorderedFrames() {
        try {
            UstrfReplayRunner().run(listOf(safeTick(2L, 2_000L), safeTick(1L, 1_000L)))
            throw AssertionError("expected rejection")
        } catch (expected: IllegalArgumentException) {
            assertTrue(expected.message.orEmpty().contains("strictly increasing"))
        }
    }

    @Test
    fun stalePerceptionAndLostPoseStopEvenWithClearObservations() {
        val oldFrame = UstrfFrameStamp(0L, 900L, "user-local-v1")
        val oldObservations = (-1..1).flatMap { lateral -> (1..4).map { forward ->
            UstrfRiskObservation(UstrfGridCoordinate(lateral, forward), 0f, 1f, 0f, 0f, null, 0f, "synthetic", 999L)
        } }
        val stale = UstrfTick(
            UstrfFrameStamp(1L, 1_000L, "user-local-v1"),
            UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
            UstrfPerceptionPacket(oldFrame, 900L, 999L, oldObservations),
            UstrfRouteIntent("user-local-v1", 0, 1f, 2_000L)
        )
        val staleField = UstrfRiskFieldBuilder().update(stale.perception)
        val staleDecision = UstrfSafetySupervisor().decide(stale, staleField, null)
        assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, staleDecision.action)
        assertTrue(UstrfSafetyReason.PERCEPTION_STALE in staleDecision.reasons)

        val lost = safeTick(1L, 1_000L, pose = UstrfPoseState.LOST)
        val lostField = UstrfRiskFieldBuilder().update(lost.perception)
        val lostDecision = UstrfSafetySupervisor().decide(lost, lostField, UstrfCorridorPlanner().plan(lostField, requireNotNull(lost.route)))
        assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, lostDecision.action)
        assertTrue(UstrfSafetyReason.POSE_NOT_TRACKING in lostDecision.reasons)
    }

    @Test
    fun captureGeometryOrMotionLossStopsInsteadOfDowngradingToAWarning() {
        val baseline = safeTick(1L, 1_000L)
        val field = UstrfRiskFieldBuilder().update(baseline.perception)
        val plan = UstrfCorridorPlanner().plan(field, requireNotNull(baseline.route))
        listOf(
            baseline.health.copy(capture = UstrfEvidenceState.MISSING) to UstrfSafetyReason.CAPTURE_UNAVAILABLE,
            baseline.health.copy(geometry = UstrfEvidenceState.MISSING) to UstrfSafetyReason.GEOMETRY_UNAVAILABLE,
            baseline.health.copy(motion = UstrfEvidenceState.MISSING) to UstrfSafetyReason.MOTION_UNAVAILABLE
        ).forEach { (health, reason) ->
            val decision = UstrfSafetySupervisor().decide(baseline.copy(health = health), field, plan)
            assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, decision.action)
            assertTrue(reason in decision.reasons)
        }
    }

    @Test
    fun invalidRouteScansWithoutDirection() {
        val tick = safeTick(1L, 1_000L, route = UstrfRouteIntent("other-frame", 0, 1f, 2_000L))
        val field = UstrfRiskFieldBuilder().update(tick.perception)
        val decision = UstrfSafetySupervisor().decide(tick, field, null)
        assertEquals(UstrfSafetyAction.SCAN, decision.action)
        assertTrue(UstrfSafetyReason.ROUTE_INVALID in decision.reasons)
        assertNull(decision.experimentalCorridorOffsetCells)
    }

    @Test
    fun slowLoopSemanticHintsCannotBypassTheFastLoopSafetyDecision() {
        val baseline = safeTick(1L, 1_000L)
        val semantic = UstrfSemanticHint(
            baseline.frame,
            1_000L,
            2_000L,
            1f,
            "open_door_ahead"
        )
        val field = UstrfRiskFieldBuilder().update(baseline.perception)
        val planner = UstrfCorridorPlanner()
        val expected = UstrfSafetySupervisor().decide(baseline, field, planner.plan(field, requireNotNull(baseline.route)))
        val withSemantic = UstrfSafetySupervisor().decide(
            baseline.copy(semanticHints = listOf(semantic)),
            field,
            planner.plan(field, requireNotNull(baseline.route))
        )

        assertEquals(expected, withSemantic)
    }

    @Test
    fun freshButWrongSourceFrameStillStops() {
        val oldFrame = UstrfFrameStamp(0L, 900L, "user-local-v1")
        val tick = safeTick(1L, 1_000L).copy(
            perception = UstrfPerceptionPacket(oldFrame, 1_000L, 2_000L, emptyList())
        )
        val decision = UstrfSafetySupervisor().decide(tick, UstrfRiskField(oldFrame, emptyMap()), null)

        assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, decision.action)
        assertTrue(UstrfSafetyReason.SOURCE_FRAME_MISMATCH in decision.reasons)
    }

    @Test
    fun centerObstacleRejectsCenterCandidateAndSelectsSideOnlyInTrace() {
        val tick = safeTick(1L, 1_000L, occupied = UstrfGridCoordinate(0, 2))
        val field = UstrfRiskFieldBuilder().update(tick.perception)
        val plan = UstrfCorridorPlanner().plan(field, requireNotNull(tick.route))
        assertTrue(!plan.candidates.first { it.offsetCells == 0 }.hardSafe)
        assertTrue(plan.selected?.offsetCells in setOf(-1, 1))
    }

    @Test
    fun missingCentralCellsFailClosed() {
        val frame = UstrfFrameStamp(1L, 1_000L, "user-local-v1")
        val tick = UstrfTick(
            frame,
            UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
            UstrfPerceptionPacket(frame, 1_000L, 2_000L, emptyList()),
            UstrfRouteIntent("user-local-v1", 0, 1f, 2_000L)
        )
        val field = UstrfRiskFieldBuilder().update(tick.perception)
        val decision = UstrfSafetySupervisor().decide(tick, field, UstrfCorridorPlanner().plan(field, requireNotNull(tick.route)))
        assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, decision.action)
        assertTrue(UstrfSafetyReason.CENTRAL_CORRIDOR_UNKNOWN in decision.reasons)
    }

    @Test
    fun dynamicTtcHardRiskRejectsEveryCorridorInsteadOfAveragingItAway() {
        val frame = UstrfFrameStamp(1L, 1_000L, "user-local-v1")
        val observations = (-1..1).flatMap { lateral -> (1..4).map { forward ->
            UstrfRiskObservation(UstrfGridCoordinate(lateral, forward), 0f, 1f, 0f, 0f, 800L, 0f, "motion", 2_000L)
        } }
        val field = UstrfRiskFieldBuilder().update(UstrfPerceptionPacket(frame, 1_000L, 2_000L, observations))

        assertNull(UstrfCorridorPlanner().plan(field, UstrfRouteIntent("user-local-v1", 0, 1f, 2_000L)).selected)
    }

    @Test
    fun staticRiskAgesDeterministicallyWhenFreshObservationsDoNotRefreshIt() {
        val builder = UstrfRiskFieldBuilder(UstrfRiskFieldConfig(staticLifetimeNs = 100L, dynamicLifetimeNs = 50L))
        val first = UstrfFrameStamp(1L, 100L, "user-local-v1")
        builder.update(UstrfPerceptionPacket(
            first,
            100L,
            200L,
            listOf(UstrfRiskObservation(UstrfGridCoordinate(0, 1), .8f, 0f, 0f, 0f, null, 0f, "geometry", 200L))
        ))
        val second = UstrfFrameStamp(2L, 150L, "user-local-v1")
        val aged = builder.update(UstrfPerceptionPacket(second, 150L, 250L, emptyList())).cellAt(UstrfGridCoordinate(0, 1))

        assertEquals(.4f, aged.occupancy, .0001f)
        assertEquals(50L, aged.ageNs)
    }

    @Test
    fun poseBufferInterpolatesAtFrameTimeAndRejectsInvalidReceipts() {
        val buffer = UstrfPoseBuffer(maxBracketGapNs = 200L)
        buffer.append(pose(100L, UstrfVector3(0f, 0f, 0f)))
        buffer.append(pose(200L, UstrfVector3(2f, 0f, 0f)))

        val available = buffer.interpolateAt(UstrfFrameStamp(1L, 150L, "camera-v1")) as UstrfPoseLookup.Available
        assertEquals(1f, available.pose.worldCameraTranslationM.x, .0001f)
        assertEquals(150L, available.pose.timestampNs)

        val missing = UstrfPoseBuffer().apply { append(pose(100L, UstrfVector3(0f, 0f, 0f))) }
            .interpolateAt(UstrfFrameStamp(2L, 150L, "camera-v1"))
        assertEquals(UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.NOT_BRACKETED), missing)
    }

    @Test
    fun poseBufferFailsClosedForLowConfidenceAndCameraFrameMismatch() {
        val lowConfidence = UstrfPoseBuffer().apply {
            append(pose(100L, UstrfVector3(0f, 0f, 0f), confidence = .5f))
            append(pose(200L, UstrfVector3(1f, 0f, 0f), confidence = .9f))
        }.interpolateAt(UstrfFrameStamp(1L, 150L, "camera-v1"))
        assertEquals(UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.LOW_CONFIDENCE), lowConfidence)

        val mismatch = UstrfPoseBuffer().apply {
            append(pose(100L, UstrfVector3(0f, 0f, 0f)))
            append(pose(200L, UstrfVector3(1f, 0f, 0f)))
        }.interpolateAt(UstrfFrameStamp(1L, 150L, "other-camera"))
        assertEquals(UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.CAMERA_FRAME_MISMATCH), mismatch)
    }

    @Test
    fun geometryProjectorMapsMetricDropAndHeadEvidenceIntoDifferentRiskChannels() {
        val frame = UstrfFrameStamp(1L, 1_000L, "camera-v1")
        val packet = UstrfGeometryPacket(
            frame,
            1_000L,
            2_000L,
            UstrfDepthScale.METRIC,
            listOf(
                UstrfMetricGeometryEvidence(.5f, 0f, UstrfHeightBand.GROUND, UstrfGeometryKind.DROP, .8f, "depth", 2_000L),
                UstrfMetricGeometryEvidence(1f, .5f, UstrfHeightBand.HEAD, UstrfGeometryKind.HEAD_OBSTACLE, .9f, "depth", 2_000L)
            )
        )

        val projected = UstrfGeometryProjector(UstrfGridSpec.DOCUMENT_FIVE_METER)
            .project(packet, 1_000L) as UstrfGeometryProjection.Available
        assertEquals(.8f, projected.observations.first { it.coordinate == UstrfGridCoordinate(0, 1) }.dropRisk, .0001f)
        assertEquals(.9f, projected.observations.first { it.coordinate == UstrfGridCoordinate(1, 2) }.headRisk, .0001f)
    }

    @Test
    fun geometryProjectorRejectsRelativeDepthAndStaleEvidence() {
        val frame = UstrfFrameStamp(1L, 1_000L, "camera-v1")
        val relative = UstrfGeometryPacket(frame, 1_000L, 2_000L, UstrfDepthScale.RELATIVE, emptyList())
        assertEquals(UstrfGeometryProjection.Unavailable(UstrfGeometryProjectionFailure.SCALE_NOT_METRIC), UstrfGeometryProjector().project(relative, 1_000L))

        val stale = UstrfGeometryPacket(frame, 1_000L, 1_000L, UstrfDepthScale.METRIC, emptyList())
        assertEquals(UstrfGeometryProjection.Unavailable(UstrfGeometryProjectionFailure.STALE), UstrfGeometryProjector().project(stale, 1_001L))
    }

    @Test
    fun ttcEstimatorFindsClosestApproachWithoutSemanticClass() {
        val estimate = UstrfTtcEstimator().estimate(
            UstrfRelativeMotionEvidence(
                UstrfVector2(2f, 0f),
                UstrfVector2(-1f, 0f),
                1_000_000_000L,
                3_000_000_000L,
                "motion"
            ),
            1_500_000_000L
        )

        assertEquals(1_500L, requireNotNull(estimate).timeToClosestApproachMs)
        assertEquals(0f, estimate.closestDistanceMeters, .0001f)
    }

    @Test
    fun ttcEstimatorRejectsStaleOrStaticMotion() {
        val stale = UstrfRelativeMotionEvidence(UstrfVector2(1f, 0f), UstrfVector2(-1f, 0f), 1_000L, 1_000L, "motion")
        assertNull(UstrfTtcEstimator().estimate(stale, 1_001L))

        val static = UstrfRelativeMotionEvidence(UstrfVector2(1f, 0f), UstrfVector2(0f, 0f), 1_000L, 2_000L, "motion")
        assertNull(UstrfTtcEstimator().estimate(static, 1_500L))
    }

    @Test
    fun perceptionAssemblerAcceptsOnlyCoTimedMetricReceiptsAndUsesTheShortestTtl() {
        val frame = UstrfFrameStamp(1L, 1_000_000_000L, "camera-v1")
        val geometry = UstrfGeometryPacket(
            frame, 1_000_000_000L, 3_000_000_000L, UstrfDepthScale.METRIC,
            listOf(UstrfMetricGeometryEvidence(.5f, 0f, UstrfHeightBand.GROUND, UstrfGeometryKind.TRAVERSABLE, 1f, "depth", 3_000_000_000L))
        )
        val motion = UstrfMotionGridEvidence(
            frame,
            UstrfGridCoordinate(0, 2),
            UstrfRelativeMotionEvidence(UstrfVector2(2f, 0f), UstrfVector2(-1f, 0f), 1_000_000_000L, 2_500_000_000L, "flow")
        )

        val assembled = UstrfPerceptionAssembler().assemble(frame, geometry, listOf(motion), 1_500_000_000L) as UstrfPerceptionAssembly.Available
        assertEquals(2_500_000_000L, assembled.packet.validUntilNs)
        assertEquals(1_500L, assembled.packet.observations.single { it.source == "flow" }.dynamicTtcMs)
    }

    @Test
    fun perceptionAssemblerRejectsWrongFrameAndUnavailableMotionInsteadOfPublishingPartialData() {
        val frame = UstrfFrameStamp(1L, 1_000L, "camera-v1")
        val geometry = UstrfGeometryPacket(frame, 1_000L, 3_000L, UstrfDepthScale.METRIC, emptyList())
        val wrongFrame = UstrfFrameStamp(2L, 1_000L, "camera-v1")
        val mismatch = UstrfPerceptionAssembler().assemble(wrongFrame, geometry, emptyList(), 1_000L)
        assertEquals(
            UstrfPerceptionAssembly.Unavailable(setOf(UstrfPerceptionAssemblyFailure.GEOMETRY_SOURCE_FRAME_MISMATCH)),
            mismatch
        )

        val staleMotion = UstrfMotionGridEvidence(
            frame,
            UstrfGridCoordinate(0, 1),
            UstrfRelativeMotionEvidence(UstrfVector2(1f, 0f), UstrfVector2(-1f, 0f), 1_000L, 1_000L, "flow")
        )
        val unavailable = UstrfPerceptionAssembler().assemble(frame, geometry, listOf(staleMotion), 1_001L)
        assertEquals(
            UstrfPerceptionAssembly.Unavailable(setOf(UstrfPerceptionAssemblyFailure.MOTION_UNAVAILABLE)),
            unavailable
        )
    }

    @Test
    fun safetySessionMapsAssemblyFailuresToAStopWithoutPublishingPartialFieldEvidence() {
        val frame = UstrfFrameStamp(1L, 1_000L, "camera-v1")
        val unavailable = UstrfPerceptionAssembly.Unavailable(
            setOf(UstrfPerceptionAssemblyFailure.MOTION_SOURCE_FRAME_MISMATCH)
        )
        val result = UstrfSafetySession().evaluate(
            UstrfSessionInput(
                frame,
                UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
                unavailable,
                UstrfRouteIntent("camera-v1", 0, 1f, 2_000L)
            )
        )

        assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, result.decision.action)
        assertTrue(UstrfSafetyReason.SOURCE_FRAME_MISMATCH in result.decision.reasons)
        assertTrue(UstrfSafetyReason.PERCEPTION_ASSEMBLY_UNAVAILABLE in result.decision.reasons)
        assertNull(result.field)
        assertEquals(unavailable.failures, result.assemblyFailures)
    }

    @Test
    fun safetySessionUsesAvailableAssemblyAndRejectsSessionTimeRollback() {
        val session = UstrfSafetySession()
        val frame = UstrfFrameStamp(1L, 1_000L, "camera-v1")
        val packet = UstrfPerceptionPacket(frame, 1_000L, 2_000L, clearObservations("camera", 2_000L))
        val input = UstrfSessionInput(
            frame,
            UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
            UstrfPerceptionAssembly.Available(packet),
            UstrfRouteIntent("camera-v1", 0, 1f, 2_000L)
        )

        assertEquals(UstrfSafetyAction.SLOW_DOWN, session.evaluate(input).decision.action)
        try {
            session.evaluate(input.copy(frame = UstrfFrameStamp(0L, 999L, "camera-v1"), perception = UstrfPerceptionAssembly.Available(packet)))
            throw AssertionError("expected rejection")
        } catch (expected: IllegalArgumentException) {
            assertTrue(expected.message.orEmpty().contains("strictly increasing"))
        }
    }

    @Test
    fun safetySessionStopsForAQueuedPacketThatExpiredAfterCaptureTime() {
        val frame = UstrfFrameStamp(1L, 1_000L, "camera-v1")
        val stalePacket = UstrfPerceptionPacket(frame, 1_000L, 1_100L, clearObservations("camera", 1_100L))
        val result = UstrfSafetySession().evaluate(
            UstrfSessionInput(
                frame,
                UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
                UstrfPerceptionAssembly.Available(stalePacket),
                UstrfRouteIntent("camera-v1", 0, 1f, 2_000L),
                decisionAtNs = 1_101L
            )
        )

        assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, result.decision.action)
        assertTrue(UstrfSafetyReason.PERCEPTION_STALE in result.decision.reasons)
        assertEquals(setOf(UstrfPerceptionAssemblyFailure.PERCEPTION_TIMING_INVALID), result.assemblyFailures)
    }

    @Test
    fun sessionTraceDigestIncludesAssemblyFailuresAndIsDeterministic() {
        fun records() = UstrfSafetySession().let { session ->
            val firstFrame = UstrfFrameStamp(1L, 1_000L, "camera-v1")
            val first = session.evaluate(
                UstrfSessionInput(
                    firstFrame,
                    UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
                    UstrfPerceptionAssembly.Available(UstrfPerceptionPacket(firstFrame, 1_000L, 2_000L, clearObservations("camera", 2_000L))),
                    UstrfRouteIntent("camera-v1", 0, 1f, 2_000L)
                )
            )
            val second = session.evaluate(
                UstrfSessionInput(
                    UstrfFrameStamp(2L, 1_100L, "camera-v1"),
                    UstrfHealth(UstrfPoseState.TRACKING, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
                    UstrfPerceptionAssembly.Unavailable(setOf(UstrfPerceptionAssemblyFailure.MOTION_UNAVAILABLE)),
                    UstrfRouteIntent("camera-v1", 0, 1f, 2_000L)
                )
            )
            listOf(first, second)
        }

        val first = records()
        val second = records()
        assertEquals(UstrfSessionTraceDigest.canonicalText(first), UstrfSessionTraceDigest.canonicalText(second))
        assertEquals(UstrfSessionTraceDigest.sha256(first), UstrfSessionTraceDigest.sha256(second))
        assertTrue(UstrfSessionTraceDigest.canonicalText(first).contains("MOTION_UNAVAILABLE"))
    }

    @Test
    fun captureReceiptValidatorRejectsLateAndRollbackCameraReceipts() {
        val validator = UstrfCaptureReceiptValidator(maximumCaptureAgeNs = 100L)
        val first = captureReceipt(1L, 1_000L, 1_000L)
        assertEquals(UstrfCaptureReceiptValidation.Valid, validator.validate(first, 1_050L))
        assertEquals(
            UstrfCaptureReceiptValidation.Unavailable(UstrfCaptureReceiptFailure.HARDWARE_TIMESTAMP_ROLLBACK),
            validator.validate(captureReceipt(2L, 1_060L, 999L), 1_070L)
        )
        assertEquals(
            UstrfCaptureReceiptValidation.Unavailable(UstrfCaptureReceiptFailure.CAPTURE_AGE_EXCEEDED),
            validator.validate(captureReceipt(2L, 1_060L, 1_060L), 1_161L)
        )
    }

    @Test
    fun captureReceiptValidatorPinsClockAndCalibrationAcrossASequence() {
        val validator = UstrfCaptureReceiptValidator()
        assertEquals(UstrfCaptureReceiptValidation.Valid, validator.validate(captureReceipt(1L, 1_000L, 1_000L), 1_001L))
        val changedCalibration = captureReceipt(2L, 1_010L, 1_010L).copy(calibrationVersion = "intrinsics-v2")
        assertEquals(
            UstrfCaptureReceiptValidation.Unavailable(UstrfCaptureReceiptFailure.CALIBRATION_CHANGED),
            validator.validate(changedCalibration, 1_011L)
        )
    }

    @Test
    fun routeReceiptResolverBindsSlowLoopIntentToTheFastLoopFrameAndDecisionTime() {
        val frame = UstrfFrameStamp(1L, 1_000L, "camera-v1")
        val receipt = UstrfRouteReceipt(frame, 1_001L, 2_000L, "camera-v1", 1, .9f, "local-route")
        val resolver = UstrfRouteReceiptResolver()
        assertEquals(
            UstrfRouteReceiptResolution.Available(UstrfRouteIntent("camera-v1", 1, .9f, 2_000L)),
            resolver.resolve(receipt, frame, 1_100L)
        )
        assertEquals(
            UstrfRouteReceiptResolution.Unavailable(UstrfRouteReceiptFailure.STALE),
            resolver.resolve(receipt, frame, 2_001L)
        )
    }

    @Test
    fun routeReceiptResolverRejectsCrossFrameAndOverwideDirectionalProposals() {
        val frame = UstrfFrameStamp(1L, 1_000L, "camera-v1")
        val resolver = UstrfRouteReceiptResolver()
        val wrongFrame = UstrfRouteReceipt(UstrfFrameStamp(2L, 1_000L, "camera-v1"), 1_001L, 2_000L, "camera-v1", 0, 1f, "route")
        assertEquals(
            UstrfRouteReceiptResolution.Unavailable(UstrfRouteReceiptFailure.SOURCE_FRAME_MISMATCH),
            resolver.resolve(wrongFrame, frame, 1_100L)
        )
        val overwide = UstrfRouteReceipt(frame, 1_001L, 2_000L, "camera-v1", 3, 1f, "route")
        assertEquals(
            UstrfRouteReceiptResolution.Unavailable(UstrfRouteReceiptFailure.OFFSET_OUT_OF_RANGE),
            resolver.resolve(overwide, frame, 1_100L)
        )
    }

    @Test
    fun uncertaintyFusionIsExplicitAndMonotonicAcrossIndependentSources() {
        assertEquals(0f, UstrfUncertaintyFusion.fuse(UstrfUncertaintyEvidence(0f, 0f, 0f, 0f)), .0001f)
        assertEquals(1f, UstrfUncertaintyFusion.fuse(UstrfUncertaintyEvidence(0f, 1f, 0f, 0f)), .0001f)
        assertEquals(.75f, UstrfUncertaintyFusion.fuse(UstrfUncertaintyEvidence(.5f, .5f, 0f, 0f)), .0001f)
    }

    private fun safeTick(
        id: Long,
        nowNs: Long,
        perceptionTtlNs: Long = nowNs + 1_000L,
        pose: UstrfPoseState = UstrfPoseState.TRACKING,
        route: UstrfRouteIntent? = UstrfRouteIntent("user-local-v1", 0, 1f, nowNs + 1_000L),
        occupied: UstrfGridCoordinate? = null
    ): UstrfTick {
        val frame = UstrfFrameStamp(id, nowNs, "user-local-v1")
        val observations = (-1..1).flatMap { lateral -> (1..4).map { forward ->
            val point = UstrfGridCoordinate(lateral, forward)
            UstrfRiskObservation(point, if (point == occupied) 1f else 0f, 1f, 0f, 0f, null, 0f, "synthetic", perceptionTtlNs)
        } }
        return UstrfTick(
            frame,
            UstrfHealth(pose, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID, UstrfEvidenceState.VALID),
            UstrfPerceptionPacket(frame, nowNs, perceptionTtlNs, observations),
            route
        )
    }

    private fun clearObservations(source: String, ttlNs: Long): List<UstrfRiskObservation> =
        (-1..1).flatMap { lateral -> (1..4).map { forward ->
            UstrfRiskObservation(UstrfGridCoordinate(lateral, forward), 0f, 1f, 0f, 0f, null, 0f, source, ttlNs)
        } }

    private fun captureReceipt(frameId: Long, capturedAtNs: Long, hardwareTimestampNs: Long) = UstrfCaptureReceipt(
        UstrfFrameStamp(frameId, capturedAtNs, "camera-v1"),
        hardwareTimestampNs,
        capturedAtNs,
        "camera-clock-v1",
        "intrinsics-v1"
    )

    private fun pose(timestampNs: Long, position: UstrfVector3, confidence: Float = 1f) = UstrfPoseSample(
        timestampNs,
        "world-v1",
        "camera-v1",
        position,
        0f,
        UstrfVector3(0f, 0f, -1f),
        UstrfPoseState.TRACKING,
        confidence,
        timestampNs + 1_000L
    )
}
