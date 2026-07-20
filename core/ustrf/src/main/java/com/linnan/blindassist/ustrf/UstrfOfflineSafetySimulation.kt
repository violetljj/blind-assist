package com.linnan.blindassist.ustrf

/**
 * Deterministic, geometry-level safety fixtures. They model labelled local evidence rather than
 * camera pixels, and therefore prove safety-kernel behaviour only -- never perception accuracy.
 */
enum class UstrfOfflineSafetyScenarioId {
    CLEAR_CORRIDOR,
    CENTER_OCCUPANCY,
    FULL_WIDTH_DROP,
    FULL_WIDTH_HEAD_OBSTACLE,
    DYNAMIC_CROSSING,
    CENTRAL_UNKNOWN,
    STALE_GEOMETRY,
    POSE_LOST
}

data class UstrfOfflineSafetyExpectation(
    val action: UstrfSafetyAction,
    val requiredReasons: Set<UstrfSafetyReason> = emptySet(),
    val expectedSelectedOffset: Int? = null
)

data class UstrfOfflineSafetyScenarioResult(
    val id: UstrfOfflineSafetyScenarioId,
    val record: UstrfSessionRecord,
    val expectation: UstrfOfflineSafetyExpectation,
    val traceDigest: String
)

/**
 * A production-isolated G0 simulator for the P0 geometry/motion/planning chain. It contains no
 * images, model predictions or physical calibration. All fixtures are explicitly synthetic.
 */
class UstrfOfflineSafetyScenarioRunner(
    private val coordinateFrame: String = "synthetic-local-v1",
    private val frameTimeNs: Long = 1_000_000_000L,
    private val ttlNs: Long = 2_000_000_000L
) {
    init {
        require(coordinateFrame.isNotBlank())
        require(frameTimeNs >= 0L && ttlNs > 0L)
    }

    fun runAll(): List<UstrfOfflineSafetyScenarioResult> = UstrfOfflineSafetyScenarioId.entries.map(::run)

    fun run(id: UstrfOfflineSafetyScenarioId): UstrfOfflineSafetyScenarioResult {
        val frame = UstrfFrameStamp(id.ordinal.toLong() + 1L, frameTimeNs, coordinateFrame)
        val scenario = fixture(id, frame)
        val session = UstrfSafetySession(
            fieldBuilder = UstrfRiskFieldBuilder(UstrfRiskFieldConfig(halfWidthCells = HALF_WIDTH_CELLS, horizonCells = HORIZON_CELLS)),
            planner = UstrfCorridorPlanner(
                horizonCells = HORIZON_CELLS,
                capsuleHalfWidthCells = 1,
                fixedCandidateOffsets = listOf(-2, -1, 0, 1, 2)
            )
        )
        val record = session.evaluate(scenario.input)
        return UstrfOfflineSafetyScenarioResult(
            id = id,
            record = record,
            expectation = scenario.expectation,
            traceDigest = UstrfSessionTraceDigest.sha256(listOf(record))
        )
    }

    private fun fixture(id: UstrfOfflineSafetyScenarioId, frame: UstrfFrameStamp): Fixture {
        val validUntilNs = frame.capturedAtNs + ttlNs
        val base = clearEvidence(validUntilNs)
        val expectation: UstrfOfflineSafetyExpectation
        val evidence: List<UstrfMetricGeometryEvidence>
        val motion: List<UstrfMotionGridEvidence>
        val health: UstrfHealth
        val assemblyNowNs: Long
        when (id) {
            UstrfOfflineSafetyScenarioId.CLEAR_CORRIDOR -> {
                evidence = base
                motion = emptyList()
                health = healthy()
                assemblyNowNs = frame.capturedAtNs
                expectation = UstrfOfflineSafetyExpectation(UstrfSafetyAction.SLOW_DOWN, expectedSelectedOffset = 0)
            }
            UstrfOfflineSafetyScenarioId.CENTER_OCCUPANCY -> {
                evidence = base + geometry(0, 2, UstrfHeightBand.LOWER_BODY, UstrfGeometryKind.OCCUPIED, .95f, validUntilNs)
                motion = emptyList()
                health = healthy()
                assemblyNowNs = frame.capturedAtNs
                expectation = UstrfOfflineSafetyExpectation(UstrfSafetyAction.SLOW_DOWN, expectedSelectedOffset = -2)
            }
            UstrfOfflineSafetyScenarioId.FULL_WIDTH_DROP -> {
                evidence = base + (-HALF_WIDTH_CELLS..HALF_WIDTH_CELLS).map { lateral ->
                    geometry(lateral, 2, UstrfHeightBand.GROUND, UstrfGeometryKind.DROP, .95f, validUntilNs)
                }
                motion = emptyList()
                health = healthy()
                assemblyNowNs = frame.capturedAtNs
                expectation = UstrfOfflineSafetyExpectation(UstrfSafetyAction.STOP_AND_REASSESS, setOf(UstrfSafetyReason.NO_SAFE_CORRIDOR))
            }
            UstrfOfflineSafetyScenarioId.FULL_WIDTH_HEAD_OBSTACLE -> {
                evidence = base + (-HALF_WIDTH_CELLS..HALF_WIDTH_CELLS).map { lateral ->
                    geometry(lateral, 2, UstrfHeightBand.HEAD, UstrfGeometryKind.HEAD_OBSTACLE, .95f, validUntilNs)
                }
                motion = emptyList()
                health = healthy()
                assemblyNowNs = frame.capturedAtNs
                expectation = UstrfOfflineSafetyExpectation(UstrfSafetyAction.STOP_AND_REASSESS, setOf(UstrfSafetyReason.NO_SAFE_CORRIDOR))
            }
            UstrfOfflineSafetyScenarioId.DYNAMIC_CROSSING -> {
                evidence = base
                motion = (-HALF_WIDTH_CELLS..HALF_WIDTH_CELLS).map { lateral ->
                    UstrfMotionGridEvidence(
                        frame,
                        UstrfGridCoordinate(lateral, 2),
                        UstrfRelativeMotionEvidence(UstrfVector2(1f, 0f), UstrfVector2(-1f, 0f), frame.capturedAtNs, validUntilNs, "synthetic-motion")
                    )
                }
                health = healthy()
                assemblyNowNs = frame.capturedAtNs
                expectation = UstrfOfflineSafetyExpectation(UstrfSafetyAction.STOP_AND_REASSESS, setOf(UstrfSafetyReason.NO_SAFE_CORRIDOR))
            }
            UstrfOfflineSafetyScenarioId.CENTRAL_UNKNOWN -> {
                evidence = base.filterNot { it.forwardMeters == 1f && it.lateralMeters == 0f }
                motion = emptyList()
                health = healthy()
                assemblyNowNs = frame.capturedAtNs
                // The candidate remains trace-only metadata. The supervisor's STOP is authoritative.
                expectation = UstrfOfflineSafetyExpectation(UstrfSafetyAction.STOP_AND_REASSESS, setOf(UstrfSafetyReason.CENTRAL_CORRIDOR_UNKNOWN), expectedSelectedOffset = -2)
            }
            UstrfOfflineSafetyScenarioId.STALE_GEOMETRY -> {
                evidence = base
                motion = emptyList()
                health = healthy()
                assemblyNowNs = validUntilNs + 1L
                expectation = UstrfOfflineSafetyExpectation(UstrfSafetyAction.STOP_AND_REASSESS, setOf(UstrfSafetyReason.GEOMETRY_UNAVAILABLE))
            }
            UstrfOfflineSafetyScenarioId.POSE_LOST -> {
                evidence = base
                motion = emptyList()
                health = healthy().copy(pose = UstrfPoseState.LOST)
                assemblyNowNs = frame.capturedAtNs
                expectation = UstrfOfflineSafetyExpectation(UstrfSafetyAction.STOP_AND_REASSESS, setOf(UstrfSafetyReason.POSE_NOT_TRACKING))
            }
        }
        val geometry = UstrfGeometryPacket(frame, frame.capturedAtNs, validUntilNs, UstrfDepthScale.METRIC, evidence)
        val assembled = UstrfPerceptionAssembler(
            geometryProjector = UstrfGeometryProjector(halfWidthCells = HALF_WIDTH_CELLS, horizonCells = HORIZON_CELLS)
        ).assemble(frame, geometry, motion, assemblyNowNs)
        return Fixture(
            UstrfSessionInput(
                frame = frame,
                health = health,
                perception = assembled,
                route = UstrfRouteIntent(coordinateFrame, 0, 1f, validUntilNs),
                decisionAtNs = assemblyNowNs
            ),
            expectation
        )
    }

    private fun clearEvidence(validUntilNs: Long): List<UstrfMetricGeometryEvidence> =
        (-HALF_WIDTH_CELLS..HALF_WIDTH_CELLS).flatMap { lateral ->
            (1..HORIZON_CELLS).map { forward ->
                geometry(lateral, forward, UstrfHeightBand.GROUND, UstrfGeometryKind.TRAVERSABLE, 1f, validUntilNs)
            }
        }

    private fun geometry(
        lateral: Int,
        forward: Int,
        heightBand: UstrfHeightBand,
        kind: UstrfGeometryKind,
        confidence: Float,
        validUntilNs: Long
    ) = UstrfMetricGeometryEvidence(
        forwardMeters = forward * CELL_SIZE_METERS,
        lateralMeters = lateral * CELL_SIZE_METERS,
        heightBand = heightBand,
        kind = kind,
        confidence = confidence,
        source = "synthetic-fixture-v1",
        validUntilNs = validUntilNs
    )

    private fun healthy() = UstrfHealth(
        pose = UstrfPoseState.TRACKING,
        capture = UstrfEvidenceState.VALID,
        geometry = UstrfEvidenceState.VALID,
        motion = UstrfEvidenceState.VALID
    )

    private data class Fixture(val input: UstrfSessionInput, val expectation: UstrfOfflineSafetyExpectation)

    private companion object {
        const val CELL_SIZE_METERS = .5f
        const val HALF_WIDTH_CELLS = 3
        const val HORIZON_CELLS = 4
    }
}
