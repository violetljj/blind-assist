package com.linnan.blindassist.ustrf

import java.nio.file.Files
import java.nio.file.Path

data class UstrfSyntheticCorridorSafetyReplayMetrics(
    val sceneCount: Int,
    val expectedCriticalCount: Int,
    val criticalStopCount: Int,
    val expectedStopCount: Int,
    val actualStopCount: Int,
    val clearSceneCount: Int,
    val clearStopCount: Int,
    val actionMatchCount: Int,
    val eligibleCorridorSelectionCount: Int,
    val matchingCorridorSelectionCount: Int,
    val faultSceneCount: Int,
    val faultStopCount: Int
)

/**
 * Replays the CUDA-generated labelled body-capsule scenarios through the same geometry assembler,
 * risk field, corridor planner and supervisor used by the pure Kotlin safety kernel.  The manifest
 * is analytic truth, not image perception or a production safety authorization.
 */
class UstrfSyntheticCorridorSafetyReplay {
    fun replay(root: Path): UstrfSyntheticCorridorSafetyReplayMetrics {
        val lines = Files.readAllLines(root.resolve("kotlin_corridor_safety_replay.tsv"))
        require(lines.size >= 2) { "corridor safety replay manifest has no rows" }
        require(lines.first().split('\t') == HEADER) { "unexpected corridor safety replay header" }
        var critical = 0; var criticalStops = 0; var expectedStops = 0; var actualStops = 0; var clear = 0; var clearStops = 0
        var actionMatches = 0; var eligibleSelections = 0; var matchingSelections = 0
        var faults = 0; var faultStops = 0
        lines.drop(1).forEachIndexed { index, line ->
            val values = line.split('\t')
            require(values.size == HEADER.size) { "malformed corridor safety replay row" }
            val row = HEADER.indices.associate { HEADER[it] to values[it] }
            val record = run(row, index.toLong() + 1L)
            val expectedAction = UstrfSafetyAction.valueOf(row.getValue("expected_action"))
            require(record.decision.action == expectedAction) { "action mismatch: ${row.getValue("scenario_id")}" }
            actionMatches += 1
            if (expectedAction == UstrfSafetyAction.STOP_AND_REASSESS) expectedStops += 1
            if (record.decision.action == UstrfSafetyAction.STOP_AND_REASSESS) actualStops += 1
            if (row.bool("expected_critical")) {
                critical += 1
                if (record.decision.action == UstrfSafetyAction.STOP_AND_REASSESS) criticalStops += 1
            }
            if (row.getValue("family") == "clear") {
                clear += 1
                if (record.decision.action == UstrfSafetyAction.STOP_AND_REASSESS) clearStops += 1
            }
            if (row.getValue("fault") != "none") {
                faults += 1
                if (record.decision.action == UstrfSafetyAction.STOP_AND_REASSESS) faultStops += 1
            } else {
                eligibleSelections += 1
                val raw = row.getValue("expected_selected_offset")
                val expected = raw.takeIf { it.isNotBlank() }?.toInt()
                require(record.decision.experimentalCorridorOffsetCells == expected) {
                    "corridor mismatch: ${row.getValue("scenario_id")}; expected=$expected actual=${record.decision.experimentalCorridorOffsetCells}"
                }
                matchingSelections += 1
            }
        }
        return UstrfSyntheticCorridorSafetyReplayMetrics(lines.size - 1, critical, criticalStops, expectedStops, actualStops, clear, clearStops, actionMatches, eligibleSelections, matchingSelections, faults, faultStops)
    }

    private fun run(row: Map<String, String>, frameId: Long): UstrfSessionRecord {
        val frame = UstrfFrameStamp(frameId, CAPTURED_AT_NS, BODY_FRAME)
        val validUntil = CAPTURED_AT_NS + TTL_NS
        val tokens = row.getValue("hazards").split(';').filter { it.isNotBlank() }.map { token ->
            val parts = token.split(':'); require(parts.size == 3) { "malformed hazard token: $token" }
            Hazard(parts[0], parts[1].toInt(), parts[2].toInt())
        }
        val unknown = tokens.filter { it.kind == "U" }.map { UstrfGridCoordinate(it.lateral, it.forward) }.toSet()
        val geometry = buildList {
            for (lateral in -3..3) for (forward in 1..4) {
                if (UstrfGridCoordinate(lateral, forward) !in unknown) add(evidence(lateral, forward, UstrfHeightBand.GROUND, UstrfGeometryKind.TRAVERSABLE, validUntil))
            }
            tokens.filter { it.kind != "M" && it.kind != "U" }.forEach { hazard ->
                when (hazard.kind) {
                    "O" -> add(evidence(hazard.lateral, hazard.forward, UstrfHeightBand.LOWER_BODY, UstrfGeometryKind.OCCUPIED, validUntil))
                    "D" -> add(evidence(hazard.lateral, hazard.forward, UstrfHeightBand.GROUND, UstrfGeometryKind.DROP, validUntil))
                    "H" -> add(evidence(hazard.lateral, hazard.forward, UstrfHeightBand.HEAD, UstrfGeometryKind.HEAD_OBSTACLE, validUntil))
                    else -> error("unknown geometry kind: ${hazard.kind}")
                }
            }
        }
        val dynamic = tokens.filter { it.kind == "M" }.map { hazard ->
            UstrfMotionGridEvidence(frame, UstrfGridCoordinate(hazard.lateral, hazard.forward), UstrfRelativeMotionEvidence(UstrfVector2(1f, 0f), UstrfVector2(-1f, 0f), CAPTURED_AT_NS, validUntil, "synthetic-corridor-motion"))
        }
        val fault = row.getValue("fault")
        val now = if (fault == "stale_geometry") validUntil + 1L else CAPTURED_AT_NS
        val health = UstrfHealth(
            pose = if (fault == "pose_lost") UstrfPoseState.LOST else UstrfPoseState.TRACKING,
            capture = UstrfEvidenceState.VALID, geometry = UstrfEvidenceState.VALID, motion = UstrfEvidenceState.VALID
        )
        val packet = UstrfGeometryPacket(frame, CAPTURED_AT_NS, validUntil, UstrfDepthScale.METRIC, geometry)
        val assembled = UstrfPerceptionAssembler(UstrfGeometryProjector(halfWidthCells = 3, horizonCells = 4)).assemble(frame, packet, dynamic, now)
        return UstrfSafetySession(
            fieldBuilder = UstrfRiskFieldBuilder(UstrfRiskFieldConfig(halfWidthCells = 3, horizonCells = 4)),
            planner = UstrfCorridorPlanner(horizonCells = 4, capsuleHalfWidthCells = 1, fixedCandidateOffsets = listOf(-2, -1, 0, 1, 2))
        ).evaluate(UstrfSessionInput(frame, health, assembled, UstrfRouteIntent(BODY_FRAME, 0, 1f, validUntil), decisionAtNs = now))
    }

    private fun evidence(lateral: Int, forward: Int, band: UstrfHeightBand, kind: UstrfGeometryKind, validUntil: Long) =
        UstrfMetricGeometryEvidence(forward * .5f, lateral * .5f, band, kind, if (kind == UstrfGeometryKind.TRAVERSABLE) 1f else .95f, "synthetic-corridor-manifest", validUntil)

    private fun Map<String, String>.bool(key: String) = getValue(key).toBooleanStrict()
    private data class Hazard(val kind: String, val lateral: Int, val forward: Int)

    private companion object {
        const val BODY_FRAME = "synthetic-body-local-v1"
        const val CAPTURED_AT_NS = 1_000_000_000L
        const val TTL_NS = 1_000_000_000L
        val HEADER = listOf("scenario_id", "family", "fault", "hazards", "expected_action", "expected_selected_offset", "expected_has_safe_corridor", "expected_critical")
    }
}
