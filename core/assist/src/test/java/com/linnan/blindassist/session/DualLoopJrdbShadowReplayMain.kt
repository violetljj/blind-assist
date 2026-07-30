package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

/**
 * Explicit host-only entry point for the JRDB engineering shadow cycle.
 *
 * Usage is intentionally separate from normal tests and runtime assembly:
 *   ./gradlew :core:assist:runDualLoopJrdbShadowReplay \
 *     --args="<frozen.tsv> <producer-receipt.json> <kernel-receipt.json>"
 */
object DualLoopJrdbShadowReplayMain {
    @JvmStatic
    fun main(args: Array<String>) {
        require(args.size == 3) {
            "expected <frozen.tsv> <producer-receipt.json> <kernel-receipt.json>"
        }
        val input = Path.of(args[0])
        val producerReceipt = Path.of(args[1])
        val output = Path.of(args[2])
        require(Files.isRegularFile(input)) { "input TSV does not exist: $input" }
        require(Files.isRegularFile(producerReceipt)) {
            "producer receipt does not exist: $producerReceipt"
        }
        require(!Files.exists(output)) { "output receipt already exists: $output" }
        validateProducerReceipt(producerReceipt, input)

        val rows = readRows(input)
        require(rows.isNotEmpty()) { "input TSV has no replay rows" }
        validateReplayRows(rows)
        val frameGroups = rows.groupBy { it.sequence to it.frameIndex }.toSortedMap(
            compareBy<Pair<String, Long>> { it.first }.thenBy { it.second }
        )
        val dispositions = linkedMapOf<String, Int>()
        val admittedBySequence = linkedMapOf<String, Int>()
        var selectedTargets = 0
        var selectedGeometryEligible = 0
        var admitted = 0
        var adapterAbstained = 0
        var nonmutationMismatches = 0
        var baselineNotifyCalls = 0
        var shadowNotifyCalls = 0

        for ((sequence, sequenceFrames) in frameGroups.entries.groupBy { it.key.first }) {
            val baselineGateway = PlannerGateway()
            val shadowGateway = PlannerGateway()
            val baseline = AssistDecisionKernel()
            val shadow = AssistDecisionKernel(
                dualLoopShadowAdmitter = DualLoopShadowAdmitter(
                    admittedSourceIdentities = setOf(
                        DualLoopSourceIdentity(
                            DualLoopShadowAdmitter.CONTRACT_ID,
                            DualLoopJrdbReplayAdapter.SOURCE_ID
                        )
                    )
                )
            )
            val firstNowMs = sequenceFrames.first().value.first().imageTimestampNs / NANOS_PER_MILLISECOND
            baseline.startSession((firstNowMs - 1L).coerceAtLeast(0L))
            shadow.startSession((firstNowMs - 1L).coerceAtLeast(0L))

            for ((_, frameRows) in sequenceFrames) {
                val first = frameRows.first()
                require(frameRows.all { row ->
                    row.imageTimestampNs == first.imageTimestampNs &&
                        row.availableAtNs == first.availableAtNs &&
                        row.frameWidth == first.frameWidth &&
                        row.frameHeight == first.frameHeight
                }) { "frame metadata differs within $sequence/${first.frameIndex}" }
                val detections = frameRows.map { it.detection() }
                val frameSize = FrameSize(first.frameWidth, first.frameHeight)
                val currentFrame = first.currentFrame()
                val nowMs = first.imageTimestampNs / NANOS_PER_MILLISECOND
                val baselineResult = baseline.processFrame(
                    detections = detections,
                    frameSize = frameSize,
                    profile = AlertProfile.STANDARD,
                    scenario = AssistScenario.GENERAL,
                    metrics = metrics(),
                    feedbackGateway = baselineGateway,
                    nowMs = nowMs,
                    sourceFrame = currentFrame,
                    decisionAtNs = first.availableAtNs
                )
                val selected = baselineResult.evaluation.rawRisk.sourceDetection
                val selectedRow = selected?.let { detection ->
                    selectedTargets += 1
                    frameRows.singleOrNull { it.detection() == detection }
                }
                val evidence = if (selectedRow?.geometryStatus == GEOMETRY_ELIGIBLE) {
                    selectedGeometryEligible += 1
                    when (val proposal = DualLoopJrdbReplayAdapter.adapt(selectedRow.replayInput())) {
                        is DualLoopJrdbReplayProposal.Available -> proposal.evidence
                        is DualLoopJrdbReplayProposal.Abstained -> {
                            adapterAbstained += 1
                            null
                        }
                    }
                } else {
                    null
                }
                val shadowResult = shadow.processFrame(
                    detections = detections,
                    frameSize = frameSize,
                    profile = AlertProfile.STANDARD,
                    scenario = AssistScenario.GENERAL,
                    metrics = metrics(),
                    feedbackGateway = shadowGateway,
                    nowMs = nowMs,
                    sourceFrame = currentFrame,
                    decisionAtNs = first.availableAtNs,
                    dualLoopMode = DualLoopRuntimeMode.SHADOW_ABSTAIN_ONLY,
                    dualLoopGeometryEvidence = evidence,
                    dualLoopDecisionClockDomain = FrameClockDomain.REPLAY_TIMELINE
                )
                val disposition = shadowResult.evaluation.dualLoopShadow.disposition.name
                dispositions[disposition] = dispositions.getOrDefault(disposition, 0) + 1
                if (shadowResult.evaluation.dualLoopShadow.admitted) {
                    admitted += 1
                    admittedBySequence[sequence] = admittedBySequence.getOrDefault(sequence, 0) + 1
                }
                if (!sameWithoutShadow(baselineResult, shadowResult) ||
                    baselineGateway.risks != shadowGateway.risks
                ) {
                    nonmutationMismatches += 1
                }
            }
            baselineNotifyCalls += baselineGateway.notifyCalls
            shadowNotifyCalls += shadowGateway.notifyCalls
        }

        val valid = admitted > 0 &&
            adapterAbstained == 0 &&
            nonmutationMismatches == 0 &&
            baselineNotifyCalls == shadowNotifyCalls
        val terminal = if (valid) {
            "ENGINEERING_SHADOW_CYCLE_VALID / DIAGNOSTIC_ONLY / NO_EFFECT_CLAIM"
        } else {
            "ENGINEERING_SHADOW_CYCLE_INVALID"
        }
        val receipt = buildReceipt(
            valid = valid,
            terminal = terminal,
            input = input,
            producerReceipt = producerReceipt,
            rows = rows,
            frameCount = frameGroups.size,
            selectedTargets = selectedTargets,
            selectedGeometryEligible = selectedGeometryEligible,
            admitted = admitted,
            adapterAbstained = adapterAbstained,
            nonmutationMismatches = nonmutationMismatches,
            baselineNotifyCalls = baselineNotifyCalls,
            shadowNotifyCalls = shadowNotifyCalls,
            dispositions = dispositions,
            admittedBySequence = admittedBySequence
        )
        Files.createDirectories(output.toAbsolutePath().parent)
        Files.writeString(output, receipt, StandardCharsets.UTF_8)
        println(receipt)
        check(valid) { terminal }
    }

    private fun sameWithoutShadow(
        baseline: AssistFrameResult,
        shadow: AssistFrameResult
    ): Boolean =
            baseline.evaluation.rawRisk == shadow.evaluation.rawRisk &&
            baseline.evaluation.preTemporalRisk == shadow.evaluation.preTemporalRisk &&
            baseline.evaluation.stableRisk == shadow.evaluation.stableRisk &&
            baseline.evaluation.riskEvent == shadow.evaluation.riskEvent &&
            baseline.feedbackDecision == shadow.feedbackDecision &&
            baseline.explanation == shadow.explanation &&
            baseline.sessionSummary == shadow.sessionSummary &&
            shadow.evaluation.dualLoopShadow.productionRiskUnchanged &&
            !shadow.evaluation.dualLoopShadow.eventMutationAllowed &&
            !shadow.evaluation.dualLoopShadow.feedbackMutationAllowed

    private fun readRows(path: Path): List<ReplayRow> {
        val lines = Files.readAllLines(path, StandardCharsets.UTF_8)
        require(lines.size >= 2) { "input TSV must contain a header and data" }
        val header = lines.first().split('\t')
        val index = header.withIndex().associate { it.value to it.index }
        return lines.drop(1).filter { it.isNotBlank() }.mapIndexed { rowIndex, line ->
            val columns = line.split('\t', ignoreCase = false, limit = header.size)
            require(columns.size == header.size) { "malformed TSV row ${rowIndex + 2}" }
            fun value(name: String): String = columns[index.getValue(name)]
            ReplayRow(
                sequence = value("sequence"),
                frameIndex = value("frame_index").toLong(),
                imageTimestampNs = value("image_timestamp_ns").toLong(),
                availableAtNs = value("available_at_ns").toLong(),
                frameWidth = value("frame_width").toInt(),
                frameHeight = value("frame_height").toInt(),
                labelId = value("label_id"),
                left = value("left").toFloat(),
                top = value("top").toFloat(),
                right = value("right").toFloat(),
                bottom = value("bottom").toFloat(),
                geometryStatus = value("geometry_status"),
                previousFrameIndex = value("previous_frame_index").toLongOrNull(),
                previousTimestampNs = value("previous_timestamp_ns").toLongOrNull(),
                signedApproachRatePerS = value("signed_approach_rate_per_s").toFloatOrNull(),
                quality = value("quality").toFloatOrNull()
            )
        }
    }

    private fun buildReceipt(
        valid: Boolean,
        terminal: String,
        input: Path,
        producerReceipt: Path,
        rows: List<ReplayRow>,
        frameCount: Int,
        selectedTargets: Int,
        selectedGeometryEligible: Int,
        admitted: Int,
        adapterAbstained: Int,
        nonmutationMismatches: Int,
        baselineNotifyCalls: Int,
        shadowNotifyCalls: Int,
        dispositions: Map<String, Int>,
        admittedBySequence: Map<String, Int>
    ): String {
        val sequenceCount = rows.map { it.sequence }.toSet().size
        val eligibleRows = rows.count { it.geometryStatus == GEOMETRY_ELIGIBLE }
        return buildString {
            append("{\n")
            append("  \"schema\": \"blindassist.dual_loop_jrdb_shadow_replay_receipt.v1\",\n")
            append("  \"status\": \"").append(if (valid) "VALID" else "INVALID").append("\",\n")
            append("  \"terminal\": ").append(jsonString(terminal)).append(",\n")
            append("  \"claim_ceiling\": \"DIAGNOSTIC_ENGINEERING_ONLY\",\n")
            append("  \"runner_implementation_id\": ")
                .append(jsonString(RUNNER_IMPLEMENTATION_ID)).append(",\n")
            append("  \"producer_implementation_sha256\": ")
                .append(jsonString(EXPECTED_PRODUCER_IMPLEMENTATION_SHA256)).append(",\n")
            append("  \"replay_denominators_recomputed\": true,\n")
            append("  \"source_contract_id\": ")
                .append(jsonString(DualLoopShadowAdmitter.CONTRACT_ID)).append(",\n")
            append("  \"source_id\": ")
                .append(jsonString(DualLoopJrdbReplayAdapter.SOURCE_ID)).append(",\n")
            append("  \"input_path\": ").append(jsonString(input.toString())).append(",\n")
            append("  \"input_sha256\": ").append(jsonString(sha256(input))).append(",\n")
            append("  \"producer_receipt_path\": ")
                .append(jsonString(producerReceipt.toString())).append(",\n")
            append("  \"producer_receipt_sha256\": ")
                .append(jsonString(sha256(producerReceipt))).append(",\n")
            append("  \"sequence_count\": ").append(sequenceCount).append(",\n")
            append("  \"frame_count\": ").append(frameCount).append(",\n")
            append("  \"detection_rows\": ").append(rows.size).append(",\n")
            append("  \"source_eligible_rows\": ").append(eligibleRows).append(",\n")
            append("  \"baseline_selected_targets\": ").append(selectedTargets).append(",\n")
            append("  \"selected_geometry_eligible\": ").append(selectedGeometryEligible).append(",\n")
            append("  \"kernel_admitted\": ").append(admitted).append(",\n")
            append("  \"adapter_abstained\": ").append(adapterAbstained).append(",\n")
            append("  \"nonmutation_mismatches\": ").append(nonmutationMismatches).append(",\n")
            append("  \"baseline_notify_calls\": ").append(baselineNotifyCalls).append(",\n")
            append("  \"shadow_notify_calls\": ").append(shadowNotifyCalls).append(",\n")
            append("  \"production_risk_unchanged\": ").append(nonmutationMismatches == 0).append(",\n")
            append("  \"event_mutation_allowed\": false,\n")
            append("  \"feedback_mutation_allowed\": false,\n")
            append("  \"production_allowlist_unchanged\": true,\n")
            append("  \"dispositions\": ").append(jsonMap(dispositions)).append(",\n")
            append("  \"admitted_by_sequence\": ").append(jsonMap(admittedBySequence)).append("\n")
            append("}\n")
        }
    }

    private fun jsonMap(values: Map<String, Int>): String =
        values.toSortedMap().entries.joinToString(prefix = "{", postfix = "}") { (key, value) ->
            "${jsonString(key)}:$value"
        }

    private fun jsonString(value: String): String =
        "\"" + value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n") + "\""

    private fun sha256(path: Path): String {
        val digest = MessageDigest.getInstance("SHA-256")
        Files.newInputStream(path).use { stream ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private fun metrics() = DetectorMetrics(0L, 0L, 0L, 0L, 0f, "jrdb-replay")

    private data class ReplayRow(
        val sequence: String,
        val frameIndex: Long,
        val imageTimestampNs: Long,
        val availableAtNs: Long,
        val frameWidth: Int,
        val frameHeight: Int,
        val labelId: String,
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float,
        val geometryStatus: String,
        val previousFrameIndex: Long?,
        val previousTimestampNs: Long?,
        val signedApproachRatePerS: Float?,
        val quality: Float?
    ) {
        fun detection() = Detection(
            classId = 0,
            label = "person",
            confidence = 1f,
            boundingBox = BoundingBox(left, top, right, bottom),
            frameSize = FrameSize(frameWidth, frameHeight),
            source = DetectionSource.OBJECT_DETECTOR
        )

        fun currentFrame() = FrameStamp(
            frameId = frameIndex,
            capturedAtNs = imageTimestampNs,
            receivedAtNs = availableAtNs,
            sourceId = sequence,
            coordinateFrame = DualLoopJrdbReplayAdapter.COORDINATE_FRAME,
            clockDomain = FrameClockDomain.REPLAY_TIMELINE
        )

        fun replayInput(): JrdbAnnotationConditionedLidarReplayInput {
            require(geometryStatus == GEOMETRY_ELIGIBLE)
            return JrdbAnnotationConditionedLidarReplayInput(
                previousFrame = FrameStamp(
                    frameId = requireNotNull(previousFrameIndex),
                    capturedAtNs = requireNotNull(previousTimestampNs),
                    receivedAtNs = requireNotNull(previousTimestampNs),
                    sourceId = sequence,
                    coordinateFrame = DualLoopJrdbReplayAdapter.COORDINATE_FRAME,
                    clockDomain = FrameClockDomain.REPLAY_TIMELINE
                ),
                currentFrame = currentFrame(),
                availableAtNs = availableAtNs,
                validUntilNs = Math.addExact(availableAtNs, TTL_NS),
                trackEpoch = "$sequence:$labelId",
                target = detection(),
                signedApproachRatePerS = requireNotNull(signedApproachRatePerS),
                quality = requireNotNull(quality)
            )
        }
    }

    private class PlannerGateway : FeedbackGateway {
        val risks = mutableListOf<RiskResult>()
        val notifyCalls: Int
            get() = risks.size

        override fun resetSession() = Unit

        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            risks += risk
            val plan = FeedbackPlanner.planFor(risk, profile = profile, scenario = scenario)
            return if (plan == null) {
                FeedbackDecision(null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)
            } else {
                FeedbackDecision(plan, triggered = true, reason = FeedbackReason.TRIGGERED)
            }
        }
    }

    private const val GEOMETRY_ELIGIBLE = "ELIGIBLE"
    private const val NANOS_PER_MILLISECOND = 1_000_000L
    private const val TTL_NS = 100_000_000L

    internal fun validateProducerReceipt(receiptPath: Path, inputPath: Path) {
        val document = Files.readString(receiptPath, StandardCharsets.UTF_8)
        require(topLevelString(document, "schema") ==
            "blindassist.dual_loop_jrdb_shadow_producer_receipt.v1")
        require(topLevelString(document, "status") == "COMPLETE")
        require(topLevelString(document, "source_id") == DualLoopJrdbReplayAdapter.SOURCE_ID)
        require(topLevelString(document, "claim_ceiling") == "DIAGNOSTIC_ENGINEERING_ONLY")
        require(topLevelString(document, "output_sha256") == sha256(inputPath))
        require(topLevelString(document, "implementation_sha256") ==
            EXPECTED_PRODUCER_IMPLEMENTATION_SHA256)
        require(topLevelLong(document, "sequence_count") == 4L)
        require(topLevelLong(document, "frame_count") == 480L)
        require(topLevelLong(document, "detection_rows") == 10_786L)
        require(topLevelLong(document, "source_eligible_rows") == 8_836L)
        require(!topLevelBoolean(document, "protected_alert_outcomes_opened"))
    }

    private fun validateReplayRows(rows: List<ReplayRow>) {
        require(rows.size == EXPECTED_REPLAY_ROWS) {
            "replay row denominator drift: ${rows.size}"
        }
        val rowsBySequence = rows.groupBy { it.sequence }
        require(rowsBySequence.keys == EXPECTED_SEQUENCE_DENOMINATORS.keys) {
            "replay sequence identity drift: ${rowsBySequence.keys.sorted()}"
        }
        for ((sequence, expected) in EXPECTED_SEQUENCE_DENOMINATORS) {
            val sequenceRows = rowsBySequence.getValue(sequence)
            val decisionFrames = sequenceRows.map { it.frameIndex }.toSet().size
            val eligibleRows = sequenceRows.count { it.geometryStatus == GEOMETRY_ELIGIBLE }
            require(sequenceRows.size == expected.rows) {
                "$sequence replay row denominator drift: ${sequenceRows.size}"
            }
            require(decisionFrames == expected.decisionFrames) {
                "$sequence decision-frame denominator drift: $decisionFrames"
            }
            require(eligibleRows == expected.eligibleRows) {
                "$sequence eligible-row denominator drift: $eligibleRows"
            }
        }
        val decisionFrames = rowsBySequence.values.sumOf { sequenceRows ->
            sequenceRows.map { it.frameIndex }.toSet().size
        }
        val eligibleRows = rows.count { it.geometryStatus == GEOMETRY_ELIGIBLE }
        require(decisionFrames == EXPECTED_DECISION_FRAMES) {
            "replay decision-frame denominator drift: $decisionFrames"
        }
        require(eligibleRows == EXPECTED_ELIGIBLE_ROWS) {
            "replay eligible-row denominator drift: $eligibleRows"
        }
    }

    private fun topLevelString(document: String, key: String): String =
        topLevelMatch(document, key, "\"([^\"]*)\"")

    private fun topLevelLong(document: String, key: String): Long =
        topLevelMatch(document, key, "([0-9]+)").toLong()

    private fun topLevelBoolean(document: String, key: String): Boolean =
        topLevelMatch(document, key, "(true|false)").toBooleanStrict()

    private fun topLevelMatch(document: String, key: String, valuePattern: String): String {
        val pattern = Regex(
            "(?m)^  \"${Regex.escape(key)}\": $valuePattern,?\\s*$"
        )
        val matches = pattern.findAll(document).toList()
        require(matches.size == 1) { "producer receipt field is missing or ambiguous: $key" }
        return matches.single().groupValues[1]
    }

    private data class ReplayDenominators(
        val rows: Int,
        val decisionFrames: Int,
        val eligibleRows: Int
    )

    private val EXPECTED_SEQUENCE_DENOMINATORS = linkedMapOf(
        "clark-center-2019-02-28_0" to ReplayDenominators(5_196, 119, 3_763),
        "gates-basement-elevators-2019-01-17_1" to ReplayDenominators(1_741, 119, 1_516),
        "meyer-green-2019-03-16_0" to ReplayDenominators(1_312, 119, 1_044),
        "stlc-111-2019-04-19_0" to ReplayDenominators(2_537, 119, 2_513)
    )
    private const val EXPECTED_REPLAY_ROWS = 10_786
    private const val EXPECTED_DECISION_FRAMES = 476
    private const val EXPECTED_ELIGIBLE_ROWS = 8_836
    private const val EXPECTED_PRODUCER_IMPLEMENTATION_SHA256 =
        "dd72ece0e910363507c60b873abc82726ae0e825353a31c51e9da1f7ae3ef4aa"
    private const val RUNNER_IMPLEMENTATION_ID =
        "DUAL_LOOP_JRDB_SHADOW_KERNEL_RUNNER_R1_DENOMINATOR_BOUND"
}
