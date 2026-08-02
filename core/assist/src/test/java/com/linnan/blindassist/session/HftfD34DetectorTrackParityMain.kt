package com.linnan.blindassist.session

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlin.math.abs

object HftfD34DetectorTrackParityMain {
    private const val EXPECTED_ROWS = 5_366
    private const val MAXIMUM_SLOPE_ERROR = 1e-5
    private const val MAXIMUM_P95_MS = 0.10
    private val frameSize = FrameSize(3_760, 480)

    @JvmStatic
    fun main(args: Array<String>) {
        require(args.size == 2) { "usage: <parity_input.tsv> <report.json>" }
        val input = File(args[0])
        val output = File(args[1])
        val rows = readRows(input)
        require(rows.size == EXPECTED_ROWS) {
            "D34 expected $EXPECTED_ROWS rows, got ${rows.size}"
        }
        execute(rows, measure = false)
        val result = execute(rows, measure = true)
        val p50Ms = percentile(result.latenciesNs, 0.50) / 1_000_000.0
        val p95Ms = percentile(result.latenciesNs, 0.95) / 1_000_000.0
        val p99Ms = percentile(result.latenciesNs, 0.99) / 1_000_000.0
        val supported =
            result.decisionMismatches == 0 &&
                result.slopePresenceMismatches == 0 &&
                result.maximumSlopeError <= MAXIMUM_SLOPE_ERROR &&
                p95Ms <= MAXIMUM_P95_MS
        val status =
            if (supported) {
                "D34_KOTLIN_SHADOW_STATE_PARITY_RUNTIME_SUPPORTED"
            } else {
                "D34_KOTLIN_SHADOW_STATE_PARITY_RUNTIME_NOT_SUPPORTED"
            }
        val report = """
            {
              "schema": "blindassist_hftf_stage_c_d34_kotlin_shadow_state_parity_v0",
              "status": "$status",
              "supported": $supported,
              "input_sha256": "${sha256(input)}",
              "row_count": ${rows.size},
              "distinct_tracks": ${rows.map { it.sequence to it.trackId }.distinct().size},
              "decision_mismatches": ${result.decisionMismatches},
              "slope_presence_mismatches": ${result.slopePresenceMismatches},
              "maximum_absolute_slope_error_per_s": ${result.maximumSlopeError},
              "producer_call_p50_ms": $p50Ms,
              "producer_call_p95_ms": $p95Ms,
              "producer_call_p99_ms": $p99Ms,
              "maximum_p95_gate_ms": $MAXIMUM_P95_MS,
              "non_actuating": true,
              "future_truth_consumed": false
            }
        """.trimIndent() + "\n"
        output.parentFile.mkdirs()
        val partial = File(output.parentFile, output.name + ".partial")
        partial.writeText(report, Charsets.UTF_8)
        Files.move(
            partial.toPath(),
            output.toPath(),
            StandardCopyOption.REPLACE_EXISTING,
            StandardCopyOption.ATOMIC_MOVE
        )
        println(report.trim())
        check(supported) { status }
    }

    private fun execute(rows: List<Row>, measure: Boolean): Result {
        var producer = CausalTrackTristateGeometryProducer()
        var key: Pair<String, Int>? = null
        var previousFrame: Int? = null
        var decisionMismatches = 0
        var slopePresenceMismatches = 0
        var maximumSlopeError = 0.0
        val latencies = ArrayList<Long>(rows.size)
        rows.forEach { row ->
            val currentKey = row.sequence to row.trackId
            if (currentKey != key) {
                producer = CausalTrackTristateGeometryProducer()
                key = currentKey
                previousFrame = null
            }
            if (previousFrame != null && row.frameIndex != previousFrame!! + 1) {
                producer.reset()
            }
            previousFrame = row.frameIndex
            val stamp = FrameStamp(
                frameId = row.frameIndex.toLong(),
                capturedAtNs = row.timestampNs,
                receivedAtNs = row.timestampNs + 1_000_000L,
                sourceId = "d34:${row.sequence}:${row.trackId}",
                coordinateFrame = "jrdb:stitched-rgb",
                clockDomain = FrameClockDomain.REPLAY_TIMELINE
            )
            val detection = Detection(
                classId = 0,
                label = "person",
                confidence = 1f,
                boundingBox = BoundingBox(
                    row.left,
                    row.top,
                    row.right,
                    row.bottom
                ),
                frameSize = frameSize
            )
            val started = if (measure) System.nanoTime() else 0L
            val actual = requireNotNull(
                producer.produce(
                    stamp,
                    detection,
                    row.timestampNs + 2_000_000L
                )
            )
            if (measure) latencies += System.nanoTime() - started
            if (actual.correctionDecision.name != row.expectedDecision) {
                decisionMismatches += 1
            }
            val actualSlope = actual.signedApproachRatePerS?.toDouble()
            if ((actualSlope == null) != (row.expectedSlope == null)) {
                slopePresenceMismatches += 1
            } else if (actualSlope != null && row.expectedSlope != null) {
                maximumSlopeError = maxOf(
                    maximumSlopeError,
                    abs(actualSlope - row.expectedSlope)
                )
            }
        }
        return Result(
            decisionMismatches,
            slopePresenceMismatches,
            maximumSlopeError,
            latencies
        )
    }

    private fun readRows(input: File): List<Row> {
        val lines = input.readLines(Charsets.UTF_8)
        require(lines.isNotEmpty()) { "D34 input is empty" }
        val header = lines.first().split('\t')
        val index = header.withIndex().associate { it.value to it.index }
        fun value(parts: List<String>, name: String): String =
            parts[requireNotNull(index[name]) { "D34 missing column $name" }]
        return lines.drop(1).filter { it.isNotBlank() }.map { line ->
            val parts = line.split('\t')
            Row(
                sequence = value(parts, "sequence"),
                trackId = value(parts, "track_id").toInt(),
                frameIndex = value(parts, "frame_index").toInt(),
                timestampNs = value(parts, "timestamp_ns").toLong(),
                left = value(parts, "left").toFloat(),
                top = value(parts, "top").toFloat(),
                right = value(parts, "right").toFloat(),
                bottom = value(parts, "bottom").toFloat(),
                expectedDecision = value(parts, "expected_decision"),
                expectedSlope = value(parts, "expected_slope_per_s")
                    .takeIf { it.isNotEmpty() }
                    ?.toDouble()
            )
        }
    }

    private fun percentile(values: List<Long>, quantile: Double): Long {
        require(values.isNotEmpty())
        val ordered = values.sorted()
        val index = ((ordered.size - 1) * quantile).toInt()
        return ordered[index]
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private data class Row(
        val sequence: String,
        val trackId: Int,
        val frameIndex: Int,
        val timestampNs: Long,
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float,
        val expectedDecision: String,
        val expectedSlope: Double?
    )

    private data class Result(
        val decisionMismatches: Int,
        val slopePresenceMismatches: Int,
        val maximumSlopeError: Double,
        val latenciesNs: List<Long>
    )
}
