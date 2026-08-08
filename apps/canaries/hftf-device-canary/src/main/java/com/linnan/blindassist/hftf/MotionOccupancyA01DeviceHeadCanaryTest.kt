package com.linnan.blindassist.hftf

import android.os.Build
import android.os.SystemClock
import android.util.AtomicFile
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.BuildConfig
import java.io.File
import java.security.MessageDigest
import java.util.zip.GZIPInputStream
import kotlin.math.abs
import kotlin.math.exp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MotionOccupancyA01DeviceHeadCanaryTest {
    @Test
    fun frozenProbabilityHeadParityAndRuntime() {
        assertTrue(BuildConfig.DUAL_LOOP_SHADOW)
        assertFalse(BuildConfig.DUAL_LOOP_ACTIVE)
        val corpus = loadCorpus()
        assertEquals(EXPECTED_ROWS, corpus.rows.size)
        assertEquals(EXPECTED_RAW_SHA256, corpus.sha256)
        repeat(WARMUP_PASSES) { corpus.rows.forEach { predict(it.features) } }

        var probabilityMismatches = 0
        var decisionMismatches = 0
        var maximumAbsoluteError = 0.0
        val latencies = ArrayList<Long>(corpus.rows.size * MEASURED_PASSES)
        repeat(MEASURED_PASSES) {
            corpus.rows.forEach { row ->
                val started = SystemClock.elapsedRealtimeNanos()
                val actual = predict(row.features)
                latencies += SystemClock.elapsedRealtimeNanos() - started
                val error = abs(actual - row.expectedProbability)
                maximumAbsoluteError = maxOf(maximumAbsoluteError, error)
                if (error > MAXIMUM_PROBABILITY_ERROR) probabilityMismatches += 1
                if ((actual >= DECISION_THRESHOLD) !=
                    (row.expectedProbability >= DECISION_THRESHOLD)
                ) {
                    decisionMismatches += 1
                }
            }
        }
        val p50Ms = percentile(latencies, 0.50) / 1_000_000.0
        val p95Ms = percentile(latencies, 0.95) / 1_000_000.0
        val p99Ms = percentile(latencies, 0.99) / 1_000_000.0
        val expectedDevice = Build.MANUFACTURER.equals("samsung", ignoreCase = true) &&
            Build.MODEL.replace("-", "").replace("_", "") == "SMS9280"
        val supported = expectedDevice &&
            probabilityMismatches == 0 &&
            decisionMismatches == 0 &&
            maximumAbsoluteError <= MAXIMUM_PROBABILITY_ERROR &&
            p95Ms <= MAXIMUM_P95_MS
        val status = if (supported) {
            "A0_1_ANDROID_PROBABILITY_HEAD_PARITY_RUNTIME_SUPPORTED"
        } else {
            "A0_1_ANDROID_PROBABILITY_HEAD_PARITY_RUNTIME_NOT_SUPPORTED"
        }
        val report = """
            {
              "schema": "blindassist_hftf_motion_occupancy_a0_1_android_head_canary_r0",
              "status": "$status",
              "supported": $supported,
              "application_id": "${BuildConfig.APPLICATION_ID}",
              "dual_loop_shadow": ${BuildConfig.DUAL_LOOP_SHADOW},
              "dual_loop_active": ${BuildConfig.DUAL_LOOP_ACTIVE},
              "device_manufacturer": "${jsonEscape(Build.MANUFACTURER)}",
              "device_model": "${jsonEscape(Build.MODEL)}",
              "android_sdk": ${Build.VERSION.SDK_INT},
              "input_raw_sha256": "${corpus.sha256}",
              "row_count": ${corpus.rows.size},
              "measured_passes": $MEASURED_PASSES,
              "probability_mismatches": $probabilityMismatches,
              "decision_mismatches_at_0_50": $decisionMismatches,
              "maximum_absolute_probability_error": $maximumAbsoluteError,
              "head_call_p50_ms": $p50Ms,
              "head_call_p95_ms": $p95Ms,
              "head_call_p99_ms": $p99Ms,
              "maximum_p95_gate_ms": $MAXIMUM_P95_MS,
              "non_actuating": true,
              "heavy_inference_covered": false,
              "fresh_data_opened": false
            }
        """.trimIndent() + "\n"
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val output = File(
            requireNotNull(context.getExternalFilesDir("hftf-a0-1-head")),
            "report.json"
        )
        val atomicOutput = AtomicFile(output)
        val stream = atomicOutput.startWrite()
        try {
            stream.write(report.toByteArray(Charsets.UTF_8))
            atomicOutput.finishWrite(stream)
        } catch (error: Throwable) {
            atomicOutput.failWrite(stream)
            throw error
        }
        println("A0_1_HEAD_REPORT_PATH=${output.absolutePath}")
        println(report.trim())
        assertTrue(status, supported)
    }

    private fun predict(features: DoubleArray): Double {
        require(features.size == FEATURE_MEAN.size)
        var logit = WEIGHTS[0]
        for (index in features.indices) {
            logit += ((features[index] - FEATURE_MEAN[index]) / FEATURE_SCALE[index]) *
                WEIGHTS[index + 1]
        }
        return 1.0 / (1.0 + exp(-logit.coerceIn(-40.0, 40.0)))
    }

    private fun loadCorpus(): Corpus {
        val context = InstrumentationRegistry.getInstrumentation().context
        val raw = context.assets.open(ASSET_NAME).use { compressed ->
            GZIPInputStream(compressed).use { it.readBytes() }
        }
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(raw)
            .joinToString("") { "%02x".format(it) }
        val lines = raw.toString(Charsets.UTF_8).lineSequence()
            .filter { it.isNotBlank() }
            .toList()
        val header = lines.first().split('\t')
        assertEquals(listOf("sequence_id", *FEATURE_NAMES, "expected_probability"), header)
        val rows = lines.drop(1).map { line ->
            val parts = line.split('\t')
            require(parts.size == header.size)
            Row(
                features = DoubleArray(FEATURE_NAMES.size) { parts[it + 1].toDouble() },
                expectedProbability = parts.last().toDouble()
            )
        }
        return Corpus(digest, rows)
    }

    private fun percentile(values: List<Long>, quantile: Double): Long {
        val ordered = values.sorted()
        return ordered[((ordered.size - 1) * quantile).toInt()]
    }

    private fun jsonEscape(value: String): String =
        value.replace("\\", "\\\\").replace("\"", "\\\"")

    private data class Corpus(val sha256: String, val rows: List<Row>)
    private data class Row(val features: DoubleArray, val expectedProbability: Double)

    private companion object {
        const val ASSET_NAME = "motion_occupancy_a0_1_android_head.tsv.gzbin"
        const val EXPECTED_ROWS = 1_716
        const val EXPECTED_RAW_SHA256 =
            "481987b60d80237b2e86c83c35ca05ba79c131c53265ad34f565f7e2512531a6"
        const val WARMUP_PASSES = 3
        const val MEASURED_PASSES = 20
        const val MAXIMUM_PROBABILITY_ERROR = 1e-12
        const val DECISION_THRESHOLD = 0.50
        const val MAXIMUM_P95_MS = 0.10
        val FEATURE_NAMES = arrayOf(
            "clearance_margin_m", "clearance_m", "horizon_m",
            "clearance_log1p_confidence", "ground_plane_residual_m",
            "log1p_obstacle_points", "band_left", "band_center", "flow_median",
            "flow_p90", "affine_tx_norm", "affine_ty_norm",
            "abs_affine_rotation_rad", "abs_affine_log_scale",
            "affine_inlier_fraction", "residual_flow_median", "residual_flow_p90",
            "motion_missing"
        )
        val FEATURE_MEAN = doubleArrayOf(
            0.22294528019106918, 1.7229452801910758, 1.5, 0.5920675945853352,
            0.010759523491919793, 7.5421707229764205, 0.33652694610778444,
            0.34251497005988024, 0.025932226663394023, 0.03476178563774346,
            0.0009669272108178092, -0.004928146689527196, 0.009763429136503767,
            0.004541239218077676, 0.8630186056458538, 0.002878066880468891,
            0.012533300691028156, 0.023952095808383235
        )
        val FEATURE_SCALE = doubleArrayOf(
            0.6332370344930998, 0.48406866784264035, 0.408248290463863,
            0.09886699560411505, 0.005074238655923685, 0.8739591721466363,
            0.47252149226373985, 0.47455080375526065, 0.02593045398054337,
            0.03166170893949314, 0.024576481357214235, 0.03278093844216182,
            0.015490911345402038, 0.006063720048204276, 0.1897882271141751,
            0.0026358466262418807, 0.016168989780418707, 0.15289994412938196
        )
        val WEIGHTS = doubleArrayOf(
            -0.49806735357206444, -1.5073026715041855, -0.316314497312087,
            1.9629278436001065, 0.07100384359800087, 0.17033004067765217,
            0.31467739976946857, -0.06352710581095698, -0.02043060095577231,
            -0.21325902687441015, 0.11918981820113246, -0.02584152312979251,
            0.35399355792057197, 0.0033683826246120934, 0.2343492145066225,
            -0.09221277524654063, 0.20580526166391594, -0.15332271917657633,
            0.05524987451216449
        )
    }
}
