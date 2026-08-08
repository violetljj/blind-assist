package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class UstrfU0ExplicitRouteDetectionGateTest {
    private val frameSize = FrameSize(1_000, 1_000)

    @Test
    fun routeDirectionChangesWhichUnmodifiedDetectionReachesKernel() {
        val leftEpisode = episode(sample(0, 500, 1.0, true, listOf(0.35, 0.35, 0.35)))
        val rightEpisode = episode(sample(0, 500, 1.0, true, listOf(0.65, 0.65, 0.65)))
        val left = detection("left", BoundingBox(260f, 760f, 390f, 940f))
        val right = detection("right", BoundingBox(610f, 760f, 740f, 940f))

        val leftResult = UstrfU0ExplicitRouteDetectionGate.gate(leftEpisode, 0, listOf(left, right), frameSize)
        val rightResult = UstrfU0ExplicitRouteDetectionGate.gate(rightEpisode, 0, listOf(left, right), frameSize)

        assertEquals(listOf("left"), leftResult.retainedDetections.map { it.label })
        assertEquals(listOf("right"), rightResult.retainedDetections.map { it.label })
        assertEquals(left, leftResult.retainedDetections.single())
        assertEquals(right, rightResult.retainedDetections.single())
    }

    @Test
    fun latestSampleAtOrBeforeFrameIsSelectedAndFutureSampleIsIgnored() {
        val route = episode(
            sample(0, 500, 1.0, true, listOf(0.35, 0.35, 0.35)),
            sample(500, 1_000, 1.0, true, listOf(0.65, 0.65, 0.65))
        )
        val left = detection("left", BoundingBox(260f, 760f, 390f, 940f))
        val right = detection("right", BoundingBox(610f, 760f, 740f, 940f))

        val beforeFuture = UstrfU0ExplicitRouteDetectionGate.gate(route, 499, listOf(left, right), frameSize)
        val atSecond = UstrfU0ExplicitRouteDetectionGate.gate(route, 500, listOf(left, right), frameSize)

        assertEquals(0, beforeFuture.selectedSample!!.sampleIndex)
        assertEquals(listOf("left"), beforeFuture.retainedDetections.map { it.label })
        assertEquals(1, atSecond.selectedSample!!.sampleIndex)
        assertEquals(listOf("right"), atSecond.retainedDetections.map { it.label })
    }

    @Test
    fun noCausalStaleLowConfidenceOrInvalidRouteCannotUpgradeIntervention() {
        val futureOnly = episode(sample(500, 1_000, 1.0, true, listOf(0.5, 0.5, 0.5)))
        val stale = episode(sample(0, 250, 1.0, true, listOf(0.5, 0.5, 0.5)))
        val lowConfidence = episode(sample(0, 500, 0.49, true, listOf(0.5, 0.5, 0.5)))
        val invalid = episode(sample(0, 500, 1.0, false, emptyList()))
        val obstacle = detection("person", BoundingBox(400f, 600f, 600f, 990f))

        val results = listOf(
            UstrfU0ExplicitRouteDetectionGate.gate(futureOnly, 0, listOf(obstacle), frameSize),
            UstrfU0ExplicitRouteDetectionGate.gate(stale, 500, listOf(obstacle), frameSize),
            UstrfU0ExplicitRouteDetectionGate.gate(lowConfidence, 0, listOf(obstacle), frameSize),
            UstrfU0ExplicitRouteDetectionGate.gate(invalid, 0, listOf(obstacle), frameSize)
        )

        results.forEach {
            assertFalse(it.routeUsable)
            assertTrue(it.retainedDetections.isEmpty())
            assertTrue(it.rows.isEmpty())
        }
        assertEquals(
            listOf("NO_CAUSAL_ROUTE_SAMPLE", "ROUTE_STALE", "ROUTE_LOW_CONFIDENCE", "ROUTE_MARKED_INVALID"),
            results.map { it.reason }
        )
    }

    @Test
    fun malformedRuntimeWaypointContractFailsClosed() {
        val malformed = episode(sample(0, 500, 1.0, true, listOf(0.5, 0.5)))
        val result = UstrfU0ExplicitRouteDetectionGate.gate(malformed, 0, emptyList(), frameSize)

        assertFalse(result.routeUsable)
        assertEquals("ROUTE_WAYPOINT_CONTRACT_INVALID", result.reason)
    }

    @Test
    fun receiptBindsCausalSampleAndPerBoxGeometry() {
        val route = episode(sample(0, 500, 0.9, true, listOf(0.5, 0.5, 0.5)))
        val obstacle = detection("person", BoundingBox(450f, 600f, 550f, 950f))
        val config = UstrfU0RouteGateConfig()
        val result = UstrfU0ExplicitRouteDetectionGate.gate(route, 0, listOf(obstacle), frameSize, config)
        val receipt = UstrfU0ExplicitRouteDetectionGate.receiptJson(route, "frame-0", 0, result, config)

        assertTrue(receipt.getBoolean("route_usable"))
        assertEquals(0, receipt.getInt("selected_sample_index"))
        assertEquals(1, receipt.getInt("input_detection_count"))
        assertEquals(1, receipt.getInt("retained_detection_count"))
        assertTrue(receipt.getJSONArray("detections").getJSONObject(0).getBoolean("kept"))
    }

    @Test(expected = IllegalStateException::class)
    fun riskModelInferredRouteIsRejectedAtParseBoundary() {
        val value = routeJson(sample(0, 500, 1.0, true, listOf(0.5, 0.5, 0.5)))
        value.getJSONObject("provider").put("inferred_by_risk_model", true)
        UstrfU0ExplicitRouteDetectionGate.parseEpisode(value, "episode-test")
    }

    private fun episode(vararg samples: JSONObject) =
        UstrfU0ExplicitRouteDetectionGate.parseEpisode(routeJson(*samples), "episode-test")

    private fun routeJson(vararg samples: JSONObject) = JSONObject()
        .put("schema", UstrfU0ExplicitRouteDetectionGate.ROUTE_SCHEMA)
        .put("episode_id", "episode-test")
        .put("parent_source_id", "source-test")
        .put("provider", JSONObject()
            .put("type", "explicit_user_choice")
            .put("provider_id", "provider-test")
            .put("inferred_by_risk_model", false)
            .put("input_space", "current_camera_frame"))
        .put("coordinate_contract", JSONObject()
            .put("space", UstrfU0ExplicitRouteDetectionGate.COORDINATE_SPACE)
            .put("projection_receipt_id", "projection-test"))
        .put("samples", org.json.JSONArray().also { array -> samples.forEach(array::put) })
        .put("fallback", JSONObject()
            .put("missing_stale_or_low_confidence_route", "context_attention_only")
            .put("directional_instruction_allowed", false)
            .put("intervention_upgrade_allowed", false))
        .put("training_isolation", JSONObject()
            .put("future_video_teacher_allowed_in_eval_or_runtime", false))

    private fun sample(
        timestampMs: Long,
        validUntilMs: Long,
        confidence: Double,
        routeValid: Boolean,
        xValues: List<Double>
    ) = JSONObject()
        .put("timestamp_ms", timestampMs)
        .put("valid_until_timestamp_ms", validUntilMs)
        .put("confidence", confidence)
        .put("route_valid", routeValid)
        .put("horizon_waypoints", org.json.JSONArray().also { array ->
            xValues.forEachIndexed { index, x ->
                array.put(JSONObject()
                    .put("horizon_ms", (index + 1) * 1_000L)
                    .put("xy_norm", org.json.JSONArray().put(x).put(listOf(0.90, 0.82, 0.74)[index])))
            }
        })

    private fun detection(label: String, box: BoundingBox) = Detection(
        classId = 0,
        label = label,
        confidence = 0.9f,
        boundingBox = box,
        frameSize = frameSize
    )
}
