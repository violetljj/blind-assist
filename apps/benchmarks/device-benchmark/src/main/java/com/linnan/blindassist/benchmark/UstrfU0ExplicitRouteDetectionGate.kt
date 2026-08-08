package com.linnan.blindassist.benchmark

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

internal data class UstrfU0RouteGateConfig(
    val minimumRouteConfidence: Double = 0.5,
    val maximumRouteAgeMs: Long = 1_000L,
    val corridorHalfWidthFrameRatio: Double = 0.08,
    val obstacleFootprintHeightRatio: Double = 0.25
) {
    init {
        require(minimumRouteConfidence in 0.0..1.0)
        require(maximumRouteAgeMs in 0L..1_000L)
        require(corridorHalfWidthFrameRatio in 0.0..0.5)
        require(obstacleFootprintHeightRatio in 0.0..1.0)
    }
}

internal data class UstrfU0RouteWaypoint(
    val horizonMs: Long,
    val xNorm: Double,
    val yNorm: Double
)

internal data class UstrfU0RouteSample(
    val sampleIndex: Int,
    val timestampMs: Long,
    val validUntilTimestampMs: Long,
    val confidence: Double,
    val routeValid: Boolean,
    val waypoints: List<UstrfU0RouteWaypoint>
)

internal data class UstrfU0ExplicitRouteEpisode(
    val episodeId: String,
    val parentSourceId: String,
    val providerType: String,
    val providerId: String,
    val projectionReceiptId: String,
    val samples: List<UstrfU0RouteSample>
)

internal data class UstrfU0DetectionGateRow(
    val detectionIndex: Int,
    val label: String,
    val confidence: Float,
    val sourceBox: BoundingBox,
    val footprintBox: BoundingBox,
    val minimumRouteDistancePx: Double,
    val corridorHalfWidthPx: Double,
    val kept: Boolean
)

internal data class UstrfU0RouteGateResult(
    val retainedDetections: List<Detection>,
    val routeUsable: Boolean,
    val reason: String,
    val selectedSample: UstrfU0RouteSample?,
    val rows: List<UstrfU0DetectionGateRow>
)

/**
 * Fixed, benchmark-only bbox × explicit-route gate for the U0 detector-route arm.
 *
 * The route is external to the risk model. For each decision frame, only the latest
 * sample generated at or before the frame can be selected. A route corridor starts
 * at the current camera bottom centre and follows the exact 1/2/3-second waypoints.
 * Only detector boxes whose bottom footprint reaches that corridor are passed, without
 * score or geometry modification, into the shared AssistDecisionKernel.
 */
internal object UstrfU0ExplicitRouteDetectionGate {
    const val ROUTE_SCHEMA = "blindassist_explicit_route_intent_episode_v1"
    const val COORDINATE_SPACE = "normalized_current_camera_frame_xy"
    const val GATE_CONTRACT_ID = "bbox_bottom_footprint_polyline_corridor_v1"
    const val UNKNOWN_ROUTE_POLICY = "context_attention_only_empty_detection_gate_v1"
    val REQUIRED_HORIZONS_MS = listOf(1_000L, 2_000L, 3_000L)
    private val runtimeProviderTypes = setOf(
        "navigation",
        "explicit_user_choice",
        "navigation_or_explicit_user_choice"
    )

    fun parseEpisode(value: JSONObject, expectedEpisodeId: String): UstrfU0ExplicitRouteEpisode {
        check(value.getString("schema") == ROUTE_SCHEMA) { "explicit route schema mismatch" }
        check(value.getString("episode_id") == expectedEpisodeId) { "explicit route episode mismatch" }
        val parentSourceId = nonBlank(value.getString("parent_source_id"), "parent_source_id")
        val provider = value.getJSONObject("provider")
        val providerType = nonBlank(provider.getString("type"), "provider.type")
        check(providerType in runtimeProviderTypes) { "explicit route provider type is not runtime-authorized" }
        val providerId = nonBlank(provider.getString("provider_id"), "provider.provider_id")
        check(!provider.getBoolean("inferred_by_risk_model")) { "route must be external to the risk model" }
        val coordinate = value.getJSONObject("coordinate_contract")
        check(coordinate.getString("space") == COORDINATE_SPACE) { "route coordinate space mismatch" }
        val projectionReceiptId = nonBlank(
            coordinate.getString("projection_receipt_id"),
            "coordinate_contract.projection_receipt_id"
        )
        val fallback = value.getJSONObject("fallback")
        check(fallback.getString("missing_stale_or_low_confidence_route") == "context_attention_only")
        check(!fallback.getBoolean("directional_instruction_allowed"))
        check(!fallback.getBoolean("intervention_upgrade_allowed"))
        val isolation = value.getJSONObject("training_isolation")
        check(!isolation.getBoolean("future_video_teacher_allowed_in_eval_or_runtime"))

        val values = value.getJSONArray("samples")
        check(values.length() > 0) { "explicit route samples must be non-empty" }
        var previousTimestamp: Long? = null
        val samples = (0 until values.length()).map { index ->
            val row = values.getJSONObject(index)
            val timestamp = row.getLong("timestamp_ms")
            val validUntil = row.getLong("valid_until_timestamp_ms")
            check(previousTimestamp == null || timestamp > previousTimestamp!!) {
                "explicit route timestamps must be strictly increasing"
            }
            previousTimestamp = timestamp
            check(validUntil >= timestamp && validUntil - timestamp <= 1_000L) {
                "explicit route validity exceeds one second"
            }
            val confidence = row.getDouble("confidence")
            check(confidence.isFinite() && confidence in 0.0..1.0) { "route confidence is invalid" }
            val routeValid = row.getBoolean("route_valid")
            val waypointRows = row.getJSONArray("horizon_waypoints")
            val waypoints = (0 until waypointRows.length()).map { waypointIndex ->
                val waypoint = waypointRows.getJSONObject(waypointIndex)
                val xy = waypoint.getJSONArray("xy_norm")
                check(xy.length() == 2) { "route waypoint must contain x/y" }
                UstrfU0RouteWaypoint(
                    horizonMs = waypoint.getLong("horizon_ms"),
                    xNorm = xy.getDouble(0),
                    yNorm = xy.getDouble(1)
                )
            }
            if (!routeValid) check(waypoints.isEmpty()) { "invalid route sample contains waypoints" }
            UstrfU0RouteSample(index, timestamp, validUntil, confidence, routeValid, waypoints)
        }
        return UstrfU0ExplicitRouteEpisode(
            expectedEpisodeId,
            parentSourceId,
            providerType,
            providerId,
            projectionReceiptId,
            samples
        )
    }

    fun gate(
        episode: UstrfU0ExplicitRouteEpisode,
        frameTimestampMs: Long,
        detections: List<Detection>,
        frameSize: FrameSize,
        config: UstrfU0RouteGateConfig = UstrfU0RouteGateConfig()
    ): UstrfU0RouteGateResult {
        require(frameSize.width > 0 && frameSize.height > 0)
        val selected = episode.samples.lastOrNull { it.timestampMs <= frameTimestampMs }
            ?: return closed("NO_CAUSAL_ROUTE_SAMPLE")
        if (!selected.routeValid) return closed("ROUTE_MARKED_INVALID", selected)
        if (frameTimestampMs > selected.validUntilTimestampMs) return closed("ROUTE_STALE", selected)
        if (frameTimestampMs - selected.timestampMs > config.maximumRouteAgeMs) {
            return closed("ROUTE_TOO_OLD", selected)
        }
        if (selected.confidence < config.minimumRouteConfidence) return closed("ROUTE_LOW_CONFIDENCE", selected)
        if (!validWaypoints(selected.waypoints)) return closed("ROUTE_WAYPOINT_CONTRACT_INVALID", selected)

        val routePoints = listOf(Point(frameSize.width / 2.0, frameSize.height.toDouble())) +
            selected.waypoints.map { Point(it.xNorm * frameSize.width, it.yNorm * frameSize.height) }
        val corridorHalfWidthPx = config.corridorHalfWidthFrameRatio * frameSize.width
        val rows = detections.mapIndexed { index, detection ->
            check(detection.frameSize == frameSize) { "detection frame size differs from decoded frame" }
            val source = detection.boundingBox.clamped(frameSize)
            check(source.width > 0f && source.height > 0f) { "detector box is empty after clamp" }
            val footprint = BoundingBox(
                source.left,
                source.bottom - source.height * config.obstacleFootprintHeightRatio.toFloat(),
                source.right,
                source.bottom
            )
            val minimumDistance = routePoints.zipWithNext().minOf { (start, end) ->
                segmentRectangleDistance(start, end, footprint)
            }
            UstrfU0DetectionGateRow(
                index,
                detection.label,
                detection.confidence,
                source,
                footprint,
                minimumDistance,
                corridorHalfWidthPx,
                minimumDistance <= corridorHalfWidthPx
            )
        }
        return UstrfU0RouteGateResult(
            retainedDetections = rows.filter { it.kept }.map { detections[it.detectionIndex] },
            routeUsable = true,
            reason = "ROUTE_USABLE",
            selectedSample = selected,
            rows = rows
        )
    }

    fun receiptJson(
        episode: UstrfU0ExplicitRouteEpisode,
        frameId: String,
        frameTimestampMs: Long,
        result: UstrfU0RouteGateResult,
        config: UstrfU0RouteGateConfig
    ): JSONObject = JSONObject()
        .put("frame_id", frameId)
        .put("frame_timestamp_ms", frameTimestampMs)
        .put("route_episode_id", episode.episodeId)
        .put("route_parent_source_id", episode.parentSourceId)
        .put("route_provider_type", episode.providerType)
        .put("route_provider_id", episode.providerId)
        .put("projection_receipt_id", episode.projectionReceiptId)
        .put("route_usable", result.routeUsable)
        .put("gate_reason", result.reason)
        .put("selected_sample_index", result.selectedSample?.sampleIndex ?: JSONObject.NULL)
        .put("selected_sample_timestamp_ms", result.selectedSample?.timestampMs ?: JSONObject.NULL)
        .put("selected_valid_until_timestamp_ms", result.selectedSample?.validUntilTimestampMs ?: JSONObject.NULL)
        .put("selected_route_confidence", result.selectedSample?.confidence ?: JSONObject.NULL)
        .put("selected_waypoints", result.selectedSample?.waypoints?.toWaypointJson() ?: JSONArray())
        .put("gate_contract_id", GATE_CONTRACT_ID)
        .put("unknown_route_policy", UNKNOWN_ROUTE_POLICY)
        .put("minimum_route_confidence", config.minimumRouteConfidence)
        .put("maximum_route_age_ms", config.maximumRouteAgeMs)
        .put("corridor_half_width_frame_ratio", config.corridorHalfWidthFrameRatio)
        .put("obstacle_footprint_height_ratio", config.obstacleFootprintHeightRatio)
        .put("input_detection_count", result.rows.size)
        .put("retained_detection_count", result.retainedDetections.size)
        .put("detections", result.rows.toDetectionJson())

    private fun validWaypoints(values: List<UstrfU0RouteWaypoint>): Boolean =
        values.size == REQUIRED_HORIZONS_MS.size &&
            values.map { it.horizonMs } == REQUIRED_HORIZONS_MS &&
            values.all { it.xNorm.isFinite() && it.yNorm.isFinite() && it.xNorm in 0.0..1.0 && it.yNorm in 0.0..1.0 }

    private fun closed(reason: String, selected: UstrfU0RouteSample? = null) = UstrfU0RouteGateResult(
        retainedDetections = emptyList(),
        routeUsable = false,
        reason = reason,
        selectedSample = selected,
        rows = emptyList()
    )

    private fun segmentRectangleDistance(start: Point, end: Point, rectangle: BoundingBox): Double {
        if (pointInRectangle(start, rectangle) || pointInRectangle(end, rectangle)) return 0.0
        val corners = listOf(
            Point(rectangle.left.toDouble(), rectangle.top.toDouble()),
            Point(rectangle.right.toDouble(), rectangle.top.toDouble()),
            Point(rectangle.right.toDouble(), rectangle.bottom.toDouble()),
            Point(rectangle.left.toDouble(), rectangle.bottom.toDouble())
        )
        if (corners.indices.any { index -> segmentsIntersect(start, end, corners[index], corners[(index + 1) % 4]) }) {
            return 0.0
        }
        return min(
            min(pointRectangleDistance(start, rectangle), pointRectangleDistance(end, rectangle)),
            corners.minOf { pointSegmentDistance(it, start, end) }
        )
    }

    private fun pointInRectangle(point: Point, rectangle: BoundingBox): Boolean =
        point.x in rectangle.left.toDouble()..rectangle.right.toDouble() &&
            point.y in rectangle.top.toDouble()..rectangle.bottom.toDouble()

    private fun pointRectangleDistance(point: Point, rectangle: BoundingBox): Double {
        val dx = max(max(rectangle.left - point.x, 0.0), point.x - rectangle.right)
        val dy = max(max(rectangle.top - point.y, 0.0), point.y - rectangle.bottom)
        return hypot(dx, dy)
    }

    private fun pointSegmentDistance(point: Point, start: Point, end: Point): Double {
        val dx = end.x - start.x
        val dy = end.y - start.y
        val lengthSquared = dx * dx + dy * dy
        if (lengthSquared == 0.0) return hypot(point.x - start.x, point.y - start.y)
        val t = (((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared).coerceIn(0.0, 1.0)
        return hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy))
    }

    private fun segmentsIntersect(a: Point, b: Point, c: Point, d: Point): Boolean {
        fun cross(p: Point, q: Point, r: Point) =
            (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)
        val abC = cross(a, b, c)
        val abD = cross(a, b, d)
        val cdA = cross(c, d, a)
        val cdB = cross(c, d, b)
        val projectionsOverlap = max(min(a.x, b.x), min(c.x, d.x)) <= min(max(a.x, b.x), max(c.x, d.x)) &&
            max(min(a.y, b.y), min(c.y, d.y)) <= min(max(a.y, b.y), max(c.y, d.y))
        return projectionsOverlap && abC * abD <= 0.0 && cdA * cdB <= 0.0
    }

    private fun nonBlank(value: String, where: String): String {
        check(value.isNotBlank() && !value.startsWith("REQUIRED_")) { "$where is not concrete" }
        return value
    }

    private fun List<UstrfU0RouteWaypoint>.toWaypointJson(): JSONArray = JSONArray().also { array ->
        forEach { waypoint ->
            array.put(
                JSONObject()
                    .put("horizon_ms", waypoint.horizonMs)
                    .put("xy_norm", JSONArray().put(waypoint.xNorm).put(waypoint.yNorm))
            )
        }
    }

    private fun List<UstrfU0DetectionGateRow>.toDetectionJson(): JSONArray = JSONArray().also { array ->
        forEach { row ->
            array.put(
                JSONObject()
                    .put("detection_index", row.detectionIndex)
                    .put("label", row.label)
                    .put("confidence", row.confidence.toDouble())
                    .put("source_box_xyxy_px", row.sourceBox.toJson())
                    .put("footprint_box_xyxy_px", row.footprintBox.toJson())
                    .put("minimum_route_distance_px", row.minimumRouteDistancePx)
                    .put("corridor_half_width_px", row.corridorHalfWidthPx)
                    .put("kept", row.kept)
            )
        }
    }

    private fun BoundingBox.toJson() = JSONArray()
        .put(left.toDouble()).put(top.toDouble()).put(right.toDouble()).put(bottom.toDouble())

    private data class Point(val x: Double, val y: Double)
}
