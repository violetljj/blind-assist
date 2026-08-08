package com.linnan.blindassist.benchmark

import com.linnan.blindassist.model.Detection
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Benchmark-only causal feature producer for the Corridor-Causal experiment.
 *
 * This class deliberately consumes only detector boxes already available from YOLO and an
 * externally supplied, pre-integrated IMU sample. It does not read camera pixels, run optical
 * flow, infer a route, emit a risk score, or call the application's alert chain. Its output is
 * an experiment contract, not a production feature contract.
 */
internal class YoloImuCausalFeatureExtractor {
    private var previous: SelectedDetection? = null
    private var previousTimestampNanos: Long? = null

    fun reset() {
        previous = null
        previousTimestampNanos = null
    }

    fun extract(input: YoloImuFeatureFrame): YoloImuCausalFeatures {
        require(input.timestampNanos >= 0L) { "timestampNanos must be non-negative" }
        previousTimestampNanos?.let { prior ->
            require(input.timestampNanos > prior) { "timestamps must be strictly increasing; call reset for a new sequence" }
        }

        val normalized = input.detections.mapNotNull(::normalize)
        val grid = FloatArray(GRID_WIDTH * GRID_HEIGHT * GRID_CHANNELS)
        normalized.forEach { detection -> writeDetection(grid, detection) }

        val selected = normalized.maxByOrNull { it.selectionScore }
        selected?.let { writeSelectedDetection(grid, it) }
        val motion = FloatArray(MOTION_CHANNELS)
        val imu = input.imu.finiteOrUnavailable()
        motion[0] = imu.translationX
        motion[1] = imu.translationY
        motion[2] = imu.logScale
        motion[3] = imu.rotationRadians
        motion[4] = if (imu.observed) 1f else 0f
        motion[5] = min(1f, normalized.size / MAX_DETECTIONS_NORMALIZER)
        motion[6] = normalized.maxOfOrNull { it.confidence } ?: 0f
        motion[14] = min(1f, normalized.sumOf { it.corridorOverlap.toDouble() }.toFloat() / MAX_DETECTIONS_NORMALIZER)
        motion[15] = normalized.map { it.corridorOverlap }.average().toFloat().takeIf { it.isFinite() } ?: 0f
        selected?.let { current ->
            motion[7] = current.area
            motion[8] = current.corridorOverlap
            motion[9] = current.centerX - 0.5f
            motion[12] = current.centerY
            motion[13] = current.bottom
            motion[16] = current.selectionScore
            previous?.takeIf { it.classId == current.classId && iou(it, current) >= SAME_OBJECT_IOU }?.let { prior ->
                motion[10] = current.centerX - prior.centerX
                motion[11] = ln((current.area + AREA_EPSILON) / (prior.area + AREA_EPSILON))
                motion[17] = 1f
            }
        }
        previousTimestampNanos?.let { prior ->
            motion[18] = min(MAX_DELTA_SECONDS, (input.timestampNanos - prior) / NANOS_PER_SECOND)
        }
        motion[19] = min(1f, kotlin.math.sqrt(imu.translationX * imu.translationX + imu.translationY * imu.translationY))

        previous = selected
        previousTimestampNanos = input.timestampNanos
        return YoloImuCausalFeatures(grid, motion, selected != null)
    }

    private fun normalize(detection: Detection): SelectedDetection? {
        val frame = detection.frameSize
        if (frame.width <= 0 || frame.height <= 0 || !detection.confidence.isFinite()) return null
        val box = detection.boundingBox
        if (!box.left.isFinite() || !box.top.isFinite() || !box.right.isFinite() || !box.bottom.isFinite()) return null
        val left = clamp01(box.left / frame.width)
        val top = clamp01(box.top / frame.height)
        val right = clamp01(box.right / frame.width)
        val bottom = clamp01(box.bottom / frame.height)
        if (right <= left || bottom <= top) return null
        val confidence = clamp01(detection.confidence)
        val area = (right - left) * (bottom - top)
        val corridorOverlap = corridorOverlap(left, top, right, bottom)
        return SelectedDetection(
            classId = detection.classId,
            confidence = confidence,
            left = left,
            top = top,
            right = right,
            bottom = bottom,
            area = area,
            corridorOverlap = corridorOverlap,
            selectionScore = confidence * area * (0.2f + 0.8f * corridorOverlap)
        )
    }

    private fun writeDetection(grid: FloatArray, detection: SelectedDetection) {
        val startColumn = (detection.left * GRID_WIDTH).toInt().coerceIn(0, GRID_WIDTH - 1)
        val endColumn = (((detection.right * GRID_WIDTH).toInt()).coerceAtMost(GRID_WIDTH - 1)).coerceAtLeast(startColumn)
        val startRow = (detection.top * GRID_HEIGHT).toInt().coerceIn(0, GRID_HEIGHT - 1)
        val endRow = (((detection.bottom * GRID_HEIGHT).toInt()).coerceAtMost(GRID_HEIGHT - 1)).coerceAtLeast(startRow)
        for (row in startRow..endRow) {
            for (column in startColumn..endColumn) {
                val cellLeft = column.toFloat() / GRID_WIDTH
                val cellTop = row.toFloat() / GRID_HEIGHT
                val cellRight = (column + 1).toFloat() / GRID_WIDTH
                val cellBottom = (row + 1).toFloat() / GRID_HEIGHT
                val coverage = intersectionArea(detection.left, detection.top, detection.right, detection.bottom, cellLeft, cellTop, cellRight, cellBottom) /
                    (1f / (GRID_WIDTH * GRID_HEIGHT))
                if (coverage <= 0f) continue
                val index = (row * GRID_WIDTH + column) * GRID_CHANNELS
                grid[index] = max(grid[index], detection.confidence * coverage)
                grid[index + 1] = min(1f, grid[index + 1] + detection.confidence * detection.area * coverage)
                grid[index + 2] = max(grid[index + 2], detection.confidence * detection.corridorOverlap * coverage)
                grid[index + 3] = max(grid[index + 3], detection.confidence * detection.bottom * coverage)
                grid[index + 5] = max(grid[index + 5], corridorPriorForCell(column, row))
                grid[index + 6] = min(1f, grid[index + 6] + coverage)
                grid[index + 7] = max(grid[index + 7], detection.confidence * kotlin.math.abs(detection.centerX - 0.5f) * 2f * coverage)
            }
        }
    }

    private fun writeSelectedDetection(grid: FloatArray, selected: SelectedDetection) {
        for (row in 0 until GRID_HEIGHT) {
            for (column in 0 until GRID_WIDTH) {
                val cellLeft = column.toFloat() / GRID_WIDTH
                val cellTop = row.toFloat() / GRID_HEIGHT
                val cellRight = (column + 1).toFloat() / GRID_WIDTH
                val cellBottom = (row + 1).toFloat() / GRID_HEIGHT
                val coverage = intersectionArea(selected.left, selected.top, selected.right, selected.bottom, cellLeft, cellTop, cellRight, cellBottom) /
                    (1f / (GRID_WIDTH * GRID_HEIGHT))
                if (coverage > 0f) grid[(row * GRID_WIDTH + column) * GRID_CHANNELS + 4] = selected.confidence * coverage
            }
        }
    }

    private fun corridorPriorForCell(column: Int, row: Int): Float {
        val y = (row + 0.5f) / GRID_HEIGHT
        if (y < 0.35f) return 0f
        val halfWidth = 0.12f + ((y - 0.35f) / 0.65f) * 0.30f
        val x = (column + 0.5f) / GRID_WIDTH
        return if (kotlin.math.abs(x - 0.5f) <= halfWidth) 1f else 0f
    }

    /** Intersection fraction of a normalized box with the fixed near-field corridor trapezoid. */
    private fun corridorOverlap(left: Float, top: Float, right: Float, bottom: Float): Float {
        val boxArea = (right - left) * (bottom - top)
        if (boxArea <= 0f) return 0f
        var polygon = listOf(
            Point(left, top), Point(right, top), Point(right, bottom), Point(left, bottom)
        )
        val corridor = listOf(
            Point(0.38f, 0.35f), Point(0.62f, 0.35f), Point(0.92f, 1f), Point(0.08f, 1f)
        )
        for (edgeIndex in corridor.indices) {
            val edgeStart = corridor[edgeIndex]
            val edgeEnd = corridor[(edgeIndex + 1) % corridor.size]
            polygon = clipToLeftOfEdge(polygon, edgeStart, edgeEnd)
            if (polygon.isEmpty()) return 0f
        }
        return clamp01(polygonArea(polygon) / boxArea)
    }

    private fun clipToLeftOfEdge(subject: List<Point>, edgeStart: Point, edgeEnd: Point): List<Point> {
        if (subject.isEmpty()) return emptyList()
        val output = mutableListOf<Point>()
        var previous = subject.last()
        var previousInside = cross(edgeStart, edgeEnd, previous) >= 0f
        subject.forEach { current ->
            val currentInside = cross(edgeStart, edgeEnd, current) >= 0f
            if (currentInside != previousInside) output += lineIntersection(previous, current, edgeStart, edgeEnd)
            if (currentInside) output += current
            previous = current
            previousInside = currentInside
        }
        return output
    }

    private fun cross(edgeStart: Point, edgeEnd: Point, point: Point): Float =
        (edgeEnd.x - edgeStart.x) * (point.y - edgeStart.y) -
            (edgeEnd.y - edgeStart.y) * (point.x - edgeStart.x)

    private fun lineIntersection(a: Point, b: Point, c: Point, d: Point): Point {
        val denominator = (a.x - b.x) * (c.y - d.y) - (a.y - b.y) * (c.x - d.x)
        if (kotlin.math.abs(denominator) < 1e-7f) return b
        val determinantA = a.x * b.y - a.y * b.x
        val determinantB = c.x * d.y - c.y * d.x
        return Point(
            (determinantA * (c.x - d.x) - (a.x - b.x) * determinantB) / denominator,
            (determinantA * (c.y - d.y) - (a.y - b.y) * determinantB) / denominator
        )
    }

    private fun polygonArea(points: List<Point>): Float {
        var sum = 0f
        for (index in points.indices) {
            val next = points[(index + 1) % points.size]
            sum += points[index].x * next.y - points[index].y * next.x
        }
        return kotlin.math.abs(sum) * 0.5f
    }

    private fun iou(a: SelectedDetection, b: SelectedDetection): Float {
        val intersection = intersectionArea(a.left, a.top, a.right, a.bottom, b.left, b.top, b.right, b.bottom)
        return intersection / max(AREA_EPSILON, a.area + b.area - intersection)
    }

    private fun intersectionArea(
        leftA: Float, topA: Float, rightA: Float, bottomA: Float,
        leftB: Float, topB: Float, rightB: Float, bottomB: Float
    ): Float = max(0f, min(rightA, rightB) - max(leftA, leftB)) * max(0f, min(bottomA, bottomB) - max(topA, topB))

    private fun clamp01(value: Float): Float = value.coerceIn(0f, 1f)

    private data class Point(val x: Float, val y: Float)

    private data class SelectedDetection(
        val classId: Int,
        val confidence: Float,
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float,
        val area: Float,
        val corridorOverlap: Float,
        val selectionScore: Float
    ) {
        val centerX: Float get() = (left + right) * 0.5f
        val centerY: Float get() = (top + bottom) * 0.5f
    }

    companion object {
        const val GRID_WIDTH = 4
        const val GRID_HEIGHT = 4
        const val GRID_CHANNELS = 8
        const val MOTION_CHANNELS = 20
        private const val MAX_DETECTIONS_NORMALIZER = 10f
        private const val SAME_OBJECT_IOU = 0.3f
        private const val AREA_EPSILON = 1e-6f
        private const val NANOS_PER_SECOND = 1_000_000_000f
        private const val MAX_DELTA_SECONDS = 1f
    }
}

internal data class YoloImuFeatureFrame(
    val timestampNanos: Long,
    val detections: List<Detection>,
    val imu: YoloImuMotion = YoloImuMotion.unavailable()
)

/**
 * Pre-integrated camera-relative motion; sensor fusion and timestamp alignment remain outside
 * this probe. It is auxiliary compensation evidence only: it must not be interpreted as a
 * heading, route choice, turn label, or direct alert rule.
 */
internal data class YoloImuMotion(
    val translationX: Float,
    val translationY: Float,
    val logScale: Float,
    val rotationRadians: Float,
    val observed: Boolean
) {
    internal fun finiteOrUnavailable(): YoloImuMotion =
        if (translationX.isFinite() && translationY.isFinite() && logScale.isFinite() && rotationRadians.isFinite()) this
        else unavailable()

    companion object {
        fun unavailable() = YoloImuMotion(0f, 0f, 0f, 0f, observed = false)
    }
}

/** A feature-only result; it intentionally carries neither a label nor an alert decision. */
internal data class YoloImuCausalFeatures(
    val spatialGridNhwc: FloatArray,
    val motion: FloatArray,
    val hasSelectedDetection: Boolean
)

/**
 * Test-only causal window packer. It retains no images or detections after conversion and emits
 * a window only once eight chronological feature points are available.
 */
internal class YoloImuCausalWindow(private val contextFrames: Int = 8) {
    private val extractor = YoloImuCausalFeatureExtractor()
    private val frames = java.util.ArrayDeque<YoloImuCausalFeatures>()

    init {
        require(contextFrames == 8) { "the frozen event-head accepts exactly 8 causal frames" }
    }

    fun reset() {
        extractor.reset()
        frames.clear()
    }

    fun append(input: YoloImuFeatureFrame): YoloImuCausalSequence? {
        frames += extractor.extract(input)
        if (frames.size > contextFrames) frames.removeFirst()
        if (frames.size < contextFrames) return null
        val spatial = FloatArray(contextFrames * GRID_VALUES_PER_FRAME)
        val motion = FloatArray(contextFrames * MOTION_VALUES_PER_FRAME)
        frames.forEachIndexed { index, features ->
            features.spatialGridNhwc.copyInto(spatial, index * GRID_VALUES_PER_FRAME)
            features.motion.copyInto(motion, index * MOTION_VALUES_PER_FRAME)
        }
        return YoloImuCausalSequence(spatial, motion)
    }

    companion object {
        const val GRID_VALUES_PER_FRAME = YoloImuCausalFeatureExtractor.GRID_WIDTH *
            YoloImuCausalFeatureExtractor.GRID_HEIGHT * YoloImuCausalFeatureExtractor.GRID_CHANNELS
        const val MOTION_VALUES_PER_FRAME = YoloImuCausalFeatureExtractor.MOTION_CHANNELS
    }
}

/** Flattened `[8, 4, 4, 8]` and `[8, 20]` tensors in chronological causal order. */
internal data class YoloImuCausalSequence(
    val spatialSequence: FloatArray,
    val motionSequence: FloatArray
)
