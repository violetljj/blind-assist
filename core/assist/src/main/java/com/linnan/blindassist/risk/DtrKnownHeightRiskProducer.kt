package com.linnan.blindassist.risk

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.CameraIntrinsics
import com.linnan.blindassist.vision.FrameStamp
import kotlin.math.hypot

/** Event outputs for the fixed DTR route-intersection runtime. */
enum class DtrSignal {
    ONSET,
    HOLD,
    CLEAR,
    UNKNOWN
}

data class DtrPrediction(
    val signal: DtrSignal,
    val rawAlert: Boolean?,
    val eventActive: Boolean,
    val eventKey: String,
    val reason: String,
    val trackId: String? = null,
    val minimumSeparationM: Float? = null,
    val intersectionThresholdM: Float? = null,
    val futureS: Float? = null,
    val personDetectionCount: Int = 0,
    val metricTrackCount: Int = 0
)

/**
 * Phone-camera DTR source selected by the public-data ceiling.
 *
 * A fixed 1.70 m upright-person prior plus camera intrinsics converts each
 * current RGB person box into wearer-relative metric position. A causal
 * 1.5-second constant-velocity fit asks whether the relative trajectory enters
 * the 0.65 m route half-width plus a 0.30 m person radius within three seconds.
 *
 * The camera must stay aligned with the wearer's short route. Relative motion
 * already includes wearer translation, so this runtime does not subtract a
 * second nominal wearer velocity. Missing calibration and short tracks are
 * UNKNOWN; they never clear an active event.
 */
class DtrKnownHeightRiskProducer {
    private data class MetricSample(
        val capturedAtNs: Long,
        val forwardM: Double,
        val leftM: Double
    )

    private data class Track(
        val id: String,
        var box: BoundingBox,
        var lastSeenNs: Long,
        val samples: ArrayDeque<MetricSample> = ArrayDeque()
    )

    private data class Estimate(
        val trackId: String,
        val forwardM: Double,
        val leftM: Double,
        val forwardVelocityMps: Double,
        val leftVelocityMps: Double
    )

    private data class Candidate(
        val track: Track,
        val detectionIndex: Int,
        val iou: Float,
        val centerDistance: Float
    )

    private data class LifecycleResult(
        val signal: DtrSignal,
        val active: Boolean,
        val eventKey: String
    )

    private val tracks = linkedMapOf<String, Track>()
    private var nextTrackId = 1L
    private var lastFrame: FrameStamp? = null
    private var eventActive = false
    private var clearCandidateSinceNs: Long? = null
    private var eventEpoch = 0L

    fun reset() {
        tracks.clear()
        nextTrackId = 1L
        lastFrame = null
        eventActive = false
        clearCandidateSinceNs = null
        eventEpoch = 0L
    }

    fun process(
        detections: List<Detection>,
        frameSize: FrameSize,
        cameraIntrinsics: CameraIntrinsics?,
        sourceFrame: FrameStamp?
    ): DtrPrediction {
        if (sourceFrame == null) {
            return predictionFor(
                lifecycle = updateLifecycle(null, null),
                rawAlert = null,
                reason = "missing_frame_timestamp",
                personDetectionCount = detections.count(::isPerson)
            )
        }
        val previous = lastFrame
        if (previous != null && !continues(previous, sourceFrame)) {
            tracks.clear()
            lastFrame = null
        }
        lastFrame = sourceFrame

        val people = detections.filter(::isPerson)
        val intrinsics = cameraIntrinsics
        if (intrinsics == null ||
            intrinsics.coordinateWidthPx != frameSize.width ||
            intrinsics.coordinateHeightPx != frameSize.height
        ) {
            tracks.clear()
            return predictionFor(
                lifecycle = updateLifecycle(sourceFrame.capturedAtNs, null),
                rawAlert = null,
                reason = "missing_display_camera_intrinsics",
                personDetectionCount = people.size
            )
        }

        removeStaleTracks(sourceFrame.capturedAtNs)
        if (people.isEmpty()) {
            val lifecycle = updateLifecycle(sourceFrame.capturedAtNs, false)
            return predictionFor(
                lifecycle = lifecycle,
                rawAlert = false,
                reason = "no_current_person_detection",
                personDetectionCount = 0
            )
        }

        val currentTracks = associate(people, frameSize, sourceFrame.capturedAtNs)
        val estimates = mutableListOf<Estimate>()
        var metricTrackCount = 0
        currentTracks.forEach { (track, detection) ->
            val metric = project(detection.boundingBox, intrinsics) ?: return@forEach
            metricTrackCount += 1
            track.samples.addLast(
                MetricSample(
                    capturedAtNs = sourceFrame.capturedAtNs,
                    forwardM = metric.first,
                    leftM = metric.second
                )
            )
            val cutoffNs = sourceFrame.capturedAtNs - TRACK_WINDOW_NS
            while (track.samples.firstOrNull()?.capturedAtNs?.let { it < cutoffNs } == true) {
                track.samples.removeFirst()
            }
            fit(track)?.let(estimates::add)
        }

        if (estimates.isEmpty()) {
            return predictionFor(
                lifecycle = updateLifecycle(sourceFrame.capturedAtNs, null),
                rawAlert = null,
                reason = "insufficient_causal_metric_track",
                personDetectionCount = people.size,
                metricTrackCount = metricTrackCount
            )
        }

        val best = estimates.map { estimate ->
            val (distance, futureS) = minimumSeparation(estimate)
            Triple(estimate, distance, futureS)
        }.minBy { it.second }
        val thresholdM = ROUTE_HALF_WIDTH_M + PERSON_RADIUS_M
        val rawAlert = best.second <= thresholdM
        return predictionFor(
            lifecycle = updateLifecycle(sourceFrame.capturedAtNs, rawAlert),
            rawAlert = rawAlert,
            reason = "route_tube_intersection_evaluated",
            trackId = best.first.trackId,
            minimumSeparationM = best.second.toFloat(),
            intersectionThresholdM = thresholdM.toFloat(),
            futureS = best.third.toFloat(),
            personDetectionCount = people.size,
            metricTrackCount = metricTrackCount
        )
    }

    private fun continues(previous: FrameStamp, current: FrameStamp): Boolean =
        current.sourceId == previous.sourceId &&
            current.coordinateFrame == previous.coordinateFrame &&
            current.clockDomain == previous.clockDomain &&
            current.frameId > previous.frameId &&
            current.capturedAtNs > previous.capturedAtNs

    private fun isPerson(detection: Detection): Boolean =
        detection.source == DetectionSource.OBJECT_DETECTOR &&
            detection.label.equals(PERSON_LABEL, ignoreCase = true)

    private fun removeStaleTracks(nowNs: Long) {
        val iterator = tracks.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (nowNs - entry.value.lastSeenNs > MAXIMUM_TRACK_GAP_NS) iterator.remove()
        }
    }

    private fun associate(
        detections: List<Detection>,
        frameSize: FrameSize,
        capturedAtNs: Long
    ): List<Pair<Track, Detection>> {
        val candidates = mutableListOf<Candidate>()
        tracks.values.forEach { track ->
            detections.forEachIndexed { index, detection ->
                val overlap = iou(track.box, detection.boundingBox)
                val centerDistance = normalizedCenterDistance(track.box, detection.boundingBox, frameSize)
                if (overlap >= MINIMUM_IOU || centerDistance <= MAXIMUM_CENTER_DISTANCE) {
                    candidates += Candidate(track, index, overlap, centerDistance)
                }
            }
        }
        candidates.sortWith(
            compareByDescending<Candidate> { it.iou }
                .thenBy { it.centerDistance }
                .thenBy { it.track.id }
                .thenBy { it.detectionIndex }
        )
        val usedTracks = mutableSetOf<String>()
        val usedDetections = mutableSetOf<Int>()
        val matched = mutableListOf<Pair<Track, Detection>>()
        candidates.forEach { candidate ->
            if (!usedTracks.add(candidate.track.id) ||
                !usedDetections.add(candidate.detectionIndex)
            ) return@forEach
            val detection = detections[candidate.detectionIndex]
            candidate.track.box = detection.boundingBox
            candidate.track.lastSeenNs = capturedAtNs
            matched += candidate.track to detection
        }
        detections.forEachIndexed { index, detection ->
            if (index in usedDetections) return@forEachIndexed
            val track = Track(
                id = "rgb-${nextTrackId++}",
                box = detection.boundingBox,
                lastSeenNs = capturedAtNs
            )
            tracks[track.id] = track
            matched += track to detection
        }
        return matched
    }

    private fun project(
        box: BoundingBox,
        intrinsics: CameraIntrinsics
    ): Pair<Double, Double>? {
        val heightPx = box.height.toDouble()
        if (!heightPx.isFinite() || heightPx <= 0.0) return null
        val forwardM = intrinsics.focalLengthYPx.toDouble() * PERSON_HEIGHT_M / heightPx
        val horizontalRay = (
            box.centerX.toDouble() - intrinsics.principalPointXPx.toDouble()
            ) / intrinsics.focalLengthXPx.toDouble()
        val leftM = -horizontalRay * forwardM
        if (!forwardM.isFinite() || !leftM.isFinite()) return null
        return forwardM to leftM
    }

    private fun fit(track: Track): Estimate? {
        val samples = track.samples.toList()
        if (samples.size < 2) return null
        val firstNs = samples.first().capturedAtNs
        val spanNs = samples.last().capturedAtNs - firstNs
        if (spanNs < MINIMUM_TRACK_SPAN_NS) return null
        val times = samples.map { (it.capturedAtNs - firstNs).toDouble() / NANOS_PER_SECOND }
        val meanTime = times.average()
        val denominator = times.sumOf { (it - meanTime) * (it - meanTime) }
        if (denominator <= EPSILON) return null
        val meanForward = samples.map { it.forwardM }.average()
        val meanLeft = samples.map { it.leftM }.average()
        val forwardVelocity = samples.indices.sumOf { index ->
            (times[index] - meanTime) * (samples[index].forwardM - meanForward)
        } / denominator
        val leftVelocity = samples.indices.sumOf { index ->
            (times[index] - meanTime) * (samples[index].leftM - meanLeft)
        } / denominator
        val currentTime = times.last()
        return Estimate(
            trackId = track.id,
            forwardM = meanForward + forwardVelocity * (currentTime - meanTime),
            leftM = meanLeft + leftVelocity * (currentTime - meanTime),
            forwardVelocityMps = forwardVelocity,
            leftVelocityMps = leftVelocity
        )
    }

    private fun minimumSeparation(estimate: Estimate): Pair<Double, Double> {
        val speedSquared = estimate.forwardVelocityMps * estimate.forwardVelocityMps +
            estimate.leftVelocityMps * estimate.leftVelocityMps
        val futureS = if (speedSquared <= EPSILON) {
            0.0
        } else {
            val unconstrained = -(
                estimate.forwardM * estimate.forwardVelocityMps +
                    estimate.leftM * estimate.leftVelocityMps
                ) / speedSquared
            unconstrained.coerceIn(0.0, ROUTE_HORIZON_S)
        }
        val forward = estimate.forwardM + estimate.forwardVelocityMps * futureS
        val left = estimate.leftM + estimate.leftVelocityMps * futureS
        return hypot(forward, left) to futureS
    }

    private fun updateLifecycle(timeNs: Long?, rawAlert: Boolean?): LifecycleResult {
        if (rawAlert == null || timeNs == null) {
            clearCandidateSinceNs = null
            return LifecycleResult(DtrSignal.UNKNOWN, eventActive, currentEventKey())
        }
        if (rawAlert) {
            clearCandidateSinceNs = null
            if (eventActive) {
                return LifecycleResult(DtrSignal.HOLD, true, currentEventKey())
            }
            eventActive = true
            eventEpoch += 1L
            return LifecycleResult(DtrSignal.ONSET, true, currentEventKey())
        }
        if (!eventActive) {
            clearCandidateSinceNs = null
            return LifecycleResult(DtrSignal.CLEAR, false, currentEventKey())
        }
        val candidateSince = clearCandidateSinceNs
        if (candidateSince == null) {
            clearCandidateSinceNs = timeNs
            return LifecycleResult(DtrSignal.HOLD, true, currentEventKey())
        }
        if (timeNs - candidateSince < CLEAR_GRACE_NS) {
            return LifecycleResult(DtrSignal.HOLD, true, currentEventKey())
        }
        eventActive = false
        clearCandidateSinceNs = null
        return LifecycleResult(DtrSignal.CLEAR, false, currentEventKey())
    }

    private fun currentEventKey(): String =
        if (eventEpoch == 0L) "dtr-route-pending" else "dtr-route-$eventEpoch"

    private fun predictionFor(
        lifecycle: LifecycleResult,
        rawAlert: Boolean?,
        reason: String,
        trackId: String? = null,
        minimumSeparationM: Float? = null,
        intersectionThresholdM: Float? = null,
        futureS: Float? = null,
        personDetectionCount: Int = 0,
        metricTrackCount: Int = 0
    ) = DtrPrediction(
        signal = lifecycle.signal,
        rawAlert = rawAlert,
        eventActive = lifecycle.active,
        eventKey = lifecycle.eventKey,
        reason = reason,
        trackId = trackId,
        minimumSeparationM = minimumSeparationM,
        intersectionThresholdM = intersectionThresholdM,
        futureS = futureS,
        personDetectionCount = personDetectionCount,
        metricTrackCount = metricTrackCount
    )

    private fun iou(first: BoundingBox, second: BoundingBox): Float {
        val left = maxOf(first.left, second.left)
        val top = maxOf(first.top, second.top)
        val right = minOf(first.right, second.right)
        val bottom = minOf(first.bottom, second.bottom)
        val intersection = maxOf(0f, right - left) * maxOf(0f, bottom - top)
        val union = first.width * first.height + second.width * second.height - intersection
        return if (union > 0f) intersection / union else 0f
    }

    private fun normalizedCenterDistance(
        first: BoundingBox,
        second: BoundingBox,
        frameSize: FrameSize
    ): Float {
        val dx = (first.centerX - second.centerX) / frameSize.width.coerceAtLeast(1)
        val dy = (first.centerY - second.centerY) / frameSize.height.coerceAtLeast(1)
        return hypot(dx.toDouble(), dy.toDouble()).toFloat()
    }

    private companion object {
        const val PERSON_LABEL = "person"
        const val PERSON_HEIGHT_M = 1.70
        const val PERSON_RADIUS_M = 0.30
        const val ROUTE_HALF_WIDTH_M = 0.65
        const val ROUTE_HORIZON_S = 3.0
        const val TRACK_WINDOW_NS = 1_500_000_000L
        const val MINIMUM_TRACK_SPAN_NS = 200_000_000L
        const val MAXIMUM_TRACK_GAP_NS = 500_000_000L
        const val CLEAR_GRACE_NS = 500_000_000L
        const val MINIMUM_IOU = 0.25f
        const val MAXIMUM_CENTER_DISTANCE = 0.12f
        const val NANOS_PER_SECOND = 1_000_000_000.0
        const val EPSILON = 1e-9
    }
}
