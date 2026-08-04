package com.linnan.blindassist.hftf.metricdepth

/**
 * Rich HFTF research output. This type intentionally lives outside the default App graph.
 * VALID means the observation contract passed; it never means that a route is safe.
 */
enum class TraversabilityFieldStatus { VALID, UNKNOWN }

enum class SweepObservationState {
    OCCUPIED_OBSERVED,
    CLEAR_OBSERVED,
    UNKNOWN_SUPPORT
}

data class MetricDepthSummary(
    val available: Boolean,
    val sourceModel: String,
    val scaleStatus: String,
    val scale: Float?,
    val anchorAgeNs: Long?,
    val anchorSource: String?,
    val finiteFraction: Float,
    val p05Meters: Float?,
    val p50Meters: Float?,
    val p95Meters: Float?
) {
    init {
        require(sourceModel.isNotBlank() && scaleStatus.isNotBlank())
        require(finiteFraction in 0f..1f)
        require(scale == null || scale.isFinite() && scale > 0f)
        require(anchorAgeNs == null || anchorAgeNs >= 0L)
        listOf(p05Meters, p50Meters, p95Meters).filterNotNull().forEach {
            require(it.isFinite() && it > 0f)
        }
        if (available) {
            require(scaleStatus == "VALID" && scale != null)
        } else {
            require(p05Meters == null && p50Meters == null && p95Meters == null) {
                "UNKNOWN metric depth cannot retain meter-valued percentiles"
            }
        }
    }
}

data class TraversabilityGroundPlane(
    val source: String,
    val normalCamera: List<Float>,
    val cameraHeightMeters: Float,
    val medianResidualMeters: Float
) {
    init {
        require(source.isNotBlank())
        require(normalCamera.size == 3 && normalCamera.all { it.isFinite() })
        require(cameraHeightMeters.isFinite() && cameraHeightMeters > 0f)
        require(medianResidualMeters.isFinite() && medianResidualMeters >= 0f)
    }
}

data class TraversabilityProvenance(
    val sourceModel: String,
    val geometry: String,
    val metricScaleSource: String
) {
    init {
        require(sourceModel.isNotBlank() && geometry.isNotBlank() && metricScaleSource.isNotBlank())
    }
}

data class DirectionalClearanceSample(
    val angleDegrees: Int,
    val nearestIntrusionMeters: Float?,
    val riskScore: Float?,
    val knownScore: Float,
    val intrusionPoints: Int,
    val supportPoints: Int,
    val observedForwardMeters: Float,
    val provenance: TraversabilityProvenance
) {
    init {
        require(angleDegrees in -89..89)
        require(nearestIntrusionMeters == null || nearestIntrusionMeters.isFinite() && nearestIntrusionMeters > 0f)
        require(riskScore == null || riskScore in 0f..1f)
        require((nearestIntrusionMeters == null) == (riskScore == null))
        require(knownScore in 0f..1f)
        require(intrusionPoints >= 0 && supportPoints >= 0)
        require(observedForwardMeters.isFinite() && observedForwardMeters >= 0f)
    }
}

data class DirectionalSweepState(
    val angleDegrees: Int,
    val state: SweepObservationState
)

data class BodySweepEnvelope(
    val horizonMeters: Float,
    val bodyHalfWidthMeters: Float,
    val lateralMarginMeters: Float,
    val directions: List<DirectionalSweepState>
) {
    init {
        require(horizonMeters.isFinite() && horizonMeters > 0f)
        require(bodyHalfWidthMeters.isFinite() && bodyHalfWidthMeters > 0f)
        require(lateralMarginMeters.isFinite() && lateralMarginMeters >= 0f)
        require(directions.isNotEmpty())
        require(directions.map { it.angleDegrees } == directions.map { it.angleDegrees }.distinct().sorted())
    }
}

data class TraversabilityIntrusionRegion(
    val regionId: Int,
    val minimumAngleDegrees: Int,
    val maximumAngleDegrees: Int,
    val nearestIntrusionMeters: Float,
    val status: String = "OBSERVED_CLASS_FREE_INTRUSION"
) {
    init {
        require(regionId > 0 && minimumAngleDegrees <= maximumAngleDegrees)
        require(nearestIntrusionMeters.isFinite() && nearestIntrusionMeters > 0f)
        require(status == "OBSERVED_CLASS_FREE_INTRUSION")
    }
}

data class ObservedClearanceCandidate(
    val angleDegrees: Int,
    val nearestIntrusionMeters: Float,
    val status: String = "DEMO_CANDIDATE_NOT_SAFE_DIRECTION"
) {
    init {
        require(angleDegrees in -89..89)
        require(nearestIntrusionMeters.isFinite() && nearestIntrusionMeters > 0f)
        require(status == "DEMO_CANDIDATE_NOT_SAFE_DIRECTION")
    }
}

data class TraversabilityTemporalTrend(
    val status: String,
    val medianDeltaMeters: Float?,
    val pairedDirectionCount: Int,
    val claimCeiling: String
) {
    init {
        require(status.isNotBlank() && pairedDirectionCount >= 0 && claimCeiling.isNotBlank())
        require(medianDeltaMeters == null || medianDeltaMeters.isFinite())
    }
}

data class TraversabilityQuality(
    val imageQualityPass: Boolean?,
    val laplacianVariance320x240: Float?,
    val underexposedFraction: Float?,
    val overexposedFraction: Float?,
    val depthFiniteFraction: Float,
    val depthSupportPass: Boolean,
    val groundSupportPass: Boolean,
    val directionSupportFraction: Float,
    val overallConfidence: Float
) {
    init {
        require(
            laplacianVariance320x240 == null ||
                laplacianVariance320x240.isFinite() && laplacianVariance320x240 >= 0f
        )
        listOf(underexposedFraction, overexposedFraction).filterNotNull().forEach {
            require(it in 0f..1f)
        }
        require(depthFiniteFraction in 0f..1f)
        require(directionSupportFraction in 0f..1f)
        require(overallConfidence in 0f..1f)
    }
}

data class MetricTraversabilityField(
    val frameId: Long,
    val capturedAtNs: Long,
    val sourceId: String,
    val status: TraversabilityFieldStatus,
    val calibratedDepth: MetricDepthSummary,
    val groundPlane: TraversabilityGroundPlane?,
    val clearanceProfile: List<DirectionalClearanceSample>,
    val sweepEnvelopes: List<BodySweepEnvelope>,
    val intrusionRegions: List<TraversabilityIntrusionRegion>,
    val bestObservedClearanceDirection: ObservedClearanceCandidate?,
    val temporalTrend: TraversabilityTemporalTrend,
    val quality: TraversabilityQuality,
    val unknownReasons: List<String>,
    val authority: String = AUTHORITY,
    val claimCeiling: String = CLAIM_CEILING
) {
    init {
        require(frameId >= 0L && capturedAtNs >= 0L && sourceId.isNotBlank())
        require(authority == AUTHORITY && claimCeiling == CLAIM_CEILING)
        require(unknownReasons.all { it.isNotBlank() })
        val angles = clearanceProfile.map { it.angleDegrees }
        require(angles == angles.distinct().sorted())
        require(sweepEnvelopes.map { it.horizonMeters } == sweepEnvelopes.map { it.horizonMeters }.distinct().sorted())
        require(sweepEnvelopes.all { envelope -> envelope.directions.map { it.angleDegrees } == angles })
        if (status == TraversabilityFieldStatus.UNKNOWN) {
            require(unknownReasons.isNotEmpty())
            require(!calibratedDepth.available)
            require(groundPlane == null && clearanceProfile.isEmpty() && sweepEnvelopes.isEmpty())
            require(intrusionRegions.isEmpty() && bestObservedClearanceDirection == null)
        } else {
            require(unknownReasons.isEmpty() && calibratedDepth.available && groundPlane != null)
            require(clearanceProfile.isNotEmpty() && sweepEnvelopes.isNotEmpty())
        }
    }

    companion object {
        const val AUTHORITY = "DEVELOPMENT_ONLY_SHADOW_DEMO"
        const val CLAIM_CEILING =
            "observed geometry only; no safety, navigation, alert, or production authority"
    }
}
