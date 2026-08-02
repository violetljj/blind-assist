package com.linnan.blindassist.hftf.metricdepth

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MetricDepthTargetCanaryTest {
    @Test
    fun registeredPersonBoxProducesMetricPositionAndQualityWithoutRiskEvidence() {
        val result = sampler().sample(
            frame = depthFrame(),
            detection = person(),
            targetKey = "track-7",
            observedAtNs = BASE_NS + 80_000_000L
        )

        assertTrue(result is MetricDepthSampleResult.Available)
        val measurement = (result as MetricDepthSampleResult.Available).measurement
        assertEquals(2f, measurement.opticalAxisDepthMeters, 0.0001f)
        assertEquals(0f, measurement.positionCameraMeters.x, 0.0001f)
        assertEquals(0f, measurement.positionCameraMeters.y, 0.0001f)
        assertEquals(2f, measurement.positionCameraMeters.z, 0.0001f)
        assertEquals(36, measurement.diagnostics.candidateSampleCount)
        assertEquals(36, measurement.diagnostics.validSampleCount)
        assertEquals(1f, measurement.diagnostics.coverage, 0.0001f)
        assertEquals(null, person().distanceEvidence)
    }

    @Test
    fun insufficientRegisteredCoverageFailsClosedWithDiagnostics() {
        val depth = IntArray(100)
        val confidence = FloatArray(100)
        listOf(22, 23, 24, 25, 32, 33, 34, 35).forEach { index ->
            depth[index] = 2_000
            confidence[index] = 0.9f
        }
        val result = sampler().sample(
            frame = depthFrame(depth, confidence),
            detection = person(),
            targetKey = "track-7",
            observedAtNs = BASE_NS + 80_000_000L
        )

        assertTrue(result is MetricDepthSampleResult.Unavailable)
        val unavailable = result as MetricDepthSampleResult.Unavailable
        assertEquals(MetricDepthSampleFailure.INSUFFICIENT_VALID_SAMPLES, unavailable.failure)
        assertEquals(8, unavailable.diagnostics.validSampleCount)
    }

    @Test
    fun staleDepthReceiptNeverProducesMeasurement() {
        val result = sampler().sample(
            frame = depthFrame().copy(validUntilNs = BASE_NS + 100_000_000L),
            detection = person(),
            targetKey = "track-7",
            observedAtNs = BASE_NS + 160_000_000L
        )

        assertEquals(
            MetricDepthSampleFailure.RECEIPT_STALE,
            (result as MetricDepthSampleResult.Unavailable).failure
        )
    }

    @Test
    fun nativeRawDepthGridIsSampledThroughRegistrationWithoutUpsamplingDuplicates() {
        val registration = registration(
            rawDepthSize = FrameSize(5, 5),
            cameraToRawDepth = MetricAffineTransform2D(
                m00 = 0.5f,
                m01 = 0f,
                m02 = 0f,
                m10 = 0f,
                m11 = 0.5f,
                m12 = 0f
            )
        )
        val result = MetricDepthTargetSampler(
            MetricDepthTargetSamplerConfig(minimumValidSamples = 1)
        ).sample(
            frame = depthFrame(
                depth = IntArray(25) { 2_000 },
                confidence = FloatArray(25) { 0.9f },
                registration = registration
            ),
            detection = person(),
            targetKey = "track-7",
            observedAtNs = BASE_NS + 80_000_000L
        )

        val measurement = (result as MetricDepthSampleResult.Available).measurement
        assertEquals(9, measurement.diagnostics.candidateSampleCount)
        assertEquals(9, measurement.diagnostics.validSampleCount)
    }

    @Test
    fun detectorRegionOutsideDepthCropIsRegistrationAbsenceNotLowDepthQuality() {
        val result = MetricDepthTargetSampler(
            MetricDepthTargetSamplerConfig(minimumValidSamples = 1)
        ).sample(
            frame = depthFrame(
                depth = IntArray(25) { 2_000 },
                confidence = FloatArray(25) { 0.9f },
                registration = registration(
                    rawDepthSize = FrameSize(5, 5),
                    cameraToRawDepth = MetricAffineTransform2D(
                        m00 = 0.5f,
                        m01 = 0f,
                        m02 = 100f,
                        m10 = 0f,
                        m11 = 0.5f,
                        m12 = 100f
                    )
                )
            ),
            detection = person(),
            targetKey = "track-7",
            observedAtNs = BASE_NS + 80_000_000L
        )

        assertEquals(
            MetricDepthSampleFailure.NO_REGISTERED_PIXELS,
            (result as MetricDepthSampleResult.Unavailable).failure
        )
        assertEquals(0, result.diagnostics.candidateSampleCount)
    }

    @Test
    fun sevenPointMetricHistoryRecoversConstantCameraRelativeMotionAtOneSecond() {
        val history = (0 until 7).map { index ->
            val timeSeconds = index * 0.05f
            measurement(
                index = index,
                capturedAtNs = BASE_NS + index * 50_000_000L,
                position = MetricVector3Meters(
                    x = 1f + 0.5f * timeSeconds,
                    y = -0.2f,
                    z = 3f - timeSeconds
                )
            )
        }

        val result = MetricDepthHistorySolver().forecast(history)

        assertTrue(result is MetricDepthHistoryResult.Available)
        val forecast = (result as MetricDepthHistoryResult.Available).forecast
        assertEquals(1.65f, forecast.predictedPositionCameraMeters.x, 0.0001f)
        assertEquals(-0.2f, forecast.predictedPositionCameraMeters.y, 0.0001f)
        assertEquals(1.7f, forecast.predictedPositionCameraMeters.z, 0.0001f)
        assertEquals(0.5f, forecast.velocityCameraMetersPerSecond.x, 0.0001f)
        assertEquals(-1f, forecast.velocityCameraMetersPerSecond.z, 0.0001f)
        assertEquals(BASE_NS + 1_300_000_000L, forecast.predictedAtNs)
    }

    @Test
    fun mixedTargetHistoryCannotCreateForecast() {
        val history = (0 until 7).map { index ->
            measurement(
                index = index,
                capturedAtNs = BASE_NS + index * 50_000_000L,
                targetKey = if (index == 4) "other-target" else "track-7"
            )
        }

        assertEquals(
            MetricDepthHistoryFailure.MIXED_TARGET,
            (MetricDepthHistorySolver().forecast(history) as MetricDepthHistoryResult.Unavailable).failure
        )
    }

    @Test
    fun rawConfidenceCapabilityTakesPrecedenceOverAutomaticDepth() {
        assertEquals(
            D45InstalledArCoreDepthCapability.READY_RAW_DEPTH_REGISTRATION_REQUIRED,
            D45ArCoreDepthCapabilityClassifier.classifyInstalled(
                supportsAutomatic = true,
                supportsRaw = true,
                hardwareDepthCameraConfigCount = 1
            )
        )
    }

    @Test
    fun automaticOnlyCapabilityCannotSatisfyFrozenConfidenceContract() {
        assertEquals(
            D45InstalledArCoreDepthCapability
                .AUTOMATIC_ONLY_ESTIMATED_CONFIDENCE_UNAVAILABLE,
            D45ArCoreDepthCapabilityClassifier.classifyInstalled(
                supportsAutomatic = true,
                supportsRaw = false,
                hardwareDepthCameraConfigCount = 0
            )
        )
    }

    private fun sampler() = MetricDepthTargetSampler()

    private fun person() = Detection(
        classId = 0,
        label = "person",
        confidence = 0.9f,
        boundingBox = BoundingBox(0f, 0f, 10f, 10f),
        frameSize = FRAME_SIZE
    )

    private fun depthFrame(
        depth: IntArray = IntArray(100) { 2_000 },
        confidence: FloatArray = FloatArray(100) { 0.9f },
        registration: MetricDepthRegistrationTransform = registration()
    ) = RegisteredMetricDepthFrame(
        sourceFrame = FrameStamp(
            frameId = 7L,
            capturedAtNs = BASE_NS,
            receivedAtNs = BASE_NS + 10_000_000L,
            sourceId = "arcore:camera0",
            coordinateFrame = "arcore:camera0:display",
            clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
        ),
        detectorDisplaySize = FRAME_SIZE,
        intrinsics = MetricDepthCameraIntrinsics(
            imageWidthPx = 10,
            imageHeightPx = 10,
            focalXpx = 10f,
            focalYpx = 10f,
            principalXpx = 5f,
            principalYpx = 5f
        ),
        depthMillimeters = depth,
        confidence = confidence,
        source = MetricDepthSource.ARCORE_RAW_REGISTERED,
        registration = registration,
        producedAtNs = BASE_NS + 20_000_000L,
        validUntilNs = BASE_NS + 150_000_000L
    )

    private fun registration(
        rawDepthSize: FrameSize = FRAME_SIZE,
        cameraToRawDepth: MetricAffineTransform2D = MetricAffineTransform2D.IDENTITY
    ) = MetricDepthRegistrationTransform(
        detectorDisplaySize = FRAME_SIZE,
        cameraImageSize = FRAME_SIZE,
        rawDepthSize = rawDepthSize,
        detectorRotationDegrees = 0,
        detectorToCameraImage = MetricAffineTransform2D.IDENTITY,
        cameraImageToRawDepth = cameraToRawDepth,
        maximumFitResidualPx = 0f,
        transformId = "arcore-image-to-depth-v1"
    )

    private fun measurement(
        index: Int,
        capturedAtNs: Long,
        position: MetricVector3Meters = MetricVector3Meters(1f, 0f, 2f),
        targetKey: String = "track-7"
    ) = MetricDepthTargetMeasurement(
        targetKey = targetKey,
        frameId = index.toLong(),
        capturedAtNs = capturedAtNs,
        source = MetricDepthSource.ARCORE_AUTOMATIC,
        registrationTransformId = "arcore-image-to-depth-v1",
        opticalAxisDepthMeters = position.z,
        positionCameraMeters = position,
        qualityScore = 0.8f,
        diagnostics = MetricDepthSampleDiagnostics(
            candidateSampleCount = 20,
            validSampleCount = 18,
            coverage = 0.9f,
            meanConfidence = 0.9f,
            relativeIqr = 0.1f,
            receiptAgeNs = 20_000_000L
        ),
        producedAtNs = capturedAtNs + 20_000_000L
    )

    private companion object {
        val FRAME_SIZE = FrameSize(10, 10)
        const val BASE_NS = 1_000_000_000L
    }
}
