package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfMetricDepthGeometryAdapterTest {
    @Test
    fun calibratedMetricSamplesProduceGroundBodyAndHeadEvidenceWithoutInventingDrops() {
        val result = adapter().project(admission(), depth(), intrinsics(), extrinsics(), ground(), 1_100L)

        assertTrue(result is UstrfMetricDepthGeometryAdapterResult.Available)
        val available = result as UstrfMetricDepthGeometryAdapterResult.Available
        assertEquals(UstrfDepthScale.METRIC, available.packet.scale)
        assertEquals(3, available.admittedEvidenceCount)
        assertTrue(available.packet.evidence.any { it.kind == UstrfGeometryKind.TRAVERSABLE && it.heightBand == UstrfHeightBand.GROUND })
        assertTrue(available.packet.evidence.any { it.kind == UstrfGeometryKind.OCCUPIED && it.heightBand == UstrfHeightBand.LOWER_BODY })
        assertTrue(available.packet.evidence.any { it.kind == UstrfGeometryKind.HEAD_OBSTACLE && it.heightBand == UstrfHeightBand.HEAD })
        assertFalse(available.packet.evidence.any { it.kind == UstrfGeometryKind.DROP })
    }

    @Test
    fun wrongRegistrationOrUnverifiedGroundFailsClosed() {
        val wrongRegistration = depth().copy(registrationTransformId = "another-transform")
        assertEquals(
            UstrfMetricDepthGeometryAdapterResult.Unavailable(UstrfMetricDepthGeometryAdapterFailure.DEPTH_REGISTRATION_MISMATCH),
            adapter().project(admission(), wrongRegistration, intrinsics(), extrinsics(), ground(), 1_100L)
        )
        assertEquals(
            UstrfMetricDepthGeometryAdapterResult.Unavailable(UstrfMetricDepthGeometryAdapterFailure.GROUND_NOT_INDEPENDENTLY_VERIFIED),
            adapter().project(admission(), depth(), intrinsics(), extrinsics(), ground().copy(independentlyVerified = false), 1_100L)
        )
    }

    @Test
    fun staleDepthNeverCreatesGeometryPacket() {
        val stale = depth().copy(validUntilNs = 1_050L)
        assertEquals(
            UstrfMetricDepthGeometryAdapterResult.Unavailable(UstrfMetricDepthGeometryAdapterFailure.DEPTH_STALE),
            adapter().project(admission(), stale, intrinsics(), extrinsics(), ground(), 1_100L)
        )
    }

    @Test
    fun admittedDepthPacketFlowsIntoTheDocumentFiveMeterGeometryGrid() {
        val projected = adapter().project(admission(), depth(), intrinsics(), extrinsics(), ground(), 1_100L)
            as UstrfMetricDepthGeometryAdapterResult.Available
        val assembled = UstrfDocumentFiveMeterProfile.perceptionAssembler().assemble(
            frame(), projected.packet, emptyList(), 1_100L
        ) as UstrfPerceptionAssembly.Available

        assertTrue(assembled.packet.observations.any {
            it.coordinate == UstrfGridCoordinate(0, 2) && it.occupancy > 0f
        })
        assertTrue(assembled.packet.observations.any {
            it.coordinate == UstrfGridCoordinate(2, 2) && it.headRisk > 0f
        })
    }

    private fun adapter() = UstrfMetricDepthGeometryAdapter(
        UstrfMetricDepthGeometryAdapterConfig(sampleStridePx = 1, minimumDepthMeters = .2f, maximumDepthMeters = 5f)
    )

    private fun frame() = UstrfFrameStamp(7L, 1_000L, "camera")

    private fun admission() = UstrfMetricGeometryProjectionAdmission.Available(
        sourceFrame = frame(),
        depthCoordinateFrame = "registered-depth",
        cameraFrame = "camera",
        bodyFrame = "body",
        calibrationId = "mount-v1",
        calibrationSourceArtifactSha256 = "a".repeat(64),
        registrationTransformId = "depth-to-camera-v1",
        validUntilNs = 2_000L
    )

    private fun intrinsics() = UstrfCameraIntrinsicsReceipt(
        cameraFrame = "camera", calibrationVersion = "cal-v1", imageWidthPx = 3, imageHeightPx = 3,
        focalXpx = 1f, focalYpx = 1f, principalXpx = 1f, principalYpx = 1f,
        verifiedAtNs = 1L, validUntilNs = 2_000L, confidence = 1f, independentlyVerified = true
    )

    private fun extrinsics() = UstrfCameraBodyFullExtrinsicsReceipt(
        cameraFrame = "camera", bodyFrame = "body", cameraToBodyTranslationM = UstrfVector3(0f, 1.5f, 0f),
        cameraToBodyQuaternionXyzw = floatArrayOf(0f, 0f, 0f, 1f), calibrationId = "mount-v1",
        verifiedAtNs = 1L, validUntilNs = 2_000L, confidence = 1f, independentlyVerified = true
    )

    private fun ground() = UstrfVerifiedGroundPlaneReceipt(
        sourceFrame = frame(), bodyFrame = "body", normal = UstrfVector3(0f, 1f, 0f), offsetMeters = 0f,
        confidence = 1f, independentlyVerified = true, validUntilNs = 2_000L
    )

    private fun depth(): UstrfRegisteredMetricDepthImage {
        val millimeters = IntArray(9)
        val confidence = FloatArray(9)
        // (u=0,v=2): ground; (u=1,v=2): lower-body; (u=2,v=1): head.
        // (u=2,v=0) is 2.5 m above ground and must stay outside the swept clearance band.
        millimeters[2] = 1_000; confidence[2] = 1f
        millimeters[6] = 1_500; confidence[6] = 1f
        millimeters[7] = 1_000; confidence[7] = 1f
        millimeters[5] = 1_000; confidence[5] = 1f
        return UstrfRegisteredMetricDepthImage(
            sourceFrame = frame(), widthPx = 3, heightPx = 3, depthCoordinateFrame = "registered-depth",
            registrationTransformId = "depth-to-camera-v1", depthMillimeters = millimeters, confidence = confidence,
            validUntilNs = 2_000L
        )
    }
}
