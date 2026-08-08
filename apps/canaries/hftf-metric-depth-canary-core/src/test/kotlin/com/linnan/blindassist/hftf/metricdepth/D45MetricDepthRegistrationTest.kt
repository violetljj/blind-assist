package com.linnan.blindassist.hftf.metricdepth

import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class D45MetricDepthRegistrationTest {
    @Test
    fun cropAndScaleCorrespondencesRecoverCameraToRawDepthAffine() {
        val result = D45MetricDepthRegistrationFactory.create(
            cameraImageSize = FrameSize(8, 6),
            rawDepthSize = FrameSize(4, 4),
            detectorRotationDegrees = 0,
            correspondences = probePoints(FrameSize(8, 6)).map { camera ->
                D45CoordinateCorrespondence(
                    cameraImagePoint = camera,
                    rawDepthPoint = MetricImagePoint(
                        x = 0.5f * camera.x,
                        y = 0.5f * camera.y + 0.5f
                    )
                )
            }
        )

        val registration =
            (result as D45MetricDepthRegistrationResult.Available).registration
        val mapped = registration.detectorToRawDepth.map(MetricImagePoint(6f, 4f))
        assertEquals(3f, mapped.x, 0.0001f)
        assertEquals(2.5f, mapped.y, 0.0001f)
        assertEquals(0f, registration.maximumFitResidualPx, 0.0001f)
    }

    @Test
    fun detectorClockwiseRotationMapsDisplayCoordinatesBackToNativeCameraImage() {
        val registration = (
            D45MetricDepthRegistrationFactory.create(
                cameraImageSize = FrameSize(4, 3),
                rawDepthSize = FrameSize(4, 3),
                detectorRotationDegrees = 90,
                correspondences = probePoints(FrameSize(4, 3)).map {
                    D45CoordinateCorrespondence(it, it)
                }
            ) as D45MetricDepthRegistrationResult.Available
            ).registration

        assertEquals(FrameSize(3, 4), registration.detectorDisplaySize)
        val native = registration.detectorToCameraImage.map(MetricImagePoint(1f, 2f))
        assertEquals(2f, native.x, 0.0001f)
        assertEquals(2f, native.y, 0.0001f)
        val roundTrip = registration.rawDepthToDetector.map(
            registration.detectorToRawDepth.map(MetricImagePoint(1.25f, 2.5f))
        )
        assertEquals(1.25f, roundTrip.x, 0.0001f)
        assertEquals(2.5f, roundTrip.y, 0.0001f)
    }

    @Test
    fun nonAffineCoordinateReceiptFailsBeforeRegistrationCanEscape() {
        val correspondences = probePoints(FrameSize(8, 6)).mapIndexed { index, point ->
            D45CoordinateCorrespondence(
                cameraImagePoint = point,
                rawDepthPoint = MetricImagePoint(
                    x = point.x / 2f + if (index == 4) 2f else 0f,
                    y = point.y / 2f
                )
            )
        }

        val result = D45MetricDepthRegistrationFactory.create(
            cameraImageSize = FrameSize(8, 6),
            rawDepthSize = FrameSize(4, 3),
            detectorRotationDegrees = 0,
            correspondences = correspondences,
            maximumAllowedResidualPx = 0.25f
        )

        assertEquals(
            D45MetricDepthRegistrationFailure.AFFINE_RESIDUAL_ABOVE_TOLERANCE,
            (result as D45MetricDepthRegistrationResult.Unavailable).failure
        )
    }

    @Test
    fun transformIdChangesWhenDetectorRotationChanges() {
        val correspondences = probePoints(FrameSize(8, 6)).map {
            D45CoordinateCorrespondence(it, MetricImagePoint(it.x / 2f, it.y / 2f))
        }
        val rotation0 = (
            D45MetricDepthRegistrationFactory.create(
                FrameSize(8, 6),
                FrameSize(4, 3),
                0,
                correspondences
            ) as D45MetricDepthRegistrationResult.Available
            ).registration
        val rotation90 = (
            D45MetricDepthRegistrationFactory.create(
                FrameSize(8, 6),
                FrameSize(4, 3),
                90,
                correspondences
            ) as D45MetricDepthRegistrationResult.Available
            ).registration

        assertTrue(rotation0.transformId.startsWith("d45-arcore-registration-v1:"))
        assertNotEquals(rotation0.transformId, rotation90.transformId)
    }

    @Test
    fun subQuantumCoordinateNoiseDoesNotFragmentRegistrationHistoryIdentity() {
        fun registration(offset: Float) = (
            D45MetricDepthRegistrationFactory.create(
                FrameSize(8, 6),
                FrameSize(4, 3),
                0,
                probePoints(FrameSize(8, 6)).map {
                    D45CoordinateCorrespondence(
                        it,
                        MetricImagePoint(it.x / 2f + offset, it.y / 2f)
                    )
                }
            ) as D45MetricDepthRegistrationResult.Available
            ).registration

        assertEquals(
            registration(0f).transformId,
            registration(0.0000001f).transformId
        )
    }

    @Test
    fun registrationFromAnotherFrameCannotUnlockRawRaster() {
        val registration = (
            D45MetricDepthRegistrationFactory.create(
                FrameSize(8, 6),
                FrameSize(4, 3),
                0,
                probePoints(FrameSize(8, 6)).map {
                    D45CoordinateCorrespondence(
                        it,
                        MetricImagePoint(it.x / 2f, it.y / 2f)
                    )
                }
            ) as D45MetricDepthRegistrationResult.Available
            ).registration
        val frame = D45UnregisteredRawMetricDepthFrame(
            sourceFrame = FrameStamp(
                frameId = 7,
                capturedAtNs = 1_000,
                receivedAtNs = 1_100,
                sourceId = "arcore:raw-depth",
                coordinateFrame = "arcore:camera-image",
                clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
            ),
            sourceImageIntrinsics = MetricDepthCameraIntrinsics(
                imageWidthPx = 8,
                imageHeightPx = 6,
                focalXpx = 8f,
                focalYpx = 8f,
                principalXpx = 4f,
                principalYpx = 3f
            ),
            raster = D45DecodedRawDepthRaster(
                widthPx = 4,
                heightPx = 3,
                depthMillimeters = IntArray(12) { 2_000 },
                confidence = FloatArray(12) { 0.9f }
            ),
            depthTimestampNs = 1_000,
            confidenceTimestampNs = 1_000,
            producedAtNs = 1_200
        )

        try {
            D45MetricDepthFrameRegistrar.register(
                frame = frame,
                registrationObservation = D45FrameBoundMetricDepthRegistration(
                    sourceFrameId = 8,
                    sourceCapturedAtNs = 1_000,
                    transform = registration
                ),
                validUntilNs = 1_300
            )
            fail("cross-frame registration must fail")
        } catch (_: IllegalArgumentException) {
            // Expected type-level/source-frame firewall.
        }
    }

    private fun probePoints(size: FrameSize) = listOf(
        MetricImagePoint(0f, 0f),
        MetricImagePoint(size.width.toFloat(), 0f),
        MetricImagePoint(0f, size.height.toFloat()),
        MetricImagePoint(size.width.toFloat(), size.height.toFloat()),
        MetricImagePoint(size.width / 2f, size.height / 2f),
        MetricImagePoint(size.width / 2f, 0f),
        MetricImagePoint(0f, size.height / 2f)
    )
}
