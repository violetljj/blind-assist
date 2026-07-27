package com.linnan.blindassist.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ProductionDetectorRoutePolicyTest {
    @Test
    fun sm8650Arm64WithLiveCapabilitySelectsQnn() {
        val profile = ProductionDeviceProfile("SM8650", listOf("arm64-v8a"))

        assertTrue(ProductionDetectorRoutePolicy.isQnnProbeEligible(profile))
        assertEquals(
            DetectorExecutionBackend.QUALCOMM_QNN_HTP,
            ProductionDetectorRoutePolicy.decide(profile, qnnHtpFp16Available = true).backend
        )
    }

    @Test
    fun unsupportedSocDoesNotEnterQnnProbeAndUsesCpu() {
        val profile = ProductionDeviceProfile("unknown", listOf("arm64-v8a"))

        assertFalse(ProductionDetectorRoutePolicy.isQnnProbeEligible(profile))
        assertEquals(
            DetectorExecutionBackend.CPU_XNNPACK,
            ProductionDetectorRoutePolicy.decide(profile, qnnHtpFp16Available = true).backend
        )
    }

    @Test
    fun missingArm64OrFailedLiveCapabilityUsesCpu() {
        val missingAbi = ProductionDeviceProfile("SM8650", listOf("x86_64"))
        val supportedDevice = ProductionDeviceProfile("SM8650", listOf("arm64-v8a"))

        assertEquals(
            DetectorExecutionBackend.CPU_XNNPACK,
            ProductionDetectorRoutePolicy.decide(missingAbi, qnnHtpFp16Available = true).backend
        )
        assertEquals(
            DetectorExecutionBackend.CPU_XNNPACK,
            ProductionDetectorRoutePolicy.decide(supportedDevice, qnnHtpFp16Available = false).backend
        )
    }
}
