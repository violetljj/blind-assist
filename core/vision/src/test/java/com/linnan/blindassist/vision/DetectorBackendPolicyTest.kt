package com.linnan.blindassist.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DetectorBackendPolicyTest {
    @Test
    fun cpuRemainsTheOnlyProductionAuthorizedBackend() {
        assertEquals(
            DetectorExecutionBackend.CPU_XNNPACK,
            DetectorBackendPolicy.productionDefault
        )
        assertTrue(
            DetectorBackendPolicy.isProductionAuthorized(
                DetectorExecutionBackend.CPU_XNNPACK
            )
        )
        assertFalse(
            DetectorBackendPolicy.isProductionAuthorized(
                DetectorExecutionBackend.QUALCOMM_QNN_HTP
            )
        )
    }

    @Test
    fun qnnHtpIsTheSingleNextDefaultCandidate() {
        assertEquals(
            DetectorExecutionBackend.QUALCOMM_QNN_HTP,
            DetectorBackendPolicy.nextDefaultCandidate
        )
        assertEquals(
            DetectorBackendRole.NEXT_DEFAULT_CANDIDATE,
            DetectorBackendPolicy.roleFor(DetectorExecutionBackend.QUALCOMM_QNN_HTP)
        )
        assertEquals(
            DetectorBackendRole.BENCHMARK_COMPARATOR,
            DetectorBackendPolicy.roleFor(DetectorExecutionBackend.GPU_DELEGATE)
        )
    }

    @Test
    fun unresolvedAndUnevaluatedGatesFailPromotionClosed() {
        val gates = DetectorBackendPolicy.decision.candidateGates
        assertEquals(BackendPromotionGate.entries.size, gates.size)
        assertEquals(
            BackendPromotionGate.entries.toSet(),
            gates.map { it.gate }.toSet()
        )
        assertTrue(gates.any { it.state == BackendPromotionGateState.HOLD })
        assertTrue(gates.any { it.state == BackendPromotionGateState.NOT_EVALUATED })
        assertFalse(DetectorBackendPolicy.decision.candidatePromotionReady)
    }

    @Test(expected = IllegalArgumentException::class)
    fun candidateCannotEnterProductionSelectionBeforeAllGatesPass() {
        DetectorBackendPolicy.requireProductionAuthorized(
            DetectorExecutionBackend.QUALCOMM_QNN_HTP
        )
    }

    @Test
    fun externalBackendInjectionIsAuthorizedOnlyForBenchmarkPackage() {
        DetectorBackendPolicy.requireBenchmarkInjectionAuthorized(
            DetectorBackendPolicy.BENCHMARK_PACKAGE
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun appPackageCannotInjectExternalBackend() {
        DetectorBackendPolicy.requireBenchmarkInjectionAuthorized(
            "com.linnan.blindassist"
        )
    }
}
