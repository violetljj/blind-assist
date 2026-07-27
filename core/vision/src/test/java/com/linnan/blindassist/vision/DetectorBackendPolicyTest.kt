package com.linnan.blindassist.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DetectorBackendPolicyTest {
    @Test
    fun productionRouteAuthorizesQnnAndCpuFallback() {
        assertEquals(
            DetectorExecutionBackend.QUALCOMM_QNN_HTP,
            DetectorBackendPolicy.productionDefault
        )
        assertTrue(
            DetectorBackendPolicy.isProductionAuthorized(
                DetectorExecutionBackend.CPU_XNNPACK
            )
        )
        assertTrue(
            DetectorBackendPolicy.isProductionAuthorized(
                DetectorExecutionBackend.QUALCOMM_QNN_HTP
            )
        )
    }

    @Test
    fun promotedQnnAndCpuFallbackAreBothProductionRouteMembers() {
        assertEquals(null, DetectorBackendPolicy.nextDefaultCandidate)
        assertEquals(
            DetectorBackendRole.PRODUCTION_DEFAULT,
            DetectorBackendPolicy.roleFor(DetectorExecutionBackend.QUALCOMM_QNN_HTP)
        )
        assertEquals(
            DetectorBackendRole.PRODUCTION_DEFAULT,
            DetectorBackendPolicy.roleFor(DetectorExecutionBackend.CPU_XNNPACK)
        )
        assertEquals(
            DetectorBackendRole.BENCHMARK_COMPARATOR,
            DetectorBackendPolicy.roleFor(DetectorExecutionBackend.GPU_DELEGATE)
        )
    }

    @Test
    fun onlyBlockingGatesControlPromotionReadiness() {
        val gates = DetectorBackendPolicy.decision.candidateGates
        assertEquals(BackendPromotionGate.entries.size, gates.size)
        assertEquals(
            BackendPromotionGate.entries.toSet(),
            gates.map { it.gate }.toSet()
        )
        assertEquals(
            emptySet<BackendPromotionGate>(),
            DetectorBackendPolicy.decision.blockingGates
                .filter { it.state != BackendPromotionGateState.PASS }
                .map { it.gate }
                .toSet()
        )
        assertTrue(DetectorBackendPolicy.decision.candidatePromotionReady)
    }

    @Test
    fun diagnosticsAndUnthresholdedReleaseMeasurementsCannotVetoPromotion() {
        val synthetic = DetectorBackendDecision(
            policyId = "test",
            productionDefault = DetectorExecutionBackend.CPU_XNNPACK,
            productionFallback = DetectorExecutionBackend.CPU_XNNPACK,
            nextDefaultCandidate = DetectorExecutionBackend.QUALCOMM_QNN_HTP,
            candidateGates = listOf(
                BackendPromotionGateDecision(
                    BackendPromotionGate.RUNTIME_INTEGRITY,
                    BackendPromotionGateClass.BLOCKING,
                    BackendPromotionGateState.PASS,
                    "verified"
                ),
                BackendPromotionGateDecision(
                    BackendPromotionGate.PACKAGE_FOOTPRINT,
                    BackendPromotionGateClass.RELEASE_CONSTRAINT,
                    BackendPromotionGateState.HOLD,
                    "measured without a frozen veto threshold"
                ),
                BackendPromotionGateDecision(
                    BackendPromotionGate.ENERGY_OBSERVATION,
                    BackendPromotionGateClass.DIAGNOSTIC,
                    BackendPromotionGateState.NOT_EVALUATED,
                    "unknown"
                )
            )
        )

        assertTrue(synthetic.candidatePromotionReady)
    }

    @Test
    fun qnnCanEnterProductionSelectionAfterAllBlockingGatesPass() {
        DetectorBackendPolicy.requireProductionAuthorized(
            DetectorExecutionBackend.QUALCOMM_QNN_HTP
        )
    }

    @Test
    fun benchmarkPackageCanInjectComparatorBackends() {
        DetectorBackendPolicy.requireExternalBackendInjectionAuthorized(
            DetectorBackendPolicy.BENCHMARK_PACKAGE,
            DetectorExecutionBackend.GPU_DELEGATE
        )
    }

    @Test
    fun candidatePackageCanInjectOnlyQnnHtp() {
        DetectorBackendPolicy.requireExternalBackendInjectionAuthorized(
            DetectorBackendPolicy.NPU_CANDIDATE_PACKAGE,
            DetectorExecutionBackend.QUALCOMM_QNN_HTP
        )
    }

    @Test
    fun appPackageCanInjectPromotedQnnBackend() {
        DetectorBackendPolicy.requireExternalBackendInjectionAuthorized(
            DetectorBackendPolicy.PRODUCTION_PACKAGE,
            DetectorExecutionBackend.QUALCOMM_QNN_HTP
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun candidatePackageCannotInjectGpuComparator() {
        DetectorBackendPolicy.requireExternalBackendInjectionAuthorized(
            DetectorBackendPolicy.NPU_CANDIDATE_PACKAGE,
            DetectorExecutionBackend.GPU_DELEGATE
        )
    }
}
