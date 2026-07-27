package com.linnan.blindassist.vision

/**
 * Stable backend identities used by production selection and isolated device benchmarks.
 *
 * Declaring a backend here does not package its runtime or authorize it for production.
 */
enum class DetectorExecutionBackend(val wireName: String) {
    CPU_XNNPACK("cpu_xnnpack"),
    GPU_DELEGATE("gpu_delegate"),
    QUALCOMM_QNN_HTP("qualcomm_qnn_htp")
}

enum class DetectorBackendRole {
    PRODUCTION_DEFAULT,
    NEXT_DEFAULT_CANDIDATE,
    BENCHMARK_COMPARATOR
}

enum class BackendPromotionGateState {
    PASS,
    HOLD,
    OBSERVED,
    NOT_EVALUATED
}

enum class BackendPromotionGateClass {
    /** A failed or unresolved item here is the only thing that can block default selection. */
    BLOCKING,
    /** Must be measured and disclosed for release, but has no authority without a frozen limit. */
    RELEASE_CONSTRAINT,
    /** Useful for attribution and monitoring; never an automatic promotion veto. */
    DIAGNOSTIC
}

enum class BackendPromotionGate {
    RUNTIME_INTEGRITY,
    CRITICAL_RISK_NON_REGRESSION,
    ALERT_LIFECYCLE_NON_REGRESSION,
    SUSTAINED_RUNTIME_STABILITY,
    TARGET_DEVICE_SCOPE_AND_CPU_ROUTE,
    ROLLBACK_INTEGRITY,
    LATENCY_IMPACT,
    PACKAGE_FOOTPRINT,
    COLD_START_IMPACT,
    DETECTION_NUMERIC_ATTRIBUTION,
    ENERGY_OBSERVATION
}

data class BackendPromotionGateDecision(
    val gate: BackendPromotionGate,
    val gateClass: BackendPromotionGateClass,
    val state: BackendPromotionGateState,
    val reason: String
)

data class DetectorBackendDecision(
    val policyId: String,
    val productionDefault: DetectorExecutionBackend,
    val productionFallback: DetectorExecutionBackend,
    val nextDefaultCandidate: DetectorExecutionBackend?,
    val candidateGates: List<BackendPromotionGateDecision>
) {
    val blockingGates: List<BackendPromotionGateDecision>
        get() = candidateGates.filter { it.gateClass == BackendPromotionGateClass.BLOCKING }

    val candidatePromotionReady: Boolean
        get() = blockingGates.isNotEmpty() &&
            blockingGates.all { it.state == BackendPromotionGateState.PASS }
}

/**
 * Current fail-closed backend decision.
 *
 * Promotion is blocked only by behavior, runtime integrity, device routing, stability,
 * and rollback gates. Numeric detector parity, package size, cold start, and energy are
 * release/diagnostic observations unless a limit was frozen before measurement.
 */
object DetectorBackendPolicy {
    const val POLICY_ID = "blindassist_detector_backend_policy_20260727_v3"
    const val PRODUCTION_PACKAGE = "com.linnan.blindassist"
    const val BENCHMARK_PACKAGE = "com.linnan.blindassist.benchmark"
    const val NPU_CANDIDATE_PACKAGE = "com.linnan.blindassist.npu.candidate"

    val decision = DetectorBackendDecision(
        policyId = POLICY_ID,
        productionDefault = DetectorExecutionBackend.QUALCOMM_QNN_HTP,
        productionFallback = DetectorExecutionBackend.CPU_XNNPACK,
        nextDefaultCandidate = null,
        candidateGates = listOf(
            BackendPromotionGateDecision(
                BackendPromotionGate.RUNTIME_INTEGRITY,
                BackendPromotionGateClass.BLOCKING,
                BackendPromotionGateState.PASS,
                "SM8650 QNN 2.47 graph finalizes successfully, reports the QNN HTP runtime marker, and never silently falls back to CPU."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.CRITICAL_RISK_NON_REGRESSION,
                BackendPromotionGateClass.BLOCKING,
                BackendPromotionGateState.PASS,
                "Bounded same-device evidence shows 100/100 CPU parity for risk and feedback with no observed critical-event miss increase."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.ALERT_LIFECYCLE_NON_REGRESSION,
                BackendPromotionGateClass.BLOCKING,
                BackendPromotionGateState.PASS,
                "The 90-frame event replay has recall 1, zero repeated delivery, zero event regeneration, and final exit for both passed events."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.SUSTAINED_RUNTIME_STABILITY,
                BackendPromotionGateClass.BLOCKING,
                BackendPromotionGateState.PASS,
                "Ten-minute 10 FPS run completed without failures or thermal throttling."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.TARGET_DEVICE_SCOPE_AND_CPU_ROUTE,
                BackendPromotionGateClass.BLOCKING,
                BackendPromotionGateState.PASS,
                "Production selects QNN HTP only for arm64 SM8650 devices after a live FP16 capability check; every unsupported or failed capability/delegate path logs its reason and returns the CPU detector."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.ROLLBACK_INTEGRITY,
                BackendPromotionGateClass.BLOCKING,
                BackendPromotionGateState.PASS,
                "The exact CPU APK hash is restorable and launchable, then the NPU-route APK is restorable; user-state preservation is not evaluable because the device has zero user-owned files, while two disclosed platform/profile markers change on version launch."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.LATENCY_IMPACT,
                BackendPromotionGateClass.RELEASE_CONSTRAINT,
                BackendPromotionGateState.OBSERVED,
                "Same-device full-pipeline detector P50/P95 is 12/15 ms on NPU versus 53/55 ms on CPU."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.PACKAGE_FOOTPRINT,
                BackendPromotionGateClass.RELEASE_CONSTRAINT,
                BackendPromotionGateState.OBSERVED,
                "The arm64 candidate is 102,511,366 bytes versus 56,385,859 bytes for the CPU debug APK; no predeclared maximum exists."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.COLD_START_IMPACT,
                BackendPromotionGateClass.RELEASE_CONSTRAINT,
                BackendPromotionGateState.OBSERVED,
                "Candidate cold launch is 1,248 ms on SM8650; no predeclared maximum exists."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.DETECTION_NUMERIC_ATTRIBUTION,
                BackendPromotionGateClass.DIAGNOSTIC,
                BackendPromotionGateState.OBSERVED,
                "Strict box-level equivalence is 86/100; all 14 differing frames are attributed and preserve 100/100 risk and feedback decisions."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.ENERGY_OBSERVATION,
                BackendPromotionGateClass.DIAGNOSTIC,
                BackendPromotionGateState.NOT_EVALUATED,
                "USB-connected temperature testing cannot establish energy efficiency; this unknown is disclosed but is not an automatic veto."
            )
        )
    )

    val productionDefault: DetectorExecutionBackend
        get() = decision.productionDefault

    val nextDefaultCandidate: DetectorExecutionBackend?
        get() = decision.nextDefaultCandidate

    fun roleFor(backend: DetectorExecutionBackend): DetectorBackendRole = when (backend) {
        decision.productionDefault,
        decision.productionFallback -> DetectorBackendRole.PRODUCTION_DEFAULT
        decision.nextDefaultCandidate?.takeIf { it == backend } ->
            DetectorBackendRole.NEXT_DEFAULT_CANDIDATE
        else -> DetectorBackendRole.BENCHMARK_COMPARATOR
    }

    fun isProductionAuthorized(backend: DetectorExecutionBackend): Boolean =
        backend == productionDefault || backend == decision.productionFallback

    fun requireProductionAuthorized(backend: DetectorExecutionBackend) {
        require(isProductionAuthorized(backend)) {
            "$backend is not production-authorized by $POLICY_ID; " +
                "candidatePromotionReady=${decision.candidatePromotionReady}"
        }
    }

    fun requireExternalBackendInjectionAuthorized(
        packageName: String,
        backend: DetectorExecutionBackend
    ) {
        val authorized = packageName == BENCHMARK_PACKAGE ||
            (packageName == NPU_CANDIDATE_PACKAGE &&
                backend == DetectorExecutionBackend.QUALCOMM_QNN_HTP) ||
            (packageName == PRODUCTION_PACKAGE &&
                backend == decision.productionDefault &&
                decision.candidatePromotionReady)
        require(authorized) {
            "External $backend injection is not authorized for $packageName"
        }
    }
}
