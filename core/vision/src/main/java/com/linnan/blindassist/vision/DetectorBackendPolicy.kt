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
    NOT_EVALUATED
}

enum class BackendPromotionGate {
    FULL_GRAPH_DELEGATION,
    FULL_PIPELINE_LATENCY,
    RISK_AND_FEEDBACK_PARITY,
    TEN_MINUTE_STABILITY,
    DETECTION_SET_PARITY,
    COLD_START_AND_PACKAGE_SIZE,
    ENERGY_EFFICIENCY,
    SHARED_EVENT_LIFECYCLE
}

data class BackendPromotionGateDecision(
    val gate: BackendPromotionGate,
    val state: BackendPromotionGateState,
    val reason: String
)

data class DetectorBackendDecision(
    val policyId: String,
    val productionDefault: DetectorExecutionBackend,
    val nextDefaultCandidate: DetectorExecutionBackend,
    val candidateGates: List<BackendPromotionGateDecision>
) {
    val candidatePromotionReady: Boolean
        get() = candidateGates.isNotEmpty() &&
            candidateGates.all { it.state == BackendPromotionGateState.PASS }
}

/**
 * Current fail-closed backend decision.
 *
 * QNN HTP is the next default candidate because it leads the same-device latency and
 * stability measurements without changing risk/feedback decisions. It remains blocked
 * from production until every declared promotion gate passes.
 */
object DetectorBackendPolicy {
    const val POLICY_ID = "blindassist_detector_backend_policy_20260727_v1"
    const val BENCHMARK_PACKAGE = "com.linnan.blindassist.benchmark"

    val decision = DetectorBackendDecision(
        policyId = POLICY_ID,
        productionDefault = DetectorExecutionBackend.CPU_XNNPACK,
        nextDefaultCandidate = DetectorExecutionBackend.QUALCOMM_QNN_HTP,
        candidateGates = listOf(
            BackendPromotionGateDecision(
                BackendPromotionGate.FULL_GRAPH_DELEGATION,
                BackendPromotionGateState.PASS,
                "SM8650 logs show 548/548 nodes delegated to QNN HTP."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.FULL_PIPELINE_LATENCY,
                BackendPromotionGateState.PASS,
                "Same-device full detector P50 is materially lower than CPU and GPU."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.RISK_AND_FEEDBACK_PARITY,
                BackendPromotionGateState.PASS,
                "100-image risk/feedback and 90-frame event decisions match CPU."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.TEN_MINUTE_STABILITY,
                BackendPromotionGateState.PASS,
                "Ten-minute 10 FPS run completed without failures or thermal throttling."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.DETECTION_SET_PARITY,
                BackendPromotionGateState.HOLD,
                "Threshold-edge detection sets are not yet identical to CPU."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.COLD_START_AND_PACKAGE_SIZE,
                BackendPromotionGateState.HOLD,
                "Cold initialization and benchmark-only QNN packaging are not release-ready."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.ENERGY_EFFICIENCY,
                BackendPromotionGateState.NOT_EVALUATED,
                "USB-connected temperature testing does not establish energy efficiency."
            ),
            BackendPromotionGateDecision(
                BackendPromotionGate.SHARED_EVENT_LIFECYCLE,
                BackendPromotionGateState.HOLD,
                "The shared tracker regenerates identities and does not exit PASSED events."
            )
        )
    )

    val productionDefault: DetectorExecutionBackend
        get() = decision.productionDefault

    val nextDefaultCandidate: DetectorExecutionBackend
        get() = decision.nextDefaultCandidate

    fun roleFor(backend: DetectorExecutionBackend): DetectorBackendRole = when (backend) {
        decision.productionDefault -> DetectorBackendRole.PRODUCTION_DEFAULT
        decision.nextDefaultCandidate -> DetectorBackendRole.NEXT_DEFAULT_CANDIDATE
        else -> DetectorBackendRole.BENCHMARK_COMPARATOR
    }

    fun isProductionAuthorized(backend: DetectorExecutionBackend): Boolean =
        backend == productionDefault

    fun requireProductionAuthorized(backend: DetectorExecutionBackend) {
        require(isProductionAuthorized(backend)) {
            "$backend is not production-authorized by $POLICY_ID; " +
                "candidatePromotionReady=${decision.candidatePromotionReady}"
        }
    }

    fun requireBenchmarkInjectionAuthorized(packageName: String) {
        require(packageName == BENCHMARK_PACKAGE) {
            "External detector backend injection is restricted to $BENCHMARK_PACKAGE"
        }
    }
}
