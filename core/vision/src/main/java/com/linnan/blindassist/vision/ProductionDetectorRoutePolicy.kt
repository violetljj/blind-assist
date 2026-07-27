package com.linnan.blindassist.vision

data class ProductionDeviceProfile(
    val socModel: String,
    val supportedAbis: List<String>
)

data class ProductionDetectorRouteDecision(
    val backend: DetectorExecutionBackend,
    val reason: String
)

/**
 * Pure, unit-testable device scope for the production detector route.
 *
 * A model name alone is never enough: QNN is attempted only for the frozen SoC
 * scope and arm64 ABI. The provider still performs a live HTP FP16 capability
 * check before creating the delegate.
 */
object ProductionDetectorRoutePolicy {
    const val SUPPORTED_SOC = "SM8650"
    const val REQUIRED_ABI = "arm64-v8a"

    fun isQnnProbeEligible(profile: ProductionDeviceProfile): Boolean =
        profile.socModel.equals(SUPPORTED_SOC, ignoreCase = true) &&
            REQUIRED_ABI in profile.supportedAbis

    fun decide(
        profile: ProductionDeviceProfile,
        qnnHtpFp16Available: Boolean
    ): ProductionDetectorRouteDecision = when {
        !profile.socModel.equals(SUPPORTED_SOC, ignoreCase = true) ->
            cpu("unsupported_soc:${profile.socModel.ifBlank { "unknown" }}")
        REQUIRED_ABI !in profile.supportedAbis ->
            cpu("required_abi_missing:$REQUIRED_ABI")
        !qnnHtpFp16Available ->
            cpu("qnn_htp_fp16_capability_unavailable")
        else -> ProductionDetectorRouteDecision(
            backend = DetectorExecutionBackend.QUALCOMM_QNN_HTP,
            reason = "supported_soc_abi_and_live_qnn_htp_fp16_capability"
        )
    }

    private fun cpu(reason: String) = ProductionDetectorRouteDecision(
        backend = DetectorExecutionBackend.CPU_XNNPACK,
        reason = reason
    )
}
