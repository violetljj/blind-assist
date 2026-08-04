package com.linnan.blindassist.ustrfbenchmark

enum class CapturePhase(val frameTarget: Int, val label: String) {
    P0(120, "P0 · 来源与几何预检"),
    R2(25, "R2 · 固定复核"),
}

data class CaptureFormState(
    val sessionId: String,
    val phase: CapturePhase = CapturePhase.P0,
    val cameraHeightM: String = "",
    val cameraHeightUncertaintyM: String = "0.01",
    val mountProfileId: String = "",
    val referenceDisplayName: String? = null,
) {
    fun validationProblems(): List<String> = buildList {
        if (!sessionId.matches(Regex("[A-Za-z0-9][A-Za-z0-9._-]{2,63}"))) {
            add("Session ID 需为 3–64 位字母、数字、点、横线或下划线")
        }
        val height = cameraHeightM.toDoubleOrNull()
        if (height == null || height !in 0.8..2.2) add("相机光心高度必须在 0.80–2.20 m")
        val uncertainty = cameraHeightUncertaintyM.toDoubleOrNull()
        if (uncertainty == null || uncertainty !in 0.0..0.02) add("量高不确定度必须在 0–0.02 m")
        if (mountProfileId.trim().length < 3) add("请填写可复现的支架编号")
        if (referenceDisplayName.isNullOrBlank()) add("请选择独立卷尺/激光参考清单")
    }

    val canStart: Boolean get() = validationProblems().isEmpty()
}

sealed interface CaptureRunState {
    data object Idle : CaptureRunState
    data class Preparing(val message: String) : CaptureRunState
    data class Capturing(val captured: Int, val target: Int) : CaptureRunState
    data class Complete(val sessionDirectory: String, val captured: Int) : CaptureRunState
    data class Hold(val reason: String) : CaptureRunState
}
