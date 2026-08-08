package com.linnan.blindassist.ustrfbenchmark

import kotlin.math.abs

enum class CapturePhase(val frameTarget: Int, val label: String) {
    DEV(25, "开发采集 · 25 帧"),
    P0(120, "首次预检 · 120 帧"),
    R2(25, "后续复核 · 25 帧"),
}

enum class MeasurementMethod(
    val label: String,
    val receiptValue: String,
    val suggestedInstrumentErrorCm: String,
) {
    SAMSUNG_QUICK_MEASURE("三星快速测量", "samsung_quick_measure_ar", "5.0"),
    LASER("激光测距仪", "laser_line_of_sight", "0.5"),
    TAPE("卷尺", "tape_horizontal_floor_distance", "1.0"),
}

data class ReferencePoint(
    val id: String,
    val label: String,
    val distanceM: Double,
)

data class CaptureFormState(
    val sessionId: String,
    val phase: CapturePhase = CapturePhase.DEV,
    val mountProfileId: String = "",
    val measurementMethod: MeasurementMethod = MeasurementMethod.LASER,
    val instrumentErrorCm: String = MeasurementMethod.LASER.suggestedInstrumentErrorCm,
    val heightReading1Cm: String = "",
    val heightReading2Cm: String = "",
    val heightReading3Cm: String = "",
    val nearDistanceM: String = "",
    val middleDistanceM: String = "",
    val farDistanceM: String = "",
    val developmentDistanceCm: String = "",
) {
    val heightReadingsCm: List<Double>?
        get() = listOf(heightReading1Cm, heightReading2Cm, heightReading3Cm)
            .map { it.toDoubleOrNull() ?: return null }

    val cameraHeightM: Double?
        get() = if (phase == CapturePhase.DEV) heightReading1Cm.toDoubleOrNull()?.div(100.0)
        else heightReadingsCm?.sorted()?.get(1)?.div(100.0)

    val cameraHeightUncertaintyM: Double?
        get() {
            if (phase == CapturePhase.DEV) return instrumentErrorCm.toDoubleOrNull()?.div(100.0)
            val readings = heightReadingsCm ?: return null
            val medianCm = readings.sorted()[1]
            val instrumentCm = instrumentErrorCm.toDoubleOrNull() ?: return null
            return (readings.maxOf { abs(it - medianCm) } + instrumentCm) / 100.0
        }

    val referencePoints: List<ReferencePoint>?
        get() {
            if (phase == CapturePhase.DEV) {
                val distanceCm = developmentDistanceCm.toDoubleOrNull() ?: return null
                return listOf(ReferencePoint("current", "当前目标", distanceCm / 100.0))
            }
            val distances = listOf(nearDistanceM, middleDistanceM, farDistanceM)
                .map { it.toDoubleOrNull() ?: return null }
            return listOf(
                ReferencePoint("near", "近处标记", distances[0]),
                ReferencePoint("middle", "中间标记", distances[1]),
                ReferencePoint("far", "远处标记", distances[2]),
            )
        }

    fun validationProblems(): List<String> = buildList {
        if (!sessionId.matches(Regex("[A-Za-z0-9][A-Za-z0-9._-]{2,63}"))) add("内部采集编号无效")
        if (mountProfileId.trim().length < 2) add("给固定支架起个名字，例如“三脚架A”")
        if (phase == CapturePhase.DEV) {
            val height = heightReading1Cm.toDoubleOrNull()
            if (height == null || height !in 80.0..220.0) add("填写一次镜头高度（80–220 cm）")
            val distance = developmentDistanceCm.toDoubleOrNull()
            if (distance == null || distance !in 10.0..1000.0) add("填写快速测量显示的距离（10–1000 cm）")
            return@buildList
        }
        val readings = heightReadingsCm
        if (readings == null || readings.any { it !in 80.0..220.0 }) {
            add("请填写三次 80–220 cm 的镜头高度")
        }
        val instrumentCm = instrumentErrorCm.toDoubleOrNull()
        if (instrumentCm == null || instrumentCm !in 0.1..2.0) add("量具误差需在 0.1–2.0 cm")
        val uncertainty = cameraHeightUncertaintyM
        if (uncertainty != null && uncertainty > 0.02) add("三次高度差异过大，请固定支架后重测")
        val points = referencePoints
        if (points == null || points.any { it.distanceM !in 0.3..10.0 }) {
            add("请填写近、中、远三个 0.3–10 m 的实测距离")
        } else if (!(points[0].distanceM < points[1].distanceM && points[1].distanceM < points[2].distanceM)) {
            add("三个距离必须按近 → 中 → 远递增")
        }
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
