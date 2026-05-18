package com.linnan.blindassist.feedback

enum class VibrationStrength(
    val storageValue: String,
    val displayName: String,
    val description: String,
    private val durationMultiplier: Float,
    val amplitude: Int
) {
    SOFT("soft", "轻柔", "降低震动时长和强度", 0.75f, 96),
    STANDARD("standard", "标准", "使用默认触觉反馈", 1f, -1),
    STRONG("strong", "强", "增强近处和迫近提醒", 1.25f, 255);

    fun scaleDuration(durationMs: Long): Long {
        return (durationMs * durationMultiplier).toLong().coerceAtLeast(60L)
    }

    companion object {
        fun fromStorageValue(value: String?): VibrationStrength {
            return values().firstOrNull { it.storageValue == value } ?: STANDARD
        }
    }
}
