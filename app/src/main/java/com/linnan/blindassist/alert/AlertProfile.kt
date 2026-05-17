package com.linnan.blindassist.alert

enum class AlertProfile(
    val storageValue: String,
    val displayName: String
) {
    QUIET("quiet", "安静"),
    STANDARD("standard", "标准"),
    SENSITIVE("sensitive", "敏感");

    fun next(): AlertProfile {
        return when (this) {
            QUIET -> STANDARD
            STANDARD -> SENSITIVE
            SENSITIVE -> QUIET
        }
    }

    companion object {
        fun fromStorageValue(value: String?): AlertProfile {
            return values().firstOrNull { it.storageValue == value } ?: STANDARD
        }
    }
}

data class AlertPolicy(
    val profile: AlertProfile,
    val mediumConfirmFrames: Int,
    val holdAlertMs: Long,
    val nearCooldownMs: Long,
    val nearVibrationMs: Long,
    val criticalCooldownMs: Long,
    val criticalVibrationMs: Long
) {
    companion object {
        fun forProfile(profile: AlertProfile): AlertPolicy {
            return when (profile) {
                AlertProfile.QUIET -> AlertPolicy(
                    profile = profile,
                    mediumConfirmFrames = 3,
                    holdAlertMs = 450L,
                    nearCooldownMs = 2200L,
                    nearVibrationMs = 100L,
                    criticalCooldownMs = 1200L,
                    criticalVibrationMs = 260L
                )
                AlertProfile.STANDARD -> AlertPolicy(
                    profile = profile,
                    mediumConfirmFrames = 2,
                    holdAlertMs = 600L,
                    nearCooldownMs = 1500L,
                    nearVibrationMs = 160L,
                    criticalCooldownMs = 850L,
                    criticalVibrationMs = 420L
                )
                AlertProfile.SENSITIVE -> AlertPolicy(
                    profile = profile,
                    mediumConfirmFrames = 1,
                    holdAlertMs = 800L,
                    nearCooldownMs = 1000L,
                    nearVibrationMs = 220L,
                    criticalCooldownMs = 650L,
                    criticalVibrationMs = 520L
                )
            }
        }
    }
}
