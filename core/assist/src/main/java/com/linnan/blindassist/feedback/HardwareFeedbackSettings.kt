package com.linnan.blindassist.feedback

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.localization.AppLanguage

/** Output preferences only: these never modify detector thresholds or its verdict. */
data class HardwareFeedbackSettings(
    val profile: AlertProfile = AlertProfile.STANDARD,
    val style: SpeechStyle = SpeechStyle.STANDARD,
    val strength: VibrationStrength = VibrationStrength.STANDARD,
    val language: AppLanguage = AppLanguage.ZH,
    val care: Boolean = false,
) {
    val reminderIntervalMs: Long get() = when (profile) {
        AlertProfile.QUIET -> 20_000L
        AlertProfile.STANDARD -> 12_000L
        AlertProfile.SENSITIVE -> 6_000L
    }

    fun text(event: HardwareDemoFeedbackEvent): String {
        val english = language == AppLanguage.EN
        val base = if (!english) event.spokenText else when (event) {
            HardwareDemoFeedbackEvent.OBSTACLE -> "Possible obstacle ahead"
            HardwareDemoFeedbackEvent.CLOSER -> "The measured distance is decreasing. Take care"
            HardwareDemoFeedbackEvent.REMINDER -> "A possible obstacle remains ahead"
            HardwareDemoFeedbackEvent.CONNECTION_LOST -> "Range data interrupted. Assessment unavailable"
            HardwareDemoFeedbackEvent.RECOVERED -> "Range data restored"
            HardwareDemoFeedbackEvent.RECOVERED_OBSTACLE -> "Range data restored. Possible obstacle ahead"
            HardwareDemoFeedbackEvent.CAMERA_ONLY -> "Camera unavailable. Range sensing only"
            HardwareDemoFeedbackEvent.CAMERA_RECOVERED -> "Camera restored"
        }
        // Connection information must remain complete even in brief mode.
        return when {
            style == SpeechStyle.BRIEF && event == HardwareDemoFeedbackEvent.CLOSER ->
                if (english) "Distance decreasing" else "测距变近"
            style == SpeechStyle.BRIEF && event == HardwareDemoFeedbackEvent.REMINDER ->
                if (english) "Possible obstacle remains" else "仍可能有障碍"
            style == SpeechStyle.DETAILED && event in setOf(HardwareDemoFeedbackEvent.OBSTACLE,
                HardwareDemoFeedbackEvent.REMINDER, HardwareDemoFeedbackEvent.RECOVERED_OBSTACLE) ->
                base + if (english) ". Check your surroundings" else "，请确认周围环境"
            else -> base
        }
    }

    fun vibrationPattern(event: HardwareDemoFeedbackEvent): LongArray {
        val raw = when (event) {
            HardwareDemoFeedbackEvent.CONNECTION_LOST -> longArrayOf(0, 250, 120, 250)
            HardwareDemoFeedbackEvent.CLOSER -> longArrayOf(0, 120, 80, 120)
            HardwareDemoFeedbackEvent.RECOVERED -> longArrayOf(0, 80)
            else -> longArrayOf(0, 180)
        }
        return LongArray(raw.size) { if (it % 2 == 1) strength.scaleDuration(raw[it]) else raw[it] }
    }

    fun vibrationAmplitudes(pattern: LongArray): IntArray =
        IntArray(pattern.size) { if (it % 2 == 0) 0 else strength.amplitude }
}
