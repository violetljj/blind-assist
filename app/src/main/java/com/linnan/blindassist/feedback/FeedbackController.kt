package com.linnan.blindassist.feedback

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import com.linnan.blindassist.alert.AlertPolicy
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import java.util.Locale

class FeedbackController(context: Context) : TextToSpeech.OnInitListener {
    var speechEnabled: Boolean = true
    var vibrationEnabled: Boolean = true

    private val appContext = context.applicationContext
    private val lastAlertAt = mutableMapOf<AlertKey, Long>()
    private var ttsReady = false
    private val tts = TextToSpeech(appContext, this)

    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val manager = appContext.getSystemService(VibratorManager::class.java)
        manager?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        appContext.getSystemService(Vibrator::class.java)
    }

    override fun onInit(status: Int) {
        ttsReady = status == TextToSpeech.SUCCESS
        if (ttsReady) {
            tts.language = Locale.CHINA
            tts.setSpeechRate(1.08f)
        }
    }

    fun notify(risk: RiskResult) {
        notify(risk, AlertProfile.STANDARD)
    }

    fun notify(risk: RiskResult, profile: AlertProfile): FeedbackDecision {
        val plan = planFor(risk, profile)
            ?: return FeedbackDecision(null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)

        val now = System.currentTimeMillis()
        val alertKey = AlertKey(risk.direction, risk.proximity)
        val last = lastAlertAt[alertKey] ?: 0L
        if (now - last < plan.cooldownMs) {
            return FeedbackDecision(plan, triggered = false, reason = FeedbackReason.COOLDOWN)
        }

        if (!speechEnabled && !vibrationEnabled) {
            return FeedbackDecision(plan, triggered = false, reason = FeedbackReason.SPEECH_DISABLED)
        }
        lastAlertAt[alertKey] = now

        if (speechEnabled && ttsReady) {
            tts.speak(risk.message, TextToSpeech.QUEUE_FLUSH, null, "risk-${now}")
        }
        if (vibrationEnabled) {
            vibrate(plan)
        }
        return FeedbackDecision(plan, triggered = true, reason = FeedbackReason.TRIGGERED)
    }

    fun shutdown() {
        tts.stop()
        tts.shutdown()
    }

    private fun vibrate(plan: FeedbackPlan) {
        val vib = vibrator ?: return
        if (!vib.hasVibrator()) return

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vib.vibrate(VibrationEffect.createOneShot(plan.vibrationMs, plan.amplitude))
        } else {
            @Suppress("DEPRECATION")
            vib.vibrate(plan.vibrationMs)
        }
    }

    private data class AlertKey(
        val direction: RiskDirection,
        val proximity: ProximityBand
    )

    companion object {
        const val STANDARD_NEAR_ALERT_COOLDOWN_MS = 1500L
        const val STANDARD_CRITICAL_ALERT_COOLDOWN_MS = 850L
        const val STANDARD_NEAR_VIBRATION_MS = 160L
        const val STANDARD_CRITICAL_VIBRATION_MS = 420L

        internal fun planFor(risk: RiskResult, profile: AlertProfile = AlertProfile.STANDARD): FeedbackPlan? {
            val policy = AlertPolicy.forProfile(profile)
            return when {
                risk.proximity == ProximityBand.CRITICAL && risk.level == RiskLevel.HIGH -> {
                    FeedbackPlan(policy.criticalCooldownMs, policy.criticalVibrationMs, VibrationEffect.DEFAULT_AMPLITUDE)
                }
                risk.proximity == ProximityBand.NEAR &&
                    (risk.level == RiskLevel.HIGH || risk.level == RiskLevel.MEDIUM) -> {
                    FeedbackPlan(policy.nearCooldownMs, policy.nearVibrationMs, VibrationEffect.DEFAULT_AMPLITUDE)
                }
                else -> null
            }
        }
    }
}

data class FeedbackPlan(
    val cooldownMs: Long,
    val vibrationMs: Long,
    val amplitude: Int
)

data class FeedbackDecision(
    val plan: FeedbackPlan?,
    val triggered: Boolean,
    val reason: FeedbackReason
) {
    fun withDisplayReason(displayReason: FeedbackReason): FeedbackDecision {
        return if (triggered) this else copy(reason = displayReason)
    }
}

enum class FeedbackReason(val displayText: String) {
    TRIGGERED("已触发反馈"),
    HELD_ALERT("提醒保持"),
    DISTANCE_TOO_FAR("距离较远"),
    UNSTABLE_RISK("风险未稳定"),
    COOLDOWN("冷却中"),
    SPEECH_DISABLED("语音关闭"),
    VIBRATION_DISABLED("震动关闭"),
    NO_FEEDBACK_RISK("无可反馈风险")
}
