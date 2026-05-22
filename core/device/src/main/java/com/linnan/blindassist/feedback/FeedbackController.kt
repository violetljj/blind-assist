package com.linnan.blindassist.feedback

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskResult
import java.util.Locale

class FeedbackController(context: Context) : TextToSpeech.OnInitListener, FeedbackGateway {
    var speechEnabled: Boolean = true
    var vibrationEnabled: Boolean = true
    var speechStyle: SpeechStyle = SpeechStyle.STANDARD
    var vibrationStrength: VibrationStrength = VibrationStrength.STANDARD
    var appLanguage: AppLanguage = AppLanguage.ZH
        set(value) {
            field = value
            applyTtsLanguage()
        }

    private val appContext = context.applicationContext
    private val lastAlertAt = mutableMapOf<AlertKey, Long>()
    private val fatigueController = FeedbackFatigueController()
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
            applyTtsLanguage()
            tts.setSpeechRate(1.08f)
        }
    }

    fun notify(risk: RiskResult) {
        notify(risk, AlertProfile.STANDARD)
    }

    fun notify(risk: RiskResult, profile: AlertProfile): FeedbackDecision {
        return notify(risk, profile, AssistScenario.GENERAL)
    }

    override fun notify(risk: RiskResult, profile: AlertProfile, scenario: AssistScenario): FeedbackDecision {
        val plan = planFor(risk, profile, vibrationStrength, scenario)
            ?: return FeedbackDecision(null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)

        val now = System.currentTimeMillis()
        val alertKey = AlertKey(risk.direction, risk.proximity)
        val last = lastAlertAt[alertKey] ?: 0L
        val effectiveCooldownMs = fatigueController.effectiveCooldownMs(risk, plan.cooldownMs, now)
        if (now - last < effectiveCooldownMs) {
            return FeedbackDecision(plan, triggered = false, reason = FeedbackReason.COOLDOWN)
        }

        if (!speechEnabled && !vibrationEnabled) {
            return FeedbackDecision(plan, triggered = false, reason = FeedbackReason.SPEECH_DISABLED)
        }
        lastAlertAt[alertKey] = now

        var speechTriggered = false
        var vibrationTriggered = false
        if (speechEnabled && ttsReady) {
            tts.speak(speechStyle.messageFor(risk, appLanguage), TextToSpeech.QUEUE_FLUSH, null, "risk-${now}")
            speechTriggered = true
        }
        if (vibrationEnabled) {
            vibrate(plan)
            vibrationTriggered = true
        }
        return FeedbackDecision(
            plan = plan,
            triggered = true,
            reason = FeedbackReason.TRIGGERED,
            speechTriggered = speechTriggered,
            vibrationTriggered = vibrationTriggered
        ).also {
            fatigueController.recordTriggered(risk, now)
        }
    }

    fun shutdown() {
        fatigueController.reset()
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

    private fun applyTtsLanguage() {
        if (!ttsReady) return
        tts.language = if (appLanguage == AppLanguage.EN) Locale.US else Locale.CHINA
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

        fun planFor(
            risk: RiskResult,
            profile: AlertProfile = AlertProfile.STANDARD,
            vibrationStrength: VibrationStrength = VibrationStrength.STANDARD,
            scenario: AssistScenario = AssistScenario.GENERAL
        ): FeedbackPlan? {
            return FeedbackPlanner.planFor(risk, profile, vibrationStrength, scenario)
        }
    }
}
