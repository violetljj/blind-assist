package com.linnan.blindassist.feedback

import android.Manifest
import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import androidx.annotation.RequiresPermission
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.util.FatalThrowables
import java.util.Locale

data class FeedbackRuntimeSettings(
    val speechEnabled: Boolean = true,
    val vibrationEnabled: Boolean = true,
    val speechStyle: SpeechStyle = SpeechStyle.STANDARD,
    val vibrationStrength: VibrationStrength = VibrationStrength.STANDARD,
    val appLanguage: AppLanguage = AppLanguage.ZH
)

fun interface FeedbackClock {
    fun nowMs(): Long
}

interface SpeechOutput {
    val ready: Boolean
    fun setLanguage(language: AppLanguage)
    fun speak(message: String, utteranceId: String): Boolean
    fun shutdown()
}

fun interface HapticOutput {
    fun vibrate(plan: FeedbackPlan): Boolean
}

class FeedbackController constructor(
    private val speechOutput: SpeechOutput,
    private val hapticOutput: HapticOutput,
    private val clock: FeedbackClock,
    initialSettings: FeedbackRuntimeSettings
) : FeedbackGateway {
    @Volatile
    private var settings = initialSettings

    private val lifecycleLock = Any()
    private var shutdown = false
    private val lastAlertAt = mutableMapOf<AlertKey, Long>()
    private val fatigueController = FeedbackFatigueController()

    constructor(context: Context) : this(
        speechOutput = AndroidSpeechOutput(context.applicationContext),
        hapticOutput = AndroidHapticOutput(context.applicationContext),
        clock = FeedbackClock { SystemClock.elapsedRealtime() },
        initialSettings = FeedbackRuntimeSettings()
    )

    init {
        speechOutput.setLanguage(initialSettings.appLanguage)
    }

    fun applySettings(newSettings: FeedbackRuntimeSettings) {
        synchronized(lifecycleLock) {
            if (shutdown) return
            settings = newSettings
            speechOutput.setLanguage(newSettings.appLanguage)
        }
    }

    override fun resetSession() {
        synchronized(lifecycleLock) {
            if (shutdown) return
            lastAlertAt.clear()
            fatigueController.reset()
        }
    }

    fun notify(risk: RiskResult): FeedbackDecision {
        return notify(risk, AlertProfile.STANDARD)
    }

    fun notify(risk: RiskResult, profile: AlertProfile): FeedbackDecision {
        return notify(risk, profile, AssistScenario.GENERAL)
    }

    override fun notify(risk: RiskResult, profile: AlertProfile, scenario: AssistScenario): FeedbackDecision {
        synchronized(lifecycleLock) {
            return notifyLocked(risk, profile, scenario)
        }
    }

    private fun notifyLocked(risk: RiskResult, profile: AlertProfile, scenario: AssistScenario): FeedbackDecision {
        if (shutdown) {
            return FeedbackDecision(null, triggered = false, reason = FeedbackReason.FEEDBACK_UNAVAILABLE)
        }
        val snapshot = settings
        val plan = planFor(risk, profile, snapshot.vibrationStrength, scenario)
            ?: return FeedbackDecision(null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)

        val now = clock.nowMs()
        val alertKey = AlertKey(risk.direction, risk.proximity)
        val last = lastAlertAt[alertKey]
        val effectiveCooldownMs = fatigueController.effectiveCooldownMs(risk, plan.cooldownMs, now)
        if (last != null && now - last < effectiveCooldownMs) {
            return FeedbackDecision(plan, triggered = false, reason = FeedbackReason.COOLDOWN)
        }

        val speechTriggered = if (snapshot.speechEnabled && speechOutput.ready) {
            speechOutput.speak(
                message = snapshot.speechStyle.messageFor(risk, snapshot.appLanguage),
                utteranceId = "risk-$now"
            )
        } else {
            false
        }
        val vibrationTriggered = if (snapshot.vibrationEnabled) {
            hapticOutput.vibrate(plan)
        } else {
            false
        }

        if (!speechTriggered && !vibrationTriggered) {
            return FeedbackDecision(
                plan = plan,
                triggered = false,
                reason = FeedbackReason.FEEDBACK_UNAVAILABLE
            )
        }

        lastAlertAt[alertKey] = now
        fatigueController.recordTriggered(risk, now)
        return FeedbackDecision(
            plan = plan,
            triggered = true,
            reason = FeedbackReason.TRIGGERED,
            speechTriggered = speechTriggered,
            vibrationTriggered = vibrationTriggered
        )
    }

    fun shutdown() {
        synchronized(lifecycleLock) {
            if (shutdown) return
            shutdown = true
            lastAlertAt.clear()
            fatigueController.reset()
            speechOutput.shutdown()
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

private class AndroidSpeechOutput(context: Context) : SpeechOutput, TextToSpeech.OnInitListener {
    private val appContext = context.applicationContext
    private val tts = TextToSpeech(appContext, this)

    @Volatile
    override var ready: Boolean = false
        private set

    @Volatile
    private var language: AppLanguage = AppLanguage.ZH

    override fun onInit(status: Int) {
        ready = status == TextToSpeech.SUCCESS
        if (ready) {
            applyLanguage()
            tts.setSpeechRate(1.08f)
        }
    }

    override fun setLanguage(language: AppLanguage) {
        this.language = language
        if (ready) {
            applyLanguage()
        }
    }

    override fun speak(message: String, utteranceId: String): Boolean {
        if (!ready) return false
        return try {
            tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, utteranceId) == TextToSpeech.SUCCESS
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            false
        }
    }

    override fun shutdown() {
        tts.stop()
        tts.shutdown()
    }

    private fun applyLanguage() {
        tts.language = if (language == AppLanguage.EN) Locale.US else Locale.CHINA
    }
}

private class AndroidHapticOutput(context: Context) : HapticOutput {
    private val appContext = context.applicationContext
    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val manager = appContext.getSystemService(VibratorManager::class.java)
        manager?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        appContext.getSystemService(Vibrator::class.java)
    }

    @RequiresPermission(Manifest.permission.VIBRATE)
    override fun vibrate(plan: FeedbackPlan): Boolean {
        val vib = vibrator ?: return false
        if (!vib.hasVibrator()) return false
        return try {
            vib.vibrate(VibrationEffect.createOneShot(plan.vibrationMs, plan.amplitude))
            true
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            false
        }
    }
}
