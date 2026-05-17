package com.linnan.blindassist.feedback

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
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
        val plan = planFor(risk) ?: return

        val now = System.currentTimeMillis()
        val alertKey = AlertKey(risk.direction, risk.proximity)
        val last = lastAlertAt[alertKey] ?: 0L
        if (now - last < plan.cooldownMs) return
        lastAlertAt[alertKey] = now

        if (speechEnabled && ttsReady) {
            tts.speak(risk.message, TextToSpeech.QUEUE_FLUSH, null, "risk-${now}")
        }
        if (vibrationEnabled) {
            vibrate(plan)
        }
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
        const val NEAR_ALERT_COOLDOWN_MS = 1500L
        const val CRITICAL_ALERT_COOLDOWN_MS = 850L
        const val NEAR_VIBRATION_MS = 160L
        const val CRITICAL_VIBRATION_MS = 420L

        internal fun planFor(risk: RiskResult): FeedbackPlan? {
            return when {
                risk.proximity == ProximityBand.CRITICAL && risk.level == RiskLevel.HIGH -> {
                    FeedbackPlan(CRITICAL_ALERT_COOLDOWN_MS, CRITICAL_VIBRATION_MS, VibrationEffect.DEFAULT_AMPLITUDE)
                }
                risk.proximity == ProximityBand.NEAR &&
                    (risk.level == RiskLevel.HIGH || risk.level == RiskLevel.MEDIUM) -> {
                    FeedbackPlan(NEAR_ALERT_COOLDOWN_MS, NEAR_VIBRATION_MS, VibrationEffect.DEFAULT_AMPLITUDE)
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
