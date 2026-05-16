package com.linnan.blindassist.feedback

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import java.util.Locale

class FeedbackController(context: Context) : TextToSpeech.OnInitListener {
    var speechEnabled: Boolean = true
    var vibrationEnabled: Boolean = true

    private val appContext = context.applicationContext
    private val lastAlertAt = mutableMapOf<RiskDirection, Long>()
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
        if (risk.level != RiskLevel.HIGH && risk.level != RiskLevel.MEDIUM) return

        val now = System.currentTimeMillis()
        val last = lastAlertAt[risk.direction] ?: 0L
        if (now - last < ALERT_COOLDOWN_MS) return
        lastAlertAt[risk.direction] = now

        if (speechEnabled && ttsReady) {
            tts.speak(risk.message, TextToSpeech.QUEUE_FLUSH, null, "risk-${now}")
        }
        if (vibrationEnabled) {
            vibrate(risk.level)
        }
    }

    fun shutdown() {
        tts.stop()
        tts.shutdown()
    }

    private fun vibrate(level: RiskLevel) {
        val vib = vibrator ?: return
        if (!vib.hasVibrator()) return

        val duration = if (level == RiskLevel.HIGH) 360L else 140L
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vib.vibrate(VibrationEffect.createOneShot(duration, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vib.vibrate(duration)
        }
    }

    companion object {
        const val ALERT_COOLDOWN_MS = 1500L
    }
}
