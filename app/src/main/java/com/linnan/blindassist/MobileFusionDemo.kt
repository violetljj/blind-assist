package com.linnan.blindassist

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import java.util.Locale
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** Demo-only geometry and gating for the fusion page (EXPLORE evidence, not a frozen version).
 *
 * - Camera<-ToF pitch set to 0 while keeping yaw/roll (CAMERA_PITCH_EXTRINSIC_DEV_20261014).
 * - v7w walk/still warning distance: RGB in-box median depth and ToF in-box distance must be within
 *   1.8 m while walking or 0.8 m while still (TASK_METRIC_SPEED_DEV_20261014). A ToF alarm with no
 *   in-box firmware zone keeps its alarm, because the model can see what firmware ranging misses.
 */
internal object MobileFusionDemoRules {
    const val WALK_LIMIT_M = 1.8
    const val STILL_LIMIT_M = 0.8

    /** Rebuilds R = Ry(yaw) Rx(pitch) Rz(roll) with pitch = 0, as pitchfix_rerun_dev_20261014.py. */
    fun withoutPitch(cameraFromTof: DoubleArray): DoubleArray {
        require(cameraFromTof.size == 9)
        val yaw = atan2(cameraFromTof[2], cameraFromTof[8])
        val roll = atan2(cameraFromTof[3], cameraFromTof[4])
        return euler(yaw, 0.0, roll)
    }

    fun pitchDeg(cameraFromTof: DoubleArray) = Math.toDegrees(asin(-cameraFromTof[5].coerceIn(-1.0, 1.0)))

    fun euler(yaw: Double, pitch: Double, roll: Double): DoubleArray {
        val ry = doubleArrayOf(cos(yaw), 0.0, sin(yaw), 0.0, 1.0, 0.0, -sin(yaw), 0.0, cos(yaw))
        val rx = doubleArrayOf(1.0, 0.0, 0.0, 0.0, cos(pitch), -sin(pitch), 0.0, sin(pitch), cos(pitch))
        val rz = doubleArrayOf(cos(roll), -sin(roll), 0.0, sin(roll), cos(roll), 0.0, 0.0, 0.0, 1.0)
        return multiply(multiply(ry, rx), rz)
    }

    fun limitM(walking: Boolean) = if (walking) WALK_LIMIT_M else STILL_LIMIT_M

    fun keepRgb(alarm: Boolean, medianDepthM: Double, limitM: Double) =
        alarm && medianDepthM.isFinite() && medianDepthM <= limitM

    fun keepTof(alarm: Boolean, depthM: Double, limitM: Double) =
        alarm && (!depthM.isFinite() || depthM <= limitM)

    private fun multiply(a: DoubleArray, b: DoubleArray) =
        DoubleArray(9) { i -> (0..2).sumOf { k -> a[i / 3 * 3 + k] * b[k * 3 + i % 3] } }
}

/** Walking when the 2 s accelerometer-magnitude std reaches 0.015 g (own_motion_rules_dev_20261013).
 * Receipt-time window; too few samples count as still, which only shortens the warning distance. */
internal class MobileGaitDetector(private val windowNs: Long = 2_000_000_000L, private val minStdG: Double = .015) {
    private val times = ArrayDeque<Long>()
    private val magnitudes = ArrayDeque<Double>()

    @Synchronized fun add(receiptNs: Long, accelG: List<Double>) {
        if (accelG.size != 3 || accelG.any { !it.isFinite() }) return
        times.addLast(receiptNs); magnitudes.addLast(sqrt(accelG.sumOf { it * it }))
        while (times.isNotEmpty() && receiptNs - times.first() > windowNs) { times.removeFirst(); magnitudes.removeFirst() }
    }

    @Synchronized fun walking(nowNs: Long): Boolean {
        if (times.size < 50 || nowNs - times.last() > 500_000_000L || times.last() - times.first() < windowNs * 3 / 4) return false
        val mean = magnitudes.average()
        return sqrt(magnitudes.sumOf { (it - mean) * (it - mean) } / magnitudes.size) >= minStdG
    }

    @Synchronized fun reset() { times.clear(); magnitudes.clear() }
}

/** Event speech/vibration for the fusion demo: first confirmed alarm immediately, repeat while it
 * persists, re-arm after one second without an alarm. Never announces "clear" or "safe". */
internal class MobileFusionAnnouncer(context: Context) {
    private val appContext = context.applicationContext
    @Volatile private var ready = false
    /** null while the engine initialises; false when Chinese speech cannot be produced. */
    @Volatile var speechAvailable: Boolean? = null
        private set
    private lateinit var tts: TextToSpeech
    init {
        tts = TextToSpeech(appContext) { status ->
            val language = if (status == TextToSpeech.SUCCESS)
                runCatching { tts.setLanguage(Locale.SIMPLIFIED_CHINESE) }.getOrDefault(TextToSpeech.LANG_NOT_SUPPORTED)
            else TextToSpeech.LANG_NOT_SUPPORTED
            ready = status == TextToSpeech.SUCCESS && language >= TextToSpeech.LANG_AVAILABLE
            speechAvailable = ready
        }
    }
    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S)
        appContext.getSystemService(VibratorManager::class.java)?.defaultVibrator
    else @Suppress("DEPRECATION") appContext.getSystemService(Vibrator::class.java)

    var speech = true
    var vibration = true
    var repeatMs = 6_000L
    private var lastAlertMs = -1L
    private var lastKind = ""
    private var quietSinceMs = -1L

    /** Returns the spoken text when an alert was issued. */
    fun update(nowMs: Long, head: Boolean, body: Boolean, distanceM: Double): String? {
        if (!head && !body) {
            if (quietSinceMs < 0) quietSinceMs = nowMs
            if (nowMs - quietSinceMs >= 1_000) { lastAlertMs = -1; lastKind = "" }
            return null
        }
        quietSinceMs = -1
        val kind = when { head && body -> "前方有障碍，注意头部"; head -> "注意头部前方障碍"; else -> "前方有障碍" }
        val due = lastAlertMs < 0 || nowMs - lastAlertMs >= repeatMs || (kind != lastKind && nowMs - lastAlertMs >= 1_500)
        if (!due) return null
        lastAlertMs = nowMs; lastKind = kind
        val text = if (distanceM.isFinite()) "%s，约%.1f米".format(kind, distanceM) else kind
        if (speech && ready) tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "fusion-alert")
        if (vibration && vibrator?.hasVibrator() == true) {
            val pattern = if (head) longArrayOf(0, 120, 90, 120, 90, 120) else longArrayOf(0, 260)
            vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
        }
        return text
    }

    fun silence() { runCatching { tts.stop() }; vibrator?.cancel(); lastAlertMs = -1; lastKind = ""; quietSinceMs = -1 }
    fun shutdown() { silence(); tts.shutdown() }
}
