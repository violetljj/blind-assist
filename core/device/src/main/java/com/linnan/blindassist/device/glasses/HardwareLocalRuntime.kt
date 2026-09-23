package com.linnan.blindassist.device.glasses

import android.graphics.Bitmap
import android.os.SystemClock
import com.linnan.blindassist.risk.DemoTofDecision
import com.linnan.blindassist.risk.HardwareCalibratedA
import com.linnan.blindassist.risk.HardwareLocalFeatures
import com.linnan.blindassist.risk.HardwareLocalGeometry
import com.linnan.blindassist.risk.HardwareLocalModel

/** Caller runs on a background dispatcher, one evaluation at a time; never queues frames.
 * Source geometry is nominal, not calibrated. Frozen simulation performance is not inherited.
 */
class HardwareLocalRuntime(private val model: HardwareLocalModel, private val clock: () -> Long = SystemClock::elapsedRealtime) {
    private var key: Pair<String, Long?>? = null
    private var cached: DemoTofDecision? = null
    private var cachedStatus = ""
    private var cachedAt = 0L
    private var cachedCameraAge = 0L

    fun evaluate(input: HardwareDemoSnapshot, observedAtMs: Long? = null): HardwareDemoSnapshot {
        if (input.mode != "live") return input
        val start = clock()
        if (!input.tofUsable) {
            key = null; cached = null
            return input.copy(status = "A+LOCAL 实验 · ToF 不可用 · ${input.status}")
        }
        val tokens = HardwareLocalGeometry.tokens(input.cells)
        val a = HardwareCalibratedA.evaluate(tokens)
        val zones = a.triggeringZones.map(HardwareLocalGeometry::rawZone)
        val nearest = if (a.alert) input.cells.filter { it.zone in zones }.minOfOrNull { it.rangeMm }?.toInt() else null
        val aDecision = DemoTofDecision(a.alert, if (!a.unknown) "A_SUPPORTED" else if (a.alert) "A_AMBIGUOUS" else "UNKNOWN",
            nearest, if (a.alert) zones else emptyList(), a.validZones)
        val queuedMs = observedAtMs?.let { (start - it).coerceAtLeast(0) } ?: 0L
        val cameraAge = input.cameraAgeMs?.plus(queuedMs)
        val tofAge = input.tofAgeMs?.plus(queuedMs)
        // Age upper bounds cannot establish exposure synchrony. This is only a coarse skew guard.
        val pairUsable = input.cameraUsable && input.image != null && cameraAge != null && tofAge != null &&
            kotlin.math.abs(cameraAge - tofAge) <= 250 && a.validZones > 0
        if (!pairUsable) {
            key = null; cached = null
            return input.copy(decision = aDecision, status = "A+LOCAL 实验 · LOCAL 暂停（画面/时龄/测距不足），仅 A · ${input.status}")
        }
        val nextKey = input.runId to input.tofSequence
        if (nextKey == key && cached != null && start - cachedAt + cachedCameraAge <= 750) {
            return input.copy(decision = cached!!, status = cachedStatus + " · " + input.status,
                cameraAgeMs = maxOf(requireNotNull(cameraAge), start - cachedAt + cachedCameraAge))
        }
        val pixels = modelPixels(requireNotNull(input.image))
        val features = HardwareLocalFeatures.extract(pixels, tokens, intArrayOf(1, 4))
        // Original alert readout uses centre LOW/HEAD queries 1 and 4 only.
        val probability = maxOf(model.predict(features[0]), model.predict(features[1]))
        val local = probability >= model.threshold
        val elapsed = clock() - start
        val result = DemoTofDecision(a.alert || local,
            if (a.alert) aDecision.state else if (local) "LOCAL_EXPERIMENT_UNKNOWN" else "UNKNOWN",
            nearest, if (a.alert) zones else emptyList(), a.validZones)
        val status = "A+LOCAL 实验 · 未标定 · A=${if(a.alert) "触发" else "未触发"} / LOCAL=${if(local) "触发" else "未触发"} · ${elapsed}ms"
        if (requireNotNull(tofAge) + elapsed > 750) {
            key = null; cached = null
            return input.copy(decision = DemoTofDecision(false, "UNKNOWN", null, emptyList(), 0),
                tofUsable = false, liveUsable = false, status = "A+LOCAL 处理后测距过期 · UNKNOWN")
        }
        if (requireNotNull(cameraAge) + elapsed > 750) {
            key = null; cached = null
            return input.copy(decision = aDecision, cameraUsable = false, image = null, liveUsable = false,
                status = "A+LOCAL 处理后画面过期 · 仅 A")
        }
        key = nextKey; cached = result; cachedAt = start; cachedCameraAge = cameraAge
        cachedStatus = status
        // Ages stay anchored to call entry; afterProcessing adds total dispatch+compute time once.
        return input.copy(decision = result, status = status + " · " + input.status)
    }

    companion object {
        /** Recheck after dispatcher return: scheduling delay also consumes freshness. */
        fun afterProcessing(raw: HardwareDemoSnapshot, result: HardwareDemoSnapshot, elapsed: Long): HardwareDemoSnapshot {
            if (raw.mode != "live") return result
            val tofFresh = result.tofUsable && (raw.tofAgeMs?.plus(elapsed)?.let { it <= 750 } ?: true)
            val cameraAge = listOfNotNull(raw.cameraAgeMs, result.cameraAgeMs).maxOrNull()?.plus(elapsed)
            val cameraFresh = result.cameraUsable && (cameraAge?.let { it <= 750 } ?: true)
            val rejectedLocal = !cameraFresh && result.decision.state == "LOCAL_EXPERIMENT_UNKNOWN"
            return result.copy(tofUsable = tofFresh, cameraUsable = cameraFresh, liveUsable = tofFresh && cameraFresh,
                tofAgeMs = raw.tofAgeMs?.plus(elapsed), cameraAgeMs = cameraAge,
                image = result.image.takeIf { cameraFresh },
                status = if (rejectedLocal) "A+LOCAL 画面过期 · LOCAL 已暂停 · UNKNOWN" else result.status,
                decision = if (!tofFresh || rejectedLocal) DemoTofDecision(false, "UNKNOWN", null, emptyList(),
                    if (tofFresh) result.decision.validZones else 0) else result.decision)
        }

        /** Centre 16:9 crop; area average, no stretching of VGA into a different aspect ratio.
         * Current VGA input gives exact 2x2 area pooling. Other sizes are rejected explicitly.
         */
        fun modelPixels(image: Bitmap): IntArray {
            require(image.width == 640 && image.height in listOf(360, 480)) { "LOCAL 需要 640×480 或 640×360 画面" }
            val source = IntArray(640 * 360)
            image.getPixels(source, 0, 640, 0, (image.height - 360) / 2, 640, 360)
            return IntArray(320 * 180) { i ->
                val offset = (i / 320 * 2) * 640 + i % 320 * 2
                fun mean(shift: Int) = (((source[offset] ushr shift) and 255) +
                    ((source[offset + 1] ushr shift) and 255) + ((source[offset + 640] ushr shift) and 255) +
                    ((source[offset + 641] ushr shift) and 255) + 2) / 4
                (255 shl 24) or (mean(16) shl 16) or (mean(8) shl 8) or mean(0)
            }
        }
    }
}
