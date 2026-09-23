package com.linnan.blindassist.risk

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.tan

data class HardwareCalibratedADecision(
    val alert: Boolean,
    val unknown: Boolean,
    val score: Double,
    val validZones: Int,
    /** Possible geometric supports; these are not independently confirmed obstacles. */
    val triggeringZones: List<Int>,
    val definiteZones: Int,
    val state: String
)

/** Frozen retained-A geometry port. Hardware registration and range conversion belong to the adapter.
 * Tokens are [range/8, valid, y0/192, x0/256, y1/192, x1/256], float32 canonical.
 * This Development-calibrated score is neither probability nor hardware confidence.
 */
object HardwareCalibratedA {
    const val THRESHOLD = 0.4071309640537889
    private val focal = 640.0 / (2.0 * tan(Math.toRadians(50.0)))

    fun evaluate(tof: Array<DoubleArray>): HardwareCalibratedADecision {
        require(tof.size == 64 && tof.all { it.size == 6 }) { "Expected 64 x 6 ToF tokens" }
        var valid = 0
        var definite = 0
        var score = 0.0
        val possible = mutableListOf<Int>()
        tof.forEachIndexed { zone, source ->
            val t = source.map { it.toFloat().toDouble() }
            if (t[1] != 1.0 || !t[0].isFinite() || t[0] <= 0.0) return@forEachIndexed
            require(t.drop(2).all { it.isFinite() }) { "Nonfinite valid zone footprint" }
            val y0 = Math.rint(t[2] * 192.0)
            val x0 = Math.rint(t[3] * 256.0)
            val y1 = Math.rint(t[4] * 192.0)
            val x1 = Math.rint(t[5] * 256.0)
            require(y0 < y1 && x0 < x1) { "Empty valid zone footprint" }
            val aa = doubleArrayOf((x0 * 640.0 / 256.0 - 320.0) / focal, (x1 * 640.0 / 256.0 - 320.0) / focal)
            val bb = doubleArrayOf((y0 * 360.0 / 192.0 - 180.0) / focal, (y1 * 360.0 / 192.0 - 180.0) / focal)
            val value = t[0] * 8.0
            val radius = 0.1 + 3.0 * (0.01 + 0.02 * value)
            val lower = max(0.1, value - radius)
            val upper = value + radius
            val best = rayLimit(closest(aa), closest(bb))
            val worst = aa.minOf { a -> bb.minOf { b -> rayLimit(a, b) } }
            val p = max(lower, 0.3) <= min(upper, best)
            if (lower >= 0.3 && upper <= worst) definite++
            valid++
            if (p) {
                possible.add(zone)
                score = max(score, supportScore(aa, bb, lower, upper))
            }
        }
        val supported = definite > 0
        val alert = supported || (possible.isNotEmpty() && score >= THRESHOLD)
        val state = if (supported) "ALERT_SUPPORTED" else if (alert) "ALERT_AMBIGUOUS"
            else if (possible.isNotEmpty()) "UNKNOWN_WITHHELD" else "NO_SUPPORTED_HIT"
        return HardwareCalibratedADecision(alert, !supported, score, valid, possible, definite, state)
    }

    private fun closest(s: DoubleArray) = if (s[0] <= 0.0 && s[1] >= 0.0) 0.0 else s.minBy { abs(it) }
    private fun rayLimit(a: Double, b: Double): Double = min(3.0, min(
        if (a == 0.0) Double.POSITIVE_INFINITY else 0.3 / abs(a),
        if (b > 0.0) 0.9 / b else if (b < 0.0) -0.2 / b else Double.POSITIVE_INFINITY
    ))

    private fun overlap(s: DoubleArray, low: Double, high: Double, z: Double): Pair<Double, Double> {
        val upper = if (s[1] <= high / z) Pair(s[1], 0.0) else Pair(0.0, high)
        val lower = if (s[0] >= low / z) Pair(s[0], 0.0) else Pair(0.0, low)
        val a = upper.first - lower.first
        val b = upper.second - lower.second
        return if (a + b / z > 0.0) Pair(a, b) else Pair(0.0, 0.0)
    }

    private fun supportScore(aa: DoubleArray, bb: DoubleArray, low: Double, high: Double): Double {
        val start = max(low, 0.3)
        val stop = min(high, 3.0)
        if (start >= stop) return 0.0
        val cuts = sortedSetOf(start, stop)
        for ((slopes, faces) in listOf(aa to doubleArrayOf(-0.3, 0.3), bb to doubleArrayOf(-0.2, 0.9))) {
            for (slope in slopes) if (slope != 0.0) for (face in faces) {
                val cut = face / slope
                if (cut > start && cut < stop) cuts.add(cut)
            }
        }
        var integral = 0.0
        for ((left, right) in cuts.zipWithNext()) {
            val (ax, bx) = overlap(aa, -0.3, 0.3, (left + right) / 2.0)
            val (ay, by) = overlap(bb, -0.2, 0.9, (left + right) / 2.0)
            integral += ax * ay * (right - left) + (ax * by + ay * bx) * ln(right / left) +
                bx * by * (1.0 / left - 1.0 / right)
        }
        val area = (aa[1] - aa[0]) * (bb[1] - bb[0])
        return (integral / (area * (high - low))).coerceIn(0.0, (stop - start) / (high - low))
    }
}
