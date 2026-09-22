package com.linnan.blindassist.risk

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.math.tan

data class DemoTofCell(
    val zone: Int,
    val rangeMm: Double,
    val status: Int,
    val targets: Int,
    val quality: String
)

data class DemoTofDecision(
    val alert: Boolean,
    val state: String,
    /** Nearest triggering raw radial return, not an axial or camera-derived distance. */
    val nearestMm: Int?,
    val triggeringZones: List<Int>,
    val validZones: Int
)

/**
 * Opt-in hardware demonstration of the ToF corridor baseline in ba_camera_corridor.py.
 * Nominal 45-degree square FOV and radial radius 0.13 + 0.06 * range are engineering
 * assumptions, not hardware calibration or transferable simulation performance.
 * Uses the whole zone footprint. Raw row increases toward camera-left, raw column
 * toward camera-up, following the coarse assembly check (not pixel registration).
 * UNKNOWN never certifies free space; even supported alerts are model support only.
 */
object TofCorridorDemo {
    private val halfSlope = tan(Math.toRadians(45.0 / 2.0))

    fun evaluate(rows: Int, cols: Int, cells: List<DemoTofCell>, usable: Boolean): DemoTofDecision {
        if (!usable || rows != cols || rows !in listOf(4, 8) ||
            cells.size != rows * cols || cells.map { it.zone }.toSet() != (0 until rows * cols).toSet()
        ) return unknown()

        var valid = 0
        var supported = false
        var nearest: Double? = null
        val triggering = mutableListOf<Int>()
        for (cell in cells.sortedBy { it.zone }) {
            if (!cell.rangeMm.isFinite() || cell.rangeMm <= 3.0 || cell.status != 5 ||
                cell.targets <= 0 || cell.quality != "KNOWN"
            ) continue
            valid++
            val row = cell.zone / cols
            val col = cell.zone % cols
            // Both raw axes run opposite the camera X-right/Y-down convention.
            val a0 = halfSlope * (1.0 - 2.0 * (row + 1) / rows)
            val a1 = halfSlope * (1.0 - 2.0 * row / rows)
            val b0 = halfSlope * (1.0 - 2.0 * (col + 1) / cols)
            val b1 = halfSlope * (1.0 - 2.0 * col / cols)
            val closestA = closestToZero(a0, a1)
            val closestB = closestToZero(b0, b1)
            val farthestA = max(abs(a0), abs(a1))
            val farthestB = max(abs(b0), abs(b1))
            val range = cell.rangeMm / 1000.0
            val radius = 0.13 + 0.06 * range
            // z = radial / sqrt(1 + a^2 + b^2). These enclosing bounds preserve
            // all rays, including zone edges. They may be ambiguous, never pinpointed.
            val lower = max(0.1, range - radius) /
                sqrt(1.0 + farthestA * farthestA + farthestB * farthestB)
            val upper = (range + radius) /
                sqrt(1.0 + closestA * closestA + closestB * closestB)
            val best = rayLimit(closestA, closestB)
            val worst = minOf(rayLimit(a0, b0), rayLimit(a0, b1), rayLimit(a1, b0), rayLimit(a1, b1))
            if (max(lower, 0.3) <= min(upper, best)) {
                triggering += cell.zone
                nearest = min(nearest ?: cell.rangeMm, cell.rangeMm)
                if (lower >= 0.3 && upper <= worst) supported = true
            }
        }
        if (triggering.isEmpty()) return unknown(valid)
        return DemoTofDecision(
            alert = true,
            state = if (supported) "ALERT_SUPPORTED" else "ALERT_AMBIGUOUS",
            nearestMm = nearest?.roundToInt(),
            triggeringZones = triggering,
            validZones = valid
        )
    }

    private fun closestToZero(low: Double, high: Double): Double =
        if (low <= 0.0 && high >= 0.0) 0.0 else if (abs(low) < abs(high)) low else high

    private fun rayLimit(a: Double, b: Double): Double {
        val x = if (a == 0.0) Double.POSITIVE_INFINITY else 0.3 / abs(a)
        val y = when {
            b > 0.0 -> 0.9 / b
            b < 0.0 -> -0.2 / b
            else -> Double.POSITIVE_INFINITY
        }
        return minOf(x, y, 3.0)
    }

    private fun unknown(valid: Int = 0) = DemoTofDecision(false, "UNKNOWN", null, emptyList(), valid)
}
