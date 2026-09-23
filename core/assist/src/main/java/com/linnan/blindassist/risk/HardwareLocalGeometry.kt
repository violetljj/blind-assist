package com.linnan.blindassist.risk

import kotlin.math.floor
import kotlin.math.sqrt
import kotlin.math.tan

/** Experimental adapter, NOT a measured camera/ToF calibration.
 * Frozen model expects 100-degree 16:9 camera and axial ranges in camera row-major order.
 * Real radial returns use nominal zone-centre axial conversion; this is an approximation.
 */
object HardwareLocalGeometry {
    private val focal = 640.0 / (2 * tan(Math.toRadians(50.0)))
    private val limit = tan(Math.toRadians(22.5))
    private val xs = IntArray(9) { floor((320 + focal * limit * (it / 4.0 - 1)) * 256 / 640 + .5).toInt() }
    private val ys = IntArray(9) { floor((180 + focal * limit * (it / 4.0 - 1)) * 192 / 360 + .5).toInt() }

    fun tokens(cells: List<DemoTofCell>): Array<DoubleArray> {
        require(cells.size == 64 && cells.map { it.zone }.toSet() == (0..63).toSet())
        val byZone = cells.associateBy { it.zone }
        return Array(64) { index ->
            val y = index / 8
            val x = index % 8
            // Raw row increases camera-left; raw column increases camera-up.
            val raw = byZone.getValue((7 - x) * 8 + (7 - y))
            val a = limit * ((x + .5) / 4 - 1)
            val b = limit * ((y + .5) / 4 - 1)
            val axial = raw.rangeMm / 1000 / sqrt(1 + a * a + b * b)
            val valid = raw.status == 5 && raw.targets > 0 && raw.quality == "KNOWN" &&
                axial.isFinite() && axial >= .1 && axial < 8
            // Public training observations are float32, including projected box edges.
            doubleArrayOf(if (valid) axial / 8 else 0.0, if (valid) 1.0 else 0.0,
                ys[y] / 192.0, xs[x] / 256.0, ys[y + 1] / 192.0, xs[x + 1] / 256.0)
                .map { it.toFloat().toDouble() }.toDoubleArray()
        }
    }

    fun rawZone(canonical: Int): Int = (7 - canonical % 8) * 8 + (7 - canonical / 8)
}
