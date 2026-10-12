package com.linnan.blindassist.vision

import java.util.Random
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/** Android engineering port of the 2026-10-12 Atom RGB readout. No model inference or image warp.
 *
 * Input disparity and nonblack mask must already have the calibrated upright resolution. Rct is
 * row-major camera <- ToF rotation; raw ToF zones are rotated 180 degrees exactly once here.
 * Geometry/thresholds follow real_rgb_small_model_dev_20261012.py and frozen_fusion_eval_v1.py.
 * The default floor RNG is Java's, NOT numpy PCG64. Supply [FloorSamples] for numerical parity
 * checks against an exported numpy draw. This class is session-owned and is not thread safe.
 */
class AtomRgbGeometry(val width: Int, val height: Int, val calibration: Calibration) {
    data class Calibration(val fx: Double, val fy: Double, val cx: Double, val cy: Double,
                           val cameraFromTof: DoubleArray) {
        init {
            require(fx.isFinite() && fy.isFinite() && fx > 0 && fy > 0)
            require(cx.isFinite() && cy.isFinite())
            require(cameraFromTof.size == 9 && cameraFromTof.all { it.isFinite() })
        }
    }
    data class Fit(val a: Double, val b: Double)
    enum class Anchor { TOF, FLOOR, PREVIOUS, NONE }
    data class Query(val pixels: Int, val alarm: Boolean)
    data class Result(val anchor: Anchor, val fit: Fit?, val poseOk: Boolean,
                      val head: Query, val body: Query)
    data class ZoneSamples(val disparity: DoubleArray, val opticalDepthFactor: DoubleArray)
    /** Pixel indices in the floor candidate set; pair indices address that sampled pixel array. */
    data class FloorSamples(val pixels: IntArray, val pairs: List<Pair<Int, Int>>)

    private var previous: Fit? = null
    private val rng = Random(20261013L)
    private val directions: Array<DoubleArray>
    // [zone][sample]: rounded upright image index or -1 and optical-depth factor.
    private val zonePixels = Array(64) { IntArray(25) { -1 } }
    private val zoneFactors = Array(64) { DoubleArray(25) { Double.NaN } }

    init {
        require(width > 1 && height > 1)
        directions = Array(width * height) { i ->
            doubleArrayOf((i % width - calibration.cx) / calibration.fx,
                (i / width - calibration.cy) / calibration.fy, 1.0)
        }
        val edge = tan(Math.PI / 8)
        for (zone in 0 until 64) for (sample in 0 until 25) {
            val x = -edge + (zone % 8 + (sample % 5 + .5) / 5) * 2 * edge / 8
            val y = -edge + (zone / 8 + (sample / 5 + .5) / 5) * 2 * edge / 8
            val norm = sqrt(x * x + y * y + 1)
            val rc = multiply(calibration.cameraFromTof, doubleArrayOf(x / norm, y / norm, 1 / norm))
            val u = calibration.fx * rc[0] / rc[2] + calibration.cx
            val v = calibration.fy * rc[1] / rc[2] + calibration.cy
            if (rc[2] > .05 && u >= 0 && u < width - 1 && v >= 0 && v < height - 1) {
                // np.round uses ties-to-even, unlike Kotlin roundToInt.
                zonePixels[zone][sample] = Math.rint(v).toInt() * width + Math.rint(u).toInt()
                zoneFactors[zone][sample] = rc[2] / sqrt(rc.sumOf { it * it })
            }
        }
    }

    fun reset() { previous = null; rng.setSeed(20261013L) }

    fun zoneDisparity(disparity: FloatArray): ZoneSamples {
        require(disparity.size == width * height)
        val dz = DoubleArray(64) { Double.NaN }
        val factors = DoubleArray(64) { Double.NaN }
        for (zone in 0 until 64) {
            val indices = zonePixels[zone].filter { it >= 0 }
            if (indices.isNotEmpty()) factors[zone] = median(zoneFactors[zone].filter { !it.isNaN() })
            if (indices.size >= 13) dz[zone] = median(indices.map { disparity[it].toDouble() }.filter { !it.isNaN() })
        }
        return ZoneSamples(dz, factors)
    }

    /** Raw firmware millimetres/statuses, unrotated row-major 8x8. */
    fun tofAnchor(disparity: FloatArray, rawDistanceMm: DoubleArray, rawStatus: IntArray): Fit? {
        require(rawDistanceMm.size == 64 && rawStatus.size == 64)
        val zones = zoneDisparity(disparity)
        val invZ = DoubleArray(64) { zone ->
            val raw = 63 - zone
            val distance = rawDistanceMm[raw] / 1000.0
            if (rawStatus[raw] in VALID_STATUS && distance > .3 && distance < 4.0)
                1 / (distance * zones.opticalDepthFactor[zone]) else Double.NaN
        }
        return fitAffine(zones.disparity, invZ)
    }

    fun process(disparity: FloatArray, validPixels: BooleanArray, rawDistanceMm: DoubleArray,
                rawStatus: IntArray, pitchDeg: Double, rollDeg: Double, headHeightM: Double = 1.35,
                floorSamples: FloorSamples? = null): Result {
        require(disparity.size == width * height && validPixels.size == disparity.size)
        val poseOk = poseInEnvelope(pitchDeg, rollDeg)
        // v1 does not anchor/update previous on gated frames. Unmeasured pose follows the Python
        // envelope policy but cannot produce corridor points from a nonfinite rotation.
        if (!poseOk) return Result(Anchor.NONE, null, false, Query(0, false), Query(0, false))
        var fit = tofAnchor(disparity, rawDistanceMm, rawStatus)
        var anchor = Anchor.TOF
        if (fit == null) { fit = floorAnchor(disparity, pitchDeg, rollDeg, headHeightM, floorSamples); anchor = Anchor.FLOOR }
        if (fit == null) { fit = previous; anchor = if (fit == null) Anchor.NONE else Anchor.PREVIOUS }
        if (fit != null) previous = fit
        val queries = if (fit == null) Query(0, false) to Query(0, false)
                      else readout(disparity, validPixels, fit, pitchDeg, rollDeg)
        return Result(anchor, fit, poseOk, queries.first, queries.second)
    }

    fun floorAnchor(disparity: FloatArray, pitchDeg: Double, rollDeg: Double,
                    headHeightM: Double = 1.35, samples: FloorSamples? = null): Fit? {
        require(disparity.size == width * height && headHeightM.isFinite() && headHeightM > 0)
        val matrix = corridorFromCamera(pitchDeg, rollDeg)
        val invZ = DoubleArray(directions.size) {
            val d = directions[it]
            (matrix[3] * d[0] + matrix[4] * d[1] + matrix[5]) / headHeightM
        }
        val candidates = invZ.indices.filter { invZ[it] * headHeightM > .1 && invZ[it] > 1 / 6.0 && invZ[it] < 1 / .8 }
        if (candidates.size < 500) return null
        val selected: IntArray
        val pairs: List<Pair<Int, Int>>
        if (samples != null) {
            require(samples.pixels.size == minOf(4000, candidates.size))
            val allowed = candidates.toHashSet()
            require(samples.pixels.toSet().size == samples.pixels.size && samples.pixels.all { it in allowed })
            require(samples.pairs.size == 64 && samples.pairs.all {
                it.first in samples.pixels.indices && it.second in samples.pixels.indices && it.first != it.second
            })
            selected = samples.pixels; pairs = samples.pairs
        } else {
            val shuffled = candidates.toIntArray()
            val n = minOf(4000, shuffled.size)
            for (i in 0 until n) { val j = i + rng.nextInt(shuffled.size - i); val t = shuffled[i]; shuffled[i] = shuffled[j]; shuffled[j] = t }
            selected = shuffled.copyOf(n)
            pairs = List(64) {
                val i = rng.nextInt(n); var j = rng.nextInt(n - 1); if (j >= i) j++
                i to j
            }
        }
        val x = selected.map { invZ[it] }
        val y = selected.map { disparity[it].toDouble() }
        val threshold = .03 * (percentile(y, .95) - percentile(y, .05) + 1e-6)
        var best: List<Int>? = null
        for ((i, j) in pairs) {
            if (abs(x[i] - x[j]) < 1e-3) continue
            val a = (y[i] - y[j]) / (x[i] - x[j]); val b = y[i] - a * x[i]
            if (a <= 0) continue
            val inliers = x.indices.filter { abs(y[it] - (a * x[it] + b)) < threshold }
            if (best == null || inliers.size > best.size) best = inliers
        }
        val inliers = best ?: return null
        if (inliers.size < .3 * x.size) return null
        return leastSquares(inliers.map { x[it] }, inliers.map { y[it] })?.takeIf { it.a > 0 }
    }

    fun readout(disparity: FloatArray, validPixels: BooleanArray, fit: Fit,
                pitchDeg: Double, rollDeg: Double): Pair<Query, Query> {
        require(disparity.size == width * height && validPixels.size == disparity.size)
        val matrix = corridorFromCamera(pitchDeg, rollDeg)
        var head = 0; var body = 0
        for (i in disparity.indices) {
            if (!validPixels[i]) continue
            val inv = (disparity[i] - fit.b) / fit.a
            if (!inv.isFinite() || inv <= 1e-3) continue
            val d = directions[i]
            val x = (matrix[0] * d[0] + matrix[1] * d[1] + matrix[2]) / inv
            val y = (matrix[3] * d[0] + matrix[4] * d[1] + matrix[5]) / inv
            val z = (matrix[6] * d[0] + matrix[7] * d[1] + matrix[8]) / inv
            if (abs(x) <= .30 && z >= .30 && z <= 2.0) {
                if (y >= -.20 && y <= .42) head++
                if (y >= .42 && y <= .90) body++
            }
        }
        val poseOk = poseInEnvelope(pitchDeg, rollDeg)
        return Query(head, head >= 150 && poseOk) to Query(body, body >= 150 && poseOk)
    }

    private fun corridorFromCamera(pitchDeg: Double, rollDeg: Double): DoubleArray {
        val pose = poseRotation(pitchDeg, rollDeg)
        val r = calibration.cameraFromTof
        return DoubleArray(9) { i -> (0..2).sumOf { k -> pose[i / 3 * 3 + k] * r[i % 3 * 3 + k] } }
    }

    companion object {
        private val VALID_STATUS = setOf(5, 6, 9, 10)
        fun poseInEnvelope(pitchDeg: Double, rollDeg: Double): Boolean =
            !(pitchDeg.isFinite() && rollDeg.isFinite() && (pitchDeg > 10 || abs(rollDeg) > 25))

        fun poseRotation(pitchDeg: Double, rollDeg: Double): DoubleArray {
            val p = Math.toRadians(pitchDeg); val r = Math.toRadians(rollDeg)
            return doubleArrayOf(cos(r), -sin(r), 0.0, cos(p) * sin(r), cos(p) * cos(r), -sin(p),
                sin(p) * sin(r), sin(p) * cos(r), cos(p))
        }

        /** Three least-squares solves and median residual trimming, as the Python reference. */
        fun fitAffine(disparity: DoubleArray, inverseOpticalDepth: DoubleArray): Fit? {
            require(disparity.size == inverseOpticalDepth.size)
            var indices = disparity.indices.filter { disparity[it].isFinite() && inverseOpticalDepth[it].isFinite() }
            if (indices.size < 6) return null
            var fit: Fit? = null
            repeat(3) {
                val current = leastSquares(indices.map { inverseOpticalDepth[it] }, indices.map { disparity[it] }) ?: return null
                fit = current
                val residual = indices.map { disparity[it] - (current.a * inverseOpticalDepth[it] + current.b) }
                val scale = median(residual.map { abs(it) }) * 1.4826 + 1e-9
                val kept = indices.filterIndexed { i, _ -> abs(residual[i]) < 2.5 * scale }
                if (kept.size < 6 || kept.size == indices.size) return current.takeIf { it.a > 0 }
                indices = kept
            }
            return fit?.takeIf { it.a > 0 }
        }

        private fun leastSquares(x: List<Double>, y: List<Double>): Fit? {
            if (x.isEmpty()) return null
            val mx = x.average(); val my = y.average()
            val xx = x.sumOf { (it - mx) * (it - mx) }
            // numpy lstsq's rank-one minimum norm for constant x, not a fabricated denominator.
            val a = if (xx == 0.0) mx * my / (mx * mx + 1.0)
                    else x.indices.sumOf { (x[it] - mx) * (y[it] - my) } / xx
            val b = if (xx == 0.0) my / (mx * mx + 1.0) else my - a * mx
            return if (a.isFinite() && b.isFinite()) Fit(a, b) else null
        }

        private fun multiply(m: DoubleArray, v: DoubleArray): DoubleArray =
            DoubleArray(3) { i -> m[3 * i] * v[0] + m[3 * i + 1] * v[1] + m[3 * i + 2] * v[2] }
        private fun median(values: List<Double>): Double = percentile(values, .5)
        private fun percentile(values: List<Double>, q: Double): Double {
            if (values.isEmpty() || values.any { it.isNaN() }) return Double.NaN
            val sorted = values.sorted(); val p = q * (sorted.size - 1); val lo = p.toInt(); val hi = minOf(lo + 1, sorted.lastIndex)
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo)
        }
    }
}

/** Frozen v2 per-query rule. Call once per ready ToF frame, including gated/unmatched false frames.
 * Reset at session boundaries. A current false frame can still confirm preceding positives, exactly
 * as frozen_fusion_eval_v2.confirm; a runtime safety gate must be applied separately by its caller.
 */
class AtomRgbTwoOfThree {
    private val head = BooleanArray(3)
    private val body = BooleanArray(3)
    private var next = 0
    fun reset() { head.fill(false); body.fill(false); next = 0 }
    fun push(headAlarm: Boolean, bodyAlarm: Boolean): Pair<Boolean, Boolean> {
        head[next] = headAlarm; body[next] = bodyAlarm; next = (next + 1) % 3
        return (head.count { it } >= 2) to (body.count { it } >= 2)
    }
}
