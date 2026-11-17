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
    /** [medianDepthM] is the in-box pixel median forward distance (NaN below the pixel gate). */
    data class Query(val pixels: Int, val alarm: Boolean, val medianDepthM: Double = Double.NaN)
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
    // Readout scratch for the in-box depth medians (demo distance gate); alarms never read it.
    private val headDepths = FloatArray(width * height)
    private val bodyDepths = FloatArray(width * height)

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

    // ---- Direction A metric chain (2026-11-18): UniDepth-V2-S optical depth times ONE scale. ----
    // Mirrors evalkit Methods.hybrid_out / hybrid_in: ToF scale (>= 5 zones) -> own-ground scale x known eye height ->
    // previous scale -> none. The scale is carried as Fit(a = s, b = 0) on disparity = 1 / z, so [readout] gives s * z.

    /** One scale = median over valid zones (>= 5) of (ToF range / model range on the zone's sample rays); null otherwise.
     * A zone needs >= 13 in-image samples with model depth > 0.1 m, a valid firmware status and 0.3 < range < 4 m. */
    fun tofScale(depthM: FloatArray, rawDistanceMm: DoubleArray, rawStatus: IntArray): Double? {
        require(depthM.size == width * height && rawDistanceMm.size == 64 && rawStatus.size == 64)
        val ratios = ArrayList<Double>()
        for (zone in 0 until 64) {
            val raw = 63 - zone
            val distance = rawDistanceMm[raw] / 1000.0
            if (rawStatus[raw] !in VALID_STATUS || distance <= .3 || distance >= 4.0) continue
            val ranges = ArrayList<Double>()
            for (s in 0 until 25) {
                val p = zonePixels[zone][s]
                if (p < 0) continue
                val z = depthM[p]
                if (z.isFinite() && z > .1f) ranges.add(z / zoneFactors[zone][s])
            }
            if (ranges.size >= 13) ratios.add(distance / median(ranges))
        }
        return if (ratios.size >= 5) median(ratios) else null
    }

    /** Scale from the model's own ground: RANSAC plane below the camera (normal within 15 deg of IMU gravity, 3 cm
     * tolerance, >= 2000 inliers, plane distance > 0.3 m) in the unscaled point cloud; s = eye height / plane distance,
     * accepted in [0.5, 2]. Same rule as evalkit core.ground_plane / Methods._floor_scale (Java RNG, not numpy's). */
    fun floorScale(depthM: FloatArray, pitchDeg: Double, rollDeg: Double, eyeHeightM: Double): Double? {
        require(depthM.size == width * height && eyeHeightM.isFinite() && eyeHeightM > 0)
        if (!pitchDeg.isFinite() || !rollDeg.isFinite()) return null
        val m = corridorFromCamera(pitchDeg, rollDeg)
        val down = doubleArrayOf(m[3], m[4], m[5])
        var count = 0
        val below = FloatArray(depthM.size * 3)
        for (i in depthM.indices) {
            val z = depthM[i]
            if (!z.isFinite() || z <= .3f || z >= 10f) continue
            val d = directions[i]
            val x = d[0] * z; val y = d[1] * z; val zz = z.toDouble()
            if (x * down[0] + y * down[1] + zz * down[2] <= .5) continue
            below[count * 3] = x.toFloat(); below[count * 3 + 1] = y.toFloat(); below[count * 3 + 2] = z; count++
        }
        if (count < FLOOR_MIN_INLIERS) return null
        val n = minOf(30000, count)
        val sel = IntArray(count) { it }
        for (i in 0 until n) { val j = i + rng.nextInt(count - i); val t = sel[i]; sel[i] = sel[j]; sel[j] = t }
        // Contiguous copy of the sample (same points, same order) so each hypothesis scans three flat arrays.
        val sx = FloatArray(n) { below[sel[it] * 3] }; val sy = FloatArray(n) { below[sel[it] * 3 + 1] }
        val sz = FloatArray(n) { below[sel[it] * 3 + 2] }
        val cosMax = cos(Math.toRadians(15.0))
        var bestCount = -1; var best: DoubleArray? = null
        repeat(300) {
            val a = rng.nextInt(n); val b = rng.nextInt(n); val c = rng.nextInt(n)
            val ux = (sx[b] - sx[a]).toDouble(); val uy = (sy[b] - sy[a]).toDouble(); val uz = (sz[b] - sz[a]).toDouble()
            val vx = (sx[c] - sx[a]).toDouble(); val vy = (sy[c] - sy[a]).toDouble(); val vz = (sz[c] - sz[a]).toDouble()
            var nx = uy * vz - uz * vy; var ny = uz * vx - ux * vz; var nz = ux * vy - uy * vx
            val norm = sqrt(nx * nx + ny * ny + nz * nz)
            if (norm < 1e-6) return@repeat
            nx /= norm; ny /= norm; nz /= norm
            if (nx * down[0] + ny * down[1] + nz * down[2] > 0) { nx = -nx; ny = -ny; nz = -nz }
            if (-(nx * down[0] + ny * down[1] + nz * down[2]) < cosMax) return@repeat
            val dist = -(nx * sx[a] + ny * sy[a] + nz * sz[a])
            if (dist <= .3) return@repeat
            val fx = nx.toFloat(); val fy = ny.toFloat(); val fz = nz.toFloat(); val fd = dist.toFloat()
            var inliers = 0
            for (k in 0 until n) if (abs(fx * sx[k] + fy * sy[k] + fz * sz[k] + fd) < .03f) inliers++
            if (inliers > bestCount) { bestCount = inliers; best = doubleArrayOf(nx, ny, nz, dist) }
        }
        val plane = best ?: return null
        var all = 0
        for (p in 0 until count)
            if (abs(plane[0] * below[p * 3] + plane[1] * below[p * 3 + 1] + plane[2] * below[p * 3 + 2] + plane[3]) < .03) all++
        if (all < FLOOR_MIN_INLIERS) return null
        val s = eyeHeightM / plane[3]
        return s.takeIf { it in .5..2.0 }
    }

    /** The adopted chain on metric depth; anchors are TOF / FLOOR / PREVIOUS / NONE as in [process]. */
    fun processMetric(depthM: FloatArray, validPixels: BooleanArray, rawDistanceMm: DoubleArray, rawStatus: IntArray,
                      pitchDeg: Double, rollDeg: Double, eyeHeightM: Double = 1.35): Result {
        require(depthM.size == width * height && validPixels.size == depthM.size)
        val poseOk = poseInEnvelope(pitchDeg, rollDeg)
        if (!poseOk) return Result(Anchor.NONE, null, false, Query(0, false), Query(0, false))
        var scale = tofScale(depthM, rawDistanceMm, rawStatus); var anchor = Anchor.TOF
        if (scale == null) { scale = floorScale(depthM, pitchDeg, rollDeg, eyeHeightM); anchor = Anchor.FLOOR }
        var fit = scale?.let { Fit(it, 0.0) }
        if (fit == null) { fit = previous; anchor = if (fit == null) Anchor.NONE else Anchor.PREVIOUS }
        if (fit != null) previous = fit
        if (fit == null) return Result(anchor, null, poseOk, Query(0, false), Query(0, false))
        val disparity = FloatArray(depthM.size) { val z = depthM[it]; if (z.isFinite() && z > 0f) 1f / z else Float.NaN }
        val queries = readout(disparity, validPixels, fit, pitchDeg, rollDeg)
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
        // Frozen NumPy operates on a float32 disparity array: Python scalar a/b and the
        // reciprocal are rounded to float32 before the float64 camera-ray projection.
        // Promoting this division to Double can move an inclusive 2 m face outside the box.
        val a = fit.a.toFloat(); val b = fit.b.toFloat()
        var head = 0; var body = 0
        for (i in disparity.indices) {
            if (!validPixels[i]) continue
            val inv = (disparity[i] - b) / a
            if (!inv.isFinite() || inv <= 1e-3f) continue
            val depth = (1f / inv).toDouble()
            val d = directions[i]
            val x = (matrix[0] * d[0] + matrix[1] * d[1] + matrix[2]) * depth
            val y = (matrix[3] * d[0] + matrix[4] * d[1] + matrix[5]) * depth
            val z = (matrix[6] * d[0] + matrix[7] * d[1] + matrix[8]) * depth
            if (abs(x) <= .30 && z >= .30 && z <= 2.0) {
                if (y >= -.20 && y <= .42) headDepths[head++] = z.toFloat()
                if (y >= .42 && y <= .90) bodyDepths[body++] = z.toFloat()
            }
        }
        val poseOk = poseInEnvelope(pitchDeg, rollDeg)
        return Query(head, head >= 150 && poseOk, medianOf(headDepths, head)) to
            Query(body, body >= 150 && poseOk, medianOf(bodyDepths, body))
    }

    private fun medianOf(values: FloatArray, count: Int): Double {
        if (count < 150) return Double.NaN
        java.util.Arrays.sort(values, 0, count)
        return if (count % 2 == 1) values[count / 2].toDouble()
               else (values[count / 2 - 1].toDouble() + values[count / 2]) / 2
    }

    /** Nearest forward distance of valid raw ToF zones whose radial hit lands in each query box
     * (HEAD to BODY). NaN when no zone lands there. Camera and ToF are treated as co-located. */
    fun tofQueryDepth(rawDistanceMm: DoubleArray, rawStatus: IntArray,
                      pitchDeg: Double, rollDeg: Double): Pair<Double, Double> {
        require(rawDistanceMm.size == 64 && rawStatus.size == 64)
        if (!pitchDeg.isFinite() || !rollDeg.isFinite()) return Double.NaN to Double.NaN
        val pose = poseRotation(pitchDeg, rollDeg)
        val edge = tan(Math.PI / 8)
        var head = Double.NaN; var body = Double.NaN
        for (zone in 0 until 64) {
            val raw = 63 - zone
            val distance = rawDistanceMm[raw] / 1000.0
            if (rawStatus[raw] !in VALID_STATUS || distance <= .3 || distance >= 4.0) continue
            val rx = -edge + (zone % 8 + .5) * 2 * edge / 8
            val ry = -edge + (zone / 8 + .5) * 2 * edge / 8
            val scale = distance / sqrt(rx * rx + ry * ry + 1)
            val p = multiply(pose, doubleArrayOf(rx * scale, ry * scale, scale))
            if (abs(p[0]) > .30 || p[2] < .30 || p[2] > 2.0) continue
            if (p[1] >= -.20 && p[1] <= .42 && !(p[2] >= head)) head = p[2]
            if (p[1] >= .42 && p[1] <= .90 && !(p[2] >= body)) body = p[2]
        }
        return head to body
    }

    /** Image quads of the HEAD and BODY boxes at 1.0 m and 2.0 m (corners TL, TR, BR, BL; null
     * when behind the camera), for display only. */
    fun boxOutlines(pitchDeg: Double, rollDeg: Double): List<List<Pair<Float, Float>?>> {
        if (!pitchDeg.isFinite() || !rollDeg.isFinite()) return emptyList()
        val m = corridorFromCamera(pitchDeg, rollDeg)
        fun project(x: Double, y: Double, z: Double): Pair<Float, Float>? {
            // corridor = M · camera with orthonormal M, so camera = Mᵀ · corridor.
            val cx = m[0] * x + m[3] * y + m[6] * z
            val cy = m[1] * x + m[4] * y + m[7] * z
            val cz = m[2] * x + m[5] * y + m[8] * z
            if (cz < .05) return null
            return (calibration.fx * cx / cz + calibration.cx).toFloat() to (calibration.fy * cy / cz + calibration.cy).toFloat()
        }
        return listOf(-.20 to .42, .42 to .90).flatMap { (top, bottom) ->
            listOf(1.0, 2.0).map { z ->
                listOf(project(-.30, top, z), project(.30, top, z), project(.30, bottom, z), project(-.30, bottom, z))
            }
        }
    }

    private fun corridorFromCamera(pitchDeg: Double, rollDeg: Double): DoubleArray {
        val pose = poseRotation(pitchDeg, rollDeg)
        val r = calibration.cameraFromTof
        return DoubleArray(9) { i -> (0..2).sumOf { k -> pose[i / 3 * 3 + k] * r[i % 3 * 3 + k] } }
    }

    companion object {
        private val VALID_STATUS = setOf(5, 6, 9, 10)
        private const val FLOOR_MIN_INLIERS = 2000
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
            // lstsq(rcond=None) tests singular values against eps * max(rows, columns).
            // Even identical x can have nonzero xx after the mean is rounded; xx == 0 then
            // invents a full-rank solve. For A=[x,1], det(A'A)=n*xx and sMax^2 is below.
            val n = x.size.toDouble()
            val sumSquares = xx + n * mx * mx
            val sMaxSquared = .5 * (sumSquares + n + kotlin.math.hypot(sumSquares - n, 2 * n * mx))
            val rankOne = sqrt(n * xx) <= Math.ulp(1.0) * maxOf(x.size, 2) * sMaxSquared
            val a = if (rankOne) mx * my / (mx * mx + 1.0)
                    else x.indices.sumOf { (x[it] - mx) * (y[it] - my) } / xx
            val b = if (rankOne) my / (mx * mx + 1.0) else my - a * mx
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
