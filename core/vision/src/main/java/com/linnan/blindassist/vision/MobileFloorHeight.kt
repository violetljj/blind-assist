package com.linnan.blindassist.vision

import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/** Causal startup engineering calibration: hold a downward view of an empty floor for 20 frames.
 * Uses per-zone medians then orthogonal least-squares plane fit, 5 cm trimming, and one refit.
 * This is NOT the offline session-wide scipy soft_l1 calibration in real_phone_imu_replay.py.
 * The caller owns start/retry, frame freshness, and invalidation when mounting/session changes.
 */
class MobileFloorHeight(private val requiredFrames: Int = 20) {
    data class Calibration(val heightM: Double, val pitchDeg: Double, val rollDeg: Double, val inliers: Int)
    data class Progress(val frameCount: Int, val targetFrames: Int, val calibration: Calibration?) {
        val complete: Boolean get() = frameCount >= targetFrames
    }
    private val frames = mutableListOf<DoubleArray>()
    private var result: Calibration? = null
    init { require(requiredFrames > 0) }
    fun reset() { frames.clear(); result = null }

    fun add(rawDistanceMm: DoubleArray, rawStatus: IntArray): Progress {
        require(rawDistanceMm.size == 64 && rawStatus.size == 64)
        if (frames.size < requiredFrames) {
            frames += DoubleArray(64) { zone ->
                val raw = 63 - zone
                if (rawStatus[raw] == 5 || rawStatus[raw] == 6 || rawStatus[raw] == 9 || rawStatus[raw] == 10)
                    rawDistanceMm[raw] else Double.NaN
            }
            if (frames.size == requiredFrames) result = fitMedians()
        }
        return Progress(frames.size, requiredFrames, result)
    }

    private fun fitMedians(): Calibration? {
        val edge = tan(Math.PI / 8)
        val points = (32 until 64).mapNotNull { zone ->
            val samples = frames.map { it[zone] }.filter { it.isFinite() }.sorted()
            if (samples.isEmpty()) return@mapNotNull null
            val mm = (samples[(samples.size - 1) / 2] + samples[samples.size / 2]) / 2
            if (mm <= 600) return@mapNotNull null
            val x = -edge + (zone % 8 + .5) * 2 * edge / 8
            val y = -edge + (zone / 8 + .5) * 2 * edge / 8
            val scale = mm / 1000 / sqrt(x * x + y * y + 1)
            doubleArrayOf(x * scale, y * scale, scale)
        }
        if (points.size < 8) return null
        val initial = plane(points) ?: return null
        val kept = points.filter { p -> abs(dot(initial.first, p) - initial.second) < .05 }
        if (kept.size < 8) return null
        val (normal, height) = plane(kept) ?: return null
        if (height !in 1.0..1.9) return null
        return Calibration(height, -Math.toDegrees(asin(normal[2].coerceIn(-1.0, 1.0))),
            Math.toDegrees(atan2(normal[0], normal[1])), kept.size)
    }

    /** Small symmetric Jacobi eigensolve: least-variance normal of centered 3D points. */
    private fun plane(points: List<DoubleArray>): Pair<DoubleArray, Double>? {
        val mean = DoubleArray(3) { k -> points.sumOf { it[k] } / points.size }
        val cov = Array(3) { i -> DoubleArray(3) { j -> points.sumOf { (it[i] - mean[i]) * (it[j] - mean[j]) } } }
        val vectors = Array(3) { i -> DoubleArray(3) { j -> if (i == j) 1.0 else 0.0 } }
        repeat(24) {
            val (p, q) = listOf(0 to 1, 0 to 2, 1 to 2).maxByOrNull { abs(cov[it.first][it.second]) }!!
            if (abs(cov[p][q]) > 1e-14) {
                val angle = .5 * atan2(2 * cov[p][q], cov[q][q] - cov[p][p])
                val c = cos(angle); val s = sin(angle)
                val pp = cov[p][p]; val qq = cov[q][q]; val pq = cov[p][q]
                for (k in 0..2) if (k != p && k != q) {
                    val kp = cov[k][p]; val kq = cov[k][q]
                    cov[k][p] = c * kp - s * kq; cov[p][k] = cov[k][p]
                    cov[k][q] = s * kp + c * kq; cov[q][k] = cov[k][q]
                }
                cov[p][p] = c * c * pp - 2 * s * c * pq + s * s * qq
                cov[q][q] = s * s * pp + 2 * s * c * pq + c * c * qq
                cov[p][q] = 0.0; cov[q][p] = 0.0
                for (k in 0..2) {
                    val kp = vectors[k][p]; val kq = vectors[k][q]
                    vectors[k][p] = c * kp - s * kq; vectors[k][q] = s * kp + c * kq
                }
            }
        }
        val order = (0..2).sortedBy { cov[it][it] }
        // Reject line/point degeneracy: a floor normal is otherwise not identifiable.
        if (cov[order[1]][order[1]] <= 1e-10) return null
        val sign = if (vectors[1][order[0]] < 0) -1.0 else 1.0
        val normal = DoubleArray(3) { vectors[it][order[0]] * sign }
        return normal to dot(normal, mean)
    }
    private fun dot(a: DoubleArray, b: DoubleArray) = (0..2).sumOf { a[it] * b[it] }
}
