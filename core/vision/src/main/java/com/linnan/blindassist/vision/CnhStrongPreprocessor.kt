package com.linnan.blindassist.vision

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

/** Frozen physical bias/noise calibration, in raw H3 zone order. */
data class CnhStrongCalibration(
    val bias: DoubleArray,
    val coefficients: DoubleArray,
    val varianceFloor: Double,
    val raisedTau: Double,
) {
    init {
        require(bias.size == 1024 && coefficients.size == 3)
        require(bias.all { it.isFinite() } && coefficients.all { it.isFinite() && it >= 0 })
        require(varianceFloor.isFinite() && varianceFloor > 0 && raisedTau.isFinite())
    }

    companion object {
        /** Exporter schema: 3 coefficients, variance floor, raised cut, 1024 bias doubles (LE). */
        fun read(file: File): CnhStrongCalibration {
            val bytes = file.readBytes()
            require(bytes.size == 1029 * 8) { "Wrong CNH calibration size" }
            val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val c = DoubleArray(3) { b.double }
            val floor = b.double
            val tau = b.double
            return CnhStrongCalibration(DoubleArray(1024) { b.double }, c, floor, tau)
        }
    }
}

/** FP32 engineering port of adapter_v5(orient=true, baseline=false) and CVR projection.
 * Current causal pose applies to the complete window; this is not segment-mean evaluation.
 */
class CnhStrongPreprocessor(private val calibration: CnhStrongCalibration) {
    companion object {
        const val VOXELS = 24 * 17 * 33
        const val INPUT_SIZE = 5 * VOXELS
        private const val SAMPLES = VOXELS * 27
        private const val WIDTH = 8 * .0375348
        private val EDGE = tan(PI / 8)
    }

    private val pointX = FloatArray(SAMPLES)
    private val pointY = FloatArray(SAMPLES)
    private val pointZ = FloatArray(SAMPLES)
    private val volumes = FloatArray(1024)
    private val masks = FloatArray(2 * VOXELS)
    private val indexes = IntArray(SAMPLES)
    private val weights = FloatArray(SAMPLES)
    private val coverage = FloatArray(VOXELS)
    private var mapPitch = Double.NaN
    private var mapRoll = Double.NaN

    init {
        var p = 0
        var v = 0
        for (ix in 0 until 24) for (iy in 0 until 17) for (iz in 0 until 33) {
            val center = doubleArrayOf(-.6 + (ix + .5) * .05, -.5 + (iy + .5) * .1, (iz + .5) * .1)
            for (sx in 0..2) for (sy in 0..2) for (sz in 0..2) {
                pointX[p] = (center[0] + ((sx + .5) / 3 - .5) * .05).toFloat()
                pointY[p] = (center[1] + ((sy + .5) / 3 - .5) * .1).toFloat()
                pointZ[p] = (center[2] + ((sz + .5) / 3 - .5) * .1).toFloat()
                p++
            }
            for (q in 0..1) {
                val low = doubleArrayOf(-.3, if (q == 0) -.2 else .42, .3)
                val high = doubleArrayOf(.3, if (q == 0) .42 else .9, 3.0)
                val step = doubleArrayOf(.05, .1, .1)
                var fraction = 1.0
                for (a in 0..2) fraction *= max(0.0, min(center[a] + step[a] / 2, high[a]) -
                    max(center[a] - step[a] / 2, low[a])) / step[a]
                masks[q * VOXELS + v] = fraction.toFloat()
            }
            v++
        }
        fun primitive(x: Double, y: Double) = atan2(x * y, sqrt(1 + x * x + y * y))
        for (y in 0..7) for (x in 0..7) {
            val xl = -EDGE + x * (2 * EDGE / 8)
            val xh = -EDGE + (x + 1) * (2 * EDGE / 8)
            val yl = -EDGE + y * (2 * EDGE / 8)
            val yh = -EDGE + (y + 1) * (2 * EDGE / 8)
            val angle = primitive(yh, xh) - primitive(yl, xh) - primitive(yh, xl) + primitive(yl, xl)
            for (bin in 0..15) volumes[(y * 8 + x) * 16 + bin] =
                (angle * (((bin + 1) * WIDTH).pow(3) - (bin * WIDTH).pow(3)) / 3).toFloat()
        }
    }

    fun normalize(histogram: FloatArray, ambient: FloatArray): FloatArray {
        require(histogram.size == 1024 && ambient.size == 64)
        require(histogram.all { it.isFinite() } && ambient.all { it.isFinite() }) { "Nonfinite CNH input" }
        val z = FloatArray(1024)
        var positive = false
        val c = calibration.coefficients
        for (zone in 0..63) for (bin in 1..15) {
            val i = zone * 16 + bin
            val physical = max(calibration.bias[i], 0.0)
            val h = histogram[i].toDouble()
            val residual = h - min(physical, max(h, 0.0))
            val variance = max(c[0] * physical + c[1] * max(ambient[zone].toDouble(), 0.0) + c[2], calibration.varianceFloor)
            val value = (residual / sqrt(variance)).toFloat()
            require(value.isFinite()) { "CNH normalization overflow" }
            z[(63 - zone) * 16 + bin] = value
            if (value > 0) positive = true
        }
        // Exact absence gate. Negative evidence remains signed when any positive bin exists.
        if (!positive) z.fill(0f)
        return z
    }

    private fun configure(pitch: Double, roll: Double) {
        require(pitch.isFinite() && roll.isFinite())
        if (pitch == mapPitch && roll == mapRoll) return
        val cp = cos(Math.toRadians(pitch)); val sp = sin(Math.toRadians(pitch))
        val cr = cos(Math.toRadians(roll)); val sr = sin(Math.toRadians(roll))
        // NumPy builds Rx @ Rz in FP64, followed by the frozen FP32 map cast.
        val r = floatArrayOf(cr.toFloat(), (-sr).toFloat(), 0f,
            (cp * sr).toFloat(), (cp * cr).toFloat(), (-sp).toFloat(),
            (sp * sr).toFloat(), (sp * cr).toFloat(), cp.toFloat())
        val edge = EDGE.toFloat()
        val denominator = (2 * EDGE).toFloat()
        coverage.fill(0f)
        for (i in 0 until SAMPLES) {
            val px = pointX[i]; val py = pointY[i]; val pz = pointZ[i]
            val x = px * r[0] + py * r[3] + pz * r[6]
            val y = px * r[1] + py * r[4] + pz * r[7]
            val z = px * r[2] + py * r[5] + pz * r[8]
            val radius = sqrt(x * x + y * y + z * z)
            val zx = floor(((x / max(z, 1e-30f) + edge) / denominator * 8f).toDouble()).toInt()
            val zy = floor(((y / max(z, 1e-30f) + edge) / denominator * 8f).toDouble()).toInt()
            val bin = floor((radius / WIDTH.toFloat()).toDouble()).toInt()
            val index = (zy.coerceIn(0, 7) * 8 + zx.coerceIn(0, 7)) * 16 + bin.coerceIn(0, 15)
            indexes[i] = index
            val valid = z > 0 && zx in 0..7 && zy in 0..7 && bin in 0..15
            weights[i] = if (valid) (.0005f / 27f) / volumes[index] else 0f
            if (valid) coverage[i / 27] += 1f
        }
        for (i in coverage.indices) coverage[i] /= 27f
        mapPitch = pitch; mapRoll = roll
    }

    fun prepare(history: List<FloatArray>, pitchDeg: Double, rollDeg: Double): FloatArray {
        require(history.size in 1..8 && history.all { it.size == 1024 })
        configure(pitchDeg, rollDeg)
        val output = FloatArray(INPUT_SIZE)
        for (voxel in 0 until VOXELS) {
            var total = 0f
            var current = 0f
            for (frame in history) {
                current = 0f
                for (i in voxel * 27 until (voxel + 1) * 27) current += frame[indexes[i]] * weights[i]
                total += current
            }
            var count = 0f
            repeat(history.size) { count += coverage[voxel] }
            output[voxel] = sign(total) * ln1p(abs(total))
            output[VOXELS + voxel] = count / 8f
            output[2 * VOXELS + voxel] = sign(current) * ln1p(abs(current))
        }
        masks.copyInto(output, 3 * VOXELS)
        require(output.all { it.isFinite() }) { "Projected CNH overflow" }
        return output
    }
}
