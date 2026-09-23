package com.linnan.blindassist.risk

import kotlin.math.*

/** Exact public-input LOCAL representation; nominal hardware registration is supplied by caller.
 * This model representation does not establish sensor registration or real-world accuracy. */
object HardwareLocalFeatures {
    const val WIDTH = 320
    const val HEIGHT = 180
    const val FEATURE_SIZE = 961
    private const val RAW_SIZE = 910
    private val focal = 320.0 / (2 * tan(Math.toRadians(50.0)))
    val queries: Array<DoubleArray> get() = arrayOf(
        doubleArrayOf(-.6, 0.0, .42, .9), doubleArrayOf(-.3, .3, .42, .9),
        doubleArrayOf(0.0, .6, .42, .9), doubleArrayOf(-.6, 0.0, -.2, .42),
        doubleArrayOf(-.3, .3, -.2, .42), doubleArrayOf(0.0, .6, -.2, .42))

    /** Packed 0xAARRGGBB RGB pixels, row-major, and public normalized range/valid/y0/x0/y1/x1. */
    fun extract(rgb: IntArray, tof: Array<DoubleArray>, queryIndices: IntArray = intArrayOf(0, 1, 2, 3, 4, 5)): Array<FloatArray> {
        require(rgb.size == WIDTH * HEIGHT && tof.size == 64)
        require(queryIndices.isNotEmpty() && queryIndices.all { it in 0..5 })
        val n = rgb.size
        val channels = Array(3) { c -> DoubleArray(n) { i -> ((rgb[i] ushr (16 - c * 8)) and 255) / 255.0 } }
        val gray = DoubleArray(n) { channels[0][it] * .299 + channels[1][it] * .587 + channels[2][it] * .114 }
        val gx = DoubleArray(n) { i ->
            val x = i % WIDTH
            abs(if (x == 0) gray[i + 1] - gray[i] else if (x == WIDTH - 1) gray[i] - gray[i - 1] else (gray[i + 1] - gray[i - 1]) / 2)
        }
        val gy = DoubleArray(n) { i ->
            val y = i / WIDTH
            abs(if (y == 0) gray[i + WIDTH] - gray[i] else if (y == HEIGHT - 1) gray[i] - gray[i - WIDTH] else (gray[i + WIDTH] - gray[i - WIDTH]) / 2)
        }
        fun stats(ids: IntArray): DoubleArray {
            val out = DoubleArray(10)
            if (ids.isEmpty()) return out
            for (c in 0..2) {
                var sum = 0.0
                for (i in ids) sum += channels[c][i]
                out[c] = sum / ids.size
                var squared = 0.0
                for (i in ids) squared += (channels[c][i] - out[c]).pow(2)
                out[c + 3] = sqrt(squared / ids.size)
            }
            for (i in ids) { out[6] += gx[i]; out[7] += gy[i] }
            out[6] /= ids.size; out[7] /= ids.size
            val sorted = DoubleArray(ids.size) { gray[ids[it]] }.apply { sort() }
            for ((slot, p) in listOf(8 to .1, 9 to .9)) {
                val at = (sorted.size - 1) * p
                val lo = floor(at).toInt(); val hi = ceil(at).toInt()
                out[slot] = sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo)
            }
            return out
        }
        val z = DoubleArray(64); val low = DoubleArray(64); val high = DoubleArray(64)
        val valid = BooleanArray(64)
        val owner = IntArray(n) { -1 }
        val common = DoubleArray(RAW_SIZE)
        var coverage = 0
        for (zone in 0 until 64) {
            val t = tof[zone]
            require(t.size == 6 && (t[1] == 0.0 || t[1] == 1.0))
            require((2..5).all { t[it].isFinite() && t[it] in 0.0..1.0 } && t[4] > t[2] && t[5] > t[3])
            val range = t[0] * 8
            valid[zone] = t[1] == 1.0 && range.isFinite() && range >= .1 && range < 8
            z[zone] = if (valid[zone]) range else 0.0
            val radius = .1 + 3 * (.01 + .02 * z[zone])
            low[zone] = if (valid[zone]) max(.1, z[zone] - radius) else 0.0
            high[zone] = if (valid[zone]) z[zone] + radius else 0.0
            val y0 = max(0, ceil(t[2] * HEIGHT - .5).toInt())
            val x0 = max(0, ceil(t[3] * WIDTH - .5).toInt())
            val y1 = min(HEIGHT, ceil(t[4] * HEIGHT - .5).toInt())
            val x1 = min(WIDTH, ceil(t[5] * WIDTH - .5).toInt())
            val ids = IntArray((y1 - y0) * (x1 - x0))
            var k = 0
            for (y in y0 until y1) for (x in x0 until x1) {
                val i = y * WIDTH + x
                require(owner[i] == -1) { "Overlapping public zone footprints" }
                owner[i] = zone; ids[k++] = i; coverage++
            }
            val offset = zone * 14
            common[offset] = if (valid[zone]) 1.0 else 0.0
            common[offset + 1] = z[zone] / 8; common[offset + 2] = low[zone] / 8; common[offset + 3] = high[zone] / 8
            stats(ids).copyInto(common, offset + 4)
        }
        stats(IntArray(n) { it }).copyInto(common, 896)
        val fixedQueries = queries
        return queryIndices.map { queryIndex ->
            val q = fixedQueries[queryIndex]
            val out = FloatArray(FEATURE_SIZE)
            common.forEachIndexed { i, v -> out[i] = v.toFloat() }
            q.forEachIndexed { i, v -> out[906 + i] = v.toFloat() }
            val buckets = Array(3) { IntArray(n) }; val counts = IntArray(3)
            for (i in 0 until n) {
                val zone = owner[i]
                if (zone < 0 || !valid[zone]) continue
                var entry = .3; var leave = 3.0
                for (axis in 0..1) {
                    val slope = if (axis == 0) (i % WIDTH + .5 - 160) / focal else (i / WIDTH + .5 - 90) / focal
                    val lo = q[axis * 2]; val hi = q[axis * 2 + 1]
                    val a: Double; val b: Double
                    if (slope != 0.0) { a = min(lo / slope, hi / slope); b = max(lo / slope, hi / slope) }
                    else if (lo <= 0 && hi >= 0) { a = Double.NEGATIVE_INFINITY; b = Double.POSITIVE_INFINITY }
                    else { a = Double.POSITIVE_INFINITY; b = Double.NEGATIVE_INFINITY }
                    entry = max(entry, a); leave = min(leave, b)
                }
                val possible = max(entry, low[zone]) <= min(leave, high[zone]) + 1e-12
                val definite = possible && low[zone] >= entry - 1e-12 && high[zone] <= leave + 1e-12
                val group = if (definite) 0 else if (possible) 1 else 2
                buckets[group][counts[group]++] = i
            }
            val means = Array(3) { DoubleArray(3) }
            for (g in 0..2) {
                val ids = buckets[g].copyOf(counts[g]); val stat = stats(ids); val offset = RAW_SIZE + g * 15
                stat.forEachIndexed { i, v -> out[offset + i] = v.toFloat() }
                means[g] = stat.copyOf(3)
                if (ids.isNotEmpty()) {
                    var sl = 0.0; var sh = 0.0; var sz = 0.0
                    for (i in ids) { val zone = owner[i]; sl += low[zone]; sh += high[zone]; sz += z[zone] }
                    val mean = sz / ids.size; var variance = 0.0
                    for (i in ids) variance += (z[owner[i]] - mean).pow(2)
                    val geometry = doubleArrayOf(sl / ids.size / 8, sh / ids.size / 8, mean / 8, sqrt(variance / ids.size) / 8, ids.size.toDouble() / max(1, coverage))
                    geometry.forEachIndexed { i, v -> out[offset + 10 + i] = v.toFloat() }
                }
            }
            for (c in 0..2) { out[955 + c] = (means[0][c] - means[2][c]).toFloat(); out[958 + c] = (means[1][c] - means[2][c]).toFloat() }
            require(out.all { it.isFinite() })
            out
        }.toTypedArray()
    }
}
