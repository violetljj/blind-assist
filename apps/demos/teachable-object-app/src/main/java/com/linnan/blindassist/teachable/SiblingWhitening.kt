package com.linnan.blindassist.teachable

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.math.sqrt

/**
 * Sibling-group whitening of the WCL prototype scores (Graduation orbit_wcl_whiten_20261108/group_white.py, toybox_samecolour_20261108).
 * Objects whose mean prototypes have cosine >= [tau] form groups (connected components). Inside a group the teaching embeddings give
 * mu and C; with lambda = tr(C)/D and L L^T = C + lambda I, members are re-ordered by the cosine of L^-1 (x - mu) to L^-1 (M_k - mu)
 * (the same cosines as the symmetric inverse root). Their scores become maxRaw + r * (w - maxW), r = sd(raw) / sd(whitened) of the
 * group's teaching frames, so groups still compete on the raw cosine and the scores keep the raw scale for [ConfidenceEma].
 * No groups: the scores are returned unchanged.
 */
class SiblingWhitening private constructor(val tau: Double, val groups: List<Group>) {
    class Group(val members: IntArray, val mu: FloatArray, val lower: FloatArray, val whitenedPrototypes: Array<FloatArray>, val r: Double)

    /** [raw] = cosine of [embedding] to the normalised mean prototypes; returns a new array. */
    fun apply(embedding: FloatArray, raw: FloatArray): FloatArray {
        if (groups.isEmpty()) return raw
        val out = raw.copyOf()
        for (g in groups) {
            val y = solve(g.lower, DoubleArray(D) { (embedding[it] - g.mu[it]).toDouble() }); normalise(y)
            val w = DoubleArray(g.members.size) { k -> var s = 0.0; val p = g.whitenedPrototypes[k]; for (d in 0 until D) s += y[d] * p[d]; s }
            val maxRaw = g.members.maxOf { raw[it] }.toDouble(); val maxW = w.max()
            for ((k, o) in g.members.withIndex()) out[o] = (maxRaw + g.r * (w[k] - maxW)).toFloat()
        }
        return out
    }

    fun write(output: OutputStream, identity: String) {
        val out = DataOutputStream(output.buffered())
        out.writeInt(MAGIC); out.writeInt(VERSION); out.writeUTF(identity); out.writeDouble(tau); out.writeInt(D); out.writeInt(groups.size)
        for (g in groups) {
            out.writeInt(g.members.size); g.members.forEach(out::writeInt); out.writeDouble(g.r)
            g.mu.forEach(out::writeFloat); g.lower.forEach(out::writeFloat); g.whitenedPrototypes.forEach { p -> p.forEach(out::writeFloat) }
        }
        out.flush()
    }

    companion object {
        const val DEFAULT_TAU = 0.70
        private const val D = WclPipeline.D
        private const val MAGIC = 0x53574854
        private const val VERSION = 1

        /** [teach]: per object, its L2-normalised teaching embeddings. tau < 0 puts every object in one group (full panel). */
        fun build(teach: List<List<FloatArray>>, tau: Double = DEFAULT_TAU): SiblingWhitening {
            val n = teach.size
            val means = Array(n) { o -> DoubleArray(D).also { m -> for (e in teach[o]) for (d in 0 until D) m[d] += e[d].toDouble(); for (d in 0 until D) m[d] /= teach[o].size } }
            val protos = Array(n) { means[it].copyOf().also(::normalise) }
            val root = IntArray(n) { it }
            fun find(i: Int): Int { var x = i; while (root[x] != x) { root[x] = root[root[x]]; x = root[x] }; return x }
            for (i in 0 until n) for (j in i + 1 until n) if (tau < 0 || dot(protos[i], protos[j]) >= tau) root[find(i)] = find(j)
            val groups = (0 until n).groupBy(::find).values.filter { it.size > 1 }.map { members ->
                val rows = members.flatMap { teach[it] }; val count = rows.size
                val mu = DoubleArray(D); for (e in rows) for (d in 0 until D) mu[d] += e[d].toDouble(); for (d in 0 until D) mu[d] /= count
                val c = DoubleArray(D * D)                                     // lower triangle of the centred covariance
                val x = DoubleArray(D)
                for (e in rows) {
                    for (d in 0 until D) x[d] = e[d] - mu[d]
                    for (i in 0 until D) { val xi = x[i]; val base = i * D; for (j in 0..i) c[base + j] += xi * x[j] }
                }
                var trace = 0.0; for (i in 0 until D) trace += c[i * D + i] / count
                val lambda = trace / D
                for (i in 0 until D) { val base = i * D; for (j in 0..i) c[base + j] /= count; c[base + i] += lambda }
                cholesky(c)
                val lower = FloatArray(D * (D + 1) / 2).also { l -> var k = 0; for (i in 0 until D) for (j in 0..i) l[k++] = c[i * D + j].toFloat() }
                val wp = Array(members.size) { k -> solve(lower, DoubleArray(D) { means[members[k]][it] - mu[it] }).also(::normalise) }
                val rawT = ArrayList<Double>(); val whiteT = ArrayList<Double>()
                for (e in rows) {
                    val y = solve(lower, DoubleArray(D) { e[it] - mu[it] }); normalise(y)
                    for ((k, o) in members.withIndex()) { var s = 0.0; for (d in 0 until D) s += e[d] * protos[o][d]; rawT += s; whiteT += dot(y, wp[k]) }
                }
                Group(members.toIntArray(), FloatArray(D) { mu[it].toFloat() }, lower, Array(members.size) { k -> FloatArray(D) { wp[k][it].toFloat() } }, sd(rawT) / sd(whiteT))
            }
            return SiblingWhitening(tau, groups)
        }

        fun read(input: InputStream, identity: String): SiblingWhitening? {
            val inp = DataInputStream(input.buffered())
            if (inp.readInt() != MAGIC || inp.readInt() != VERSION || inp.readUTF() != identity) return null
            val tau = inp.readDouble(); check(inp.readInt() == D) { "whitening dimension" }
            val groups = List(inp.readInt()) {
                val k = inp.readInt(); val members = IntArray(k) { inp.readInt() }; val r = inp.readDouble()
                val mu = FloatArray(D) { inp.readFloat() }; val lower = FloatArray(D * (D + 1) / 2) { inp.readFloat() }
                Group(members, mu, lower, Array(k) { FloatArray(D) { inp.readFloat() } }, r)
            }
            return SiblingWhitening(tau, groups)
        }

        /** In-place Cholesky of a full D x D array holding the lower triangle (row-major). */
        private fun cholesky(a: DoubleArray) {
            for (j in 0 until D) {
                var s = a[j * D + j]; for (k in 0 until j) s -= a[j * D + k] * a[j * D + k]
                check(s > 0) { "covariance not positive definite" }
                val ljj = sqrt(s); a[j * D + j] = ljj
                for (i in j + 1 until D) { var t = a[i * D + j]; for (k in 0 until j) t -= a[i * D + k] * a[j * D + k]; a[i * D + j] = t / ljj }
            }
        }

        /** Forward substitution with the packed lower triangle: returns L^-1 b. */
        private fun solve(lower: FloatArray, b: DoubleArray): DoubleArray {
            val y = DoubleArray(D); var k = 0
            for (i in 0 until D) { var s = b[i]; for (j in 0 until i) s -= lower[k + j] * y[j]; y[i] = s / lower[k + i]; k += i + 1 }
            return y
        }

        private fun dot(a: DoubleArray, b: DoubleArray): Double { var s = 0.0; for (d in a.indices) s += a[d] * b[d]; return s }

        private fun normalise(v: DoubleArray) { val n = sqrt(dot(v, v)).coerceAtLeast(1e-12); for (d in v.indices) v[d] /= n }

        private fun sd(v: List<Double>): Double { val m = v.average(); return sqrt(v.sumOf { (it - m) * (it - m) } / v.size) }
    }
}
