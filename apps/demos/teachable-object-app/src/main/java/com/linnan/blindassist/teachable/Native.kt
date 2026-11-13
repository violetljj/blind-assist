package com.linnan.blindassist.teachable

import java.io.Closeable

internal object Native {
    init {
        System.loadLibrary("teachable_jni")
    }

    /** outputTensors: names to return; empty keeps SNPE's default of the last layer only. */
    external fun engineCreate(path: String, htp: Boolean, dspPath: String, outputTensors: Array<String>): Long
    external fun engineNames(handle: Long, outputs: Boolean): Array<String>
    external fun engineRun(handle: Long, inputs: Array<FloatArray>): Array<FloatArray>
    external fun engineDestroy(handle: Long)

    /**
     * patches: all teaching patches grouped by object ([sum(counts), dim]). simmax / simtop: handles of HTP engines for
     * the other-object max and the own top-6 terms (384-wide only), 0 for the CPU. Returns [objects, parts, dim] followed
     * by 3 timings in ms. The B CPU cutoff is exact for the retained rows and parts. Optional diagnostics record the
     * retained global patch indices (unused slots are -1) and counts of negative dots, positive dots and pruned rows.
     */
    private external fun buildMemory(patches: FloatArray, counts: IntArray, keep: Float, parts: Int, simmax: Long, simtop: Long,
                                     dim: Int, cutoff: Boolean, retained: IntArray?, diagnostics: LongArray?,
                                     cacheScores: FloatArray?, cacheWitness: IntArray?, cacheOrder: IntArray?, cachedObjects: Int,
                                     nextScores: FloatArray?, nextWitness: IntArray?, nextOrder: IntArray?): FloatArray

    /** Per-row scores and coverage, and the actual negative traversal order for each object. */
    class TeachingCache(counts: IntArray) {
        private val rows = counts.sum()
        val scores = FloatArray(2 * rows)
        val witness = IntArray(2 * rows)
        val order = IntArray(counts.sumOf { minOf(30_000, rows - it) })
        val bytes: Long get() = scores.size.toLong() * Float.SIZE_BYTES +
            (witness.size.toLong() + order.size) * Int.SIZE_BYTES
    }

    /** Memory and its timing breakdown (ms): other-object max, own top-6, k-means. */
    class Built(val parts: FloatArray, val otherMs: Double, val ownMs: Double, val kmeansMs: Double)

    fun build(patches: FloatArray, counts: IntArray, keep: Float, parts: Int, simmax: SnpeEngine? = null, simtop: SnpeEngine? = null,
              dim: Int = TeachablePipeline.D, cutoff: Boolean = false, retained: IntArray? = null,
              diagnostics: LongArray? = null, cache: TeachingCache? = null, cachedObjects: Int = 0,
              nextCache: TeachingCache? = null): Built {
        val r = buildMemory(patches, counts, keep, parts, simmax?.handle ?: 0L, simtop?.handle ?: 0L, dim, cutoff,
            retained, diagnostics, cache?.scores, cache?.witness, cache?.order, cachedObjects,
            nextCache?.scores, nextCache?.witness, nextCache?.order)
        val n = r.size - 3
        return Built(r.copyOf(n), r[n].toDouble(), r[n + 1].toDouble(), r[n + 2].toDouble())
    }

    /** CHW float image of size x size; returns the out x out roi_align crop (aligned, 2x2 samples per bin). */
    external fun roiAlign(image: FloatArray, size: Int, x0: Float, y0: Float, x1: Float, y1: Float, out: Int): FloatArray
}

/** One DLC on the CPU or the HTP; inputs and outputs by name. */
class SnpeEngine(path: String, htp: Boolean, dspPath: String, outputTensors: List<String> = emptyList()) : Closeable {
    internal val handle = Native.engineCreate(path, htp, dspPath, outputTensors.toTypedArray())
    val inputs: List<String> = Native.engineNames(handle, false).toList()
    val outputs: List<String> = Native.engineNames(handle, true).toList()

    fun run(byName: Map<String, FloatArray>): Map<String, FloatArray> {
        val result = Native.engineRun(handle, Array(inputs.size) { requireNotNull(byName[inputs[it]]) { "missing ${inputs[it]}" } })
        return outputs.zip(result).toMap()
    }

    override fun close() = Native.engineDestroy(handle)
}
