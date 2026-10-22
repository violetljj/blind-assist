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
     * patches: all teaching patches grouped by object ([sum(counts), 384]). simmax / simtop: handles of HTP engines for
     * the other-object max and the own top-6 terms, 0 for the CPU. Returns [objects, parts, 384] followed by 3 timings in ms.
     */
    external fun buildMemory(patches: FloatArray, counts: IntArray, keep: Float, parts: Int, simmax: Long, simtop: Long): FloatArray

    /** Memory and its timing breakdown (ms): other-object max, own top-6, k-means. */
    class Built(val parts: FloatArray, val otherMs: Double, val ownMs: Double, val kmeansMs: Double)

    fun build(patches: FloatArray, counts: IntArray, keep: Float, parts: Int, simmax: SnpeEngine? = null, simtop: SnpeEngine? = null): Built {
        val r = buildMemory(patches, counts, keep, parts, simmax?.handle ?: 0L, simtop?.handle ?: 0L)
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
