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

    /** patches: all teaching patches grouped by object ([sum(counts), 384]); returns [objects, parts, 384]. */
    external fun buildMemory(patches: FloatArray, counts: IntArray, keep: Float, parts: Int): FloatArray

    /** CHW float image of size x size; returns the out x out roi_align crop (aligned, 2x2 samples per bin). */
    external fun roiAlign(image: FloatArray, size: Int, x0: Float, y0: Float, x1: Float, y1: Float, out: Int): FloatArray
}

/** One DLC on the CPU or the HTP; inputs and outputs by name. */
class SnpeEngine(path: String, htp: Boolean, dspPath: String, outputTensors: List<String> = emptyList()) : Closeable {
    private val handle = Native.engineCreate(path, htp, dspPath, outputTensors.toTypedArray())
    val inputs: List<String> = Native.engineNames(handle, false).toList()
    val outputs: List<String> = Native.engineNames(handle, true).toList()

    fun run(byName: Map<String, FloatArray>): Map<String, FloatArray> {
        val result = Native.engineRun(handle, Array(inputs.size) { requireNotNull(byName[inputs[it]]) { "missing ${inputs[it]}" } })
        return outputs.zip(result).toMap()
    }

    override fun close() = Native.engineDestroy(handle)
}
