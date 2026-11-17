package com.linnan.blindassist.vision

import android.app.Application
import android.graphics.Bitmap
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.floor

/** UniDepth-V2-S on the HTP for the Atom upright camera (direction A, 2026-11-18). Mirrors the research path
 * (official `infer` at 315,560 input pixels): RGB/255, ImageNet normalize, bilinear (align_corners=False, no
 * antialias) 640x480 to 658x490 (HxW), constant camera rays as the second input, pts_3d z bilinear back to 640x480.
 * Model and rays come from host/unidepth_atom_export_20261118.py; nothing is downloaded. */
class UniDepthSnpeRuntime(application: Application, modelFile: File, raysFile: File) : MetricDepthRuntime {
    private var handle: Long
    private val imageIndex: Int
    private var closed = false

    init {
        require(modelFile.isFile) { "UniDepth DLC missing: $modelFile" }
        require(raysFile.isFile) { "UniDepth rays missing: $raysFile" }
        System.loadLibrary("SnpeHtpV75Stub")
        System.loadLibrary("SnpeHtpV75CalculatorStub")
        System.loadLibrary("SnpeHtpPrepare")
        System.loadLibrary("dav2_snpe_jni")
        handle = nativeCreate(modelFile.absolutePath, Dav2SnpeDepthEstimator.skeletonDirectory(application).absolutePath)
        check(handle != 0L)
        try {
            val names = nativeInputNames(handle).toList()
            require(names.toSet() == setOf("rgbs", "rays")) { "Expected inputs rgbs and rays; got $names" }
            imageIndex = names.indexOf("rgbs")
            for (i in names.indices) require(nativeInputSize(handle, i) == 3 * H * W) { "Expected 3x${H}x$W inputs" }
            nativeSetInput(handle, names.indexOf("rays"), readRaw(raysFile, 3 * H * W))
        } catch (error: Throwable) {
            nativeDestroy(handle); handle = 0
            throw error
        }
    }

    @Synchronized
    override fun estimate(upright: Bitmap): FloatArray = postprocess(executeNchw(preprocessNchw(upright)))

    /** Raw model call on an already preprocessed NCHW input (fixture checks); returns pts_3d NCHW. */
    @Synchronized
    fun executeNchw(input: FloatArray): FloatArray {
        check(!closed) { "UniDepth runtime is closed" }
        require(input.size == 3 * H * W)
        nativeSetInput(handle, imageIndex, input)
        return nativeExecute(handle).also { require(it.size == 3 * H * W) { "Unexpected pts_3d size ${it.size}" } }
    }

    @Synchronized
    override fun close() {
        if (!closed) { closed = true; nativeDestroy(handle); handle = 0 }
    }

    private external fun nativeCreate(path: String, dspPath: String): Long
    private external fun nativeInputNames(handle: Long): Array<String>
    private external fun nativeInputSize(handle: Long, index: Int): Int
    private external fun nativeSetInput(handle: Long, index: Int, values: FloatArray)
    private external fun nativeExecute(handle: Long): FloatArray
    private external fun nativeDestroy(handle: Long)

    companion object {
        const val H = 658
        const val W = 490
        const val SRC_W = 480
        const val SRC_H = 640
        private val MEAN = floatArrayOf(.485f, .456f, .406f)
        private val STD = floatArrayOf(.229f, .224f, .225f)

        private class Axis(val i0: IntArray, val i1: IntArray, val w1: FloatArray)

        /** PyTorch F.interpolate bilinear, align_corners=False, no antialias: src = (dst + .5) * in / out - .5, clamped at 0. */
        private fun axis(input: Int, output: Int): Axis {
            val scale = input.toDouble() / output
            val i0 = IntArray(output); val i1 = IntArray(output); val w1 = FloatArray(output)
            for (d in 0 until output) {
                val src = maxOf(0.0, (d + .5) * scale - .5)
                val f = floor(src).toInt().coerceAtMost(input - 1)
                i0[d] = f; i1[d] = minOf(f + 1, input - 1); w1[d] = (src - f).toFloat()
            }
            return Axis(i0, i1, w1)
        }
        private val upX = axis(SRC_W, W)
        private val upY = axis(SRC_H, H)
        private val downX = axis(W, SRC_W)
        private val downY = axis(H, SRC_H)

        fun preprocessNchw(bitmap: Bitmap): FloatArray {
            require(bitmap.width == SRC_W && bitmap.height == SRC_H) { "Expected undistorted upright Atom 480x640" }
            val px = IntArray(SRC_W * SRC_H)
            bitmap.getPixels(px, 0, SRC_W, 0, 0, SRC_W, SRC_H)
            val src = Array(3) { FloatArray(SRC_W * SRC_H) }
            for (i in px.indices) {
                val p = px[i]
                src[0][i] = (((p shr 16) and 255) / 255f - MEAN[0]) / STD[0]
                src[1][i] = (((p shr 8) and 255) / 255f - MEAN[1]) / STD[1]
                src[2][i] = ((p and 255) / 255f - MEAN[2]) / STD[2]
            }
            val out = FloatArray(3 * H * W)
            for (c in 0..2) resize(src[c], SRC_W, upX, upY, out, c * H * W, W)
            return out
        }

        /** pts_3d NCHW to optical z (third channel) at 640x480 (HxW). */
        fun postprocess(pts: FloatArray): FloatArray {
            val z = pts.copyOfRange(2 * H * W, 3 * H * W)
            val out = FloatArray(SRC_W * SRC_H)
            resize(z, W, downX, downY, out, 0, SRC_W)
            return out
        }

        private fun resize(src: FloatArray, srcW: Int, ax: Axis, ay: Axis, out: FloatArray, offset: Int, outW: Int) {
            for (y in ay.i0.indices) {
                val r0 = ay.i0[y] * srcW; val r1 = ay.i1[y] * srcW; val wy = ay.w1[y]
                for (x in ax.i0.indices) {
                    val wx = ax.w1[x]
                    val top = src[r0 + ax.i0[x]] * (1 - wx) + src[r0 + ax.i1[x]] * wx
                    val bottom = src[r1 + ax.i0[x]] * (1 - wx) + src[r1 + ax.i1[x]] * wx
                    out[offset + y * outW + x] = top * (1 - wy) + bottom * wy
                }
            }
        }

        fun readRaw(file: File, count: Int): FloatArray {
            val bytes = file.readBytes()
            require(bytes.size == count * 4) { "Expected $count float32 in $file, got ${bytes.size} bytes" }
            return FloatArray(count).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
        }
    }
}
