package com.linnan.blindassist.vision

import android.graphics.Bitmap
import java.nio.ByteBuffer

internal object NativeBitmapPreprocessor {
    private val available: Boolean = runCatching {
        System.loadLibrary("blindassist_vision")
        true
    }.getOrDefault(false)

    fun writeArgbToFloat(bitmap: Bitmap, output: ByteBuffer): Boolean {
        if (!available || !output.isDirect || bitmap.config != Bitmap.Config.ARGB_8888) return false
        output.rewind()
        return runCatching { writeArgbToFloatNative(bitmap, output) }.getOrDefault(false).also {
            output.rewind()
        }
    }

    fun writePaddedArgbToFloat(
        bitmap: Bitmap,
        output: ByteBuffer,
        inputSize: Int,
        top: Int
    ): Boolean {
        if (
            !available || !output.isDirect || bitmap.config != Bitmap.Config.ARGB_8888 ||
            bitmap.width != inputSize || inputSize <= 0 || top < 0 || top + bitmap.height > inputSize
        ) return false
        output.rewind()
        return runCatching {
            writePaddedArgbToFloatNative(bitmap, output, inputSize, top)
        }.getOrDefault(false).also { output.rewind() }
    }

    private external fun writeArgbToFloatNative(bitmap: Bitmap, output: ByteBuffer): Boolean

    private external fun writePaddedArgbToFloatNative(
        bitmap: Bitmap,
        output: ByteBuffer,
        inputSize: Int,
        top: Int
    ): Boolean
}
