package com.linnan.blindassist.vision

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

fun ImageProxy.toArgbBitmap(): Bitmap {
    val plane = planes.first()
    val buffer = plane.buffer
    buffer.rewind()

    val bitmap = if (plane.pixelStride == RGBA_PIXEL_STRIDE && plane.rowStride == width * RGBA_PIXEL_STRIDE) {
        Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
            it.copyPixelsFromBuffer(buffer)
        }
    } else {
        copyRgbaWithStride(buffer, width, height, plane.rowStride, plane.pixelStride)
    }

    val rotation = imageInfo.rotationDegrees
    if (rotation == 0) return bitmap

    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
    val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    bitmap.recycle()
    return rotated
}

private fun copyRgbaWithStride(
    buffer: java.nio.ByteBuffer,
    width: Int,
    height: Int,
    rowStride: Int,
    pixelStride: Int
): Bitmap {
    val source = buffer.duplicate()
    val pixels = IntArray(width * height)
    for (y in 0 until height) {
        for (x in 0 until width) {
            val offset = y * rowStride + x * pixelStride
            if (offset + 3 >= source.capacity()) continue
            val r = source.get(offset).toInt() and 0xFF
            val g = source.get(offset + 1).toInt() and 0xFF
            val b = source.get(offset + 2).toInt() and 0xFF
            val a = source.get(offset + 3).toInt() and 0xFF
            pixels[y * width + x] = Color.argb(a, r, g, b)
        }
    }
    return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
}

private const val RGBA_PIXEL_STRIDE = 4
