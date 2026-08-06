package com.linnan.blindassist.hftf

import java.nio.ByteBuffer
import java.nio.ByteOrder

internal class Dav2Yuv420RgbConverter : AutoCloseable {
    val output = ByteArray(Dav2PreprocessContract.INPUT_BYTES)
    val directOutput: ByteBuffer = ByteBuffer.allocateDirect(Dav2PreprocessContract.INPUT_BYTES)
        .order(ByteOrder.nativeOrder())
    private var handle = nativeCreate()

    init { check(handle != 0L) { "unable to create native YUV converter" } }

    fun convert(frame: OwnedYuv420Frame): ByteArray {
        check(handle != 0L)
        nativeConvert(handle, frame.y, frame.u, frame.v, frame.width, frame.height,
            frame.rotationDegrees, output)
        return output
    }

    fun convertDirect(frame: OwnedYuv420Frame): ByteBuffer {
        check(handle != 0L)
        directOutput.clear()
        nativeConvertDirect(
            handle, frame.y, frame.u, frame.v, frame.width, frame.height,
            frame.rotationDegrees, directOutput,
        )
        directOutput.position(0)
        directOutput.limit(Dav2PreprocessContract.INPUT_BYTES)
        return directOutput
    }

    override fun close() {
        if (handle != 0L) nativeDestroy(handle)
        handle = 0L
    }

    private external fun nativeCreate(): Long
    private external fun nativeConvert(handle: Long, y: ByteArray, u: ByteArray, v: ByteArray,
        width: Int, height: Int, rotationDegrees: Int, output: ByteArray)
    private external fun nativeConvertDirect(handle: Long, y: ByteArray, u: ByteArray, v: ByteArray,
        width: Int, height: Int, rotationDegrees: Int, output: ByteBuffer)
    private external fun nativeDestroy(handle: Long)

    companion object { init { System.loadLibrary("dav2_preprocess_native") } }
}

internal class OwnedYuv420Frame(
    maxWidth: Int,
    maxHeight: Int,
    private val onRelease: (OwnedYuv420Frame) -> Unit,
) : AutoCloseable {
    val y = ByteArray(maxWidth * maxHeight)
    val u = ByteArray(maxWidth * maxHeight / 4)
    val v = ByteArray(maxWidth * maxHeight / 4)
    var width = 0
    var height = 0
    var rotationDegrees = 0
    var sensorTimestampNanos = 0L
    var receivedAtNanos = 0L
    var copyCompletedAtNanos = 0L
    var sequence = 0L
    var stage = ""
    var started = false
    private var leased = false

    fun lease(): OwnedYuv420Frame {
        check(!leased)
        leased = true
        started = false
        stage = ""
        return this
    }

    override fun close() {
        if (!leased) return
        leased = false
        onRelease(this)
    }
}
