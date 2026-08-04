package com.linnan.blindassist.hftf

internal class Dav2Yuv420RgbConverter : AutoCloseable {
    val output = ByteArray(Dav2PreprocessContract.INPUT_BYTES)
    private var handle = nativeCreate()

    init { check(handle != 0L) { "unable to create native YUV converter" } }

    fun convert(frame: OwnedYuv420Frame): ByteArray {
        check(handle != 0L)
        nativeConvert(handle, frame.y, frame.u, frame.v, frame.width, frame.height,
            frame.rotationDegrees, output)
        return output
    }

    override fun close() {
        if (handle != 0L) nativeDestroy(handle)
        handle = 0L
    }

    private external fun nativeCreate(): Long
    private external fun nativeConvert(handle: Long, y: ByteArray, u: ByteArray, v: ByteArray,
        width: Int, height: Int, rotationDegrees: Int, output: ByteArray)
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
