package com.linnan.blindassist.hftf

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONObject

internal class Dav2QnnCachedContext(
    cachedDlcPath: String,
    nativeLibraryDir: String,
) : AutoCloseable {
    private var handle: Long
    val metadata: JSONObject
    val output: ByteBuffer
    var lastInputFnv1a64: Long = 0L
        private set

    init {
        System.loadLibrary("cdsprpc")
        handle = nativeCreate(
            cachedDlcPath,
            "$nativeLibraryDir/libQnnHtp.so",
            "$nativeLibraryDir/libQnnSystem.so",
        )
        check(handle != 0L)
        metadata = JSONObject(nativeMetadata(handle))
        output = ByteBuffer.allocateDirect(metadata.getInt("output_bytes")).order(ByteOrder.nativeOrder())
    }

    fun execute(input: ByteBuffer): ByteBuffer {
        check(handle != 0L)
        require(input.isDirect) { "QNN input must be a direct buffer" }
        require(input.limit() == metadata.getInt("input_bytes")) {
            "QNN input limit ${input.limit()} != ${metadata.getInt("input_bytes")}"
        }
        input.rewind()
        output.rewind()
        lastInputFnv1a64 = nativeExecute(handle, input, output)
        output.rewind()
        return output
    }

    override fun close() {
        if (handle != 0L) nativeDestroy(handle)
        handle = 0L
    }

    private external fun nativeCreate(cachedDlcPath: String, backendPath: String, systemPath: String): Long
    private external fun nativeMetadata(handle: Long): String
    private external fun nativeExecute(handle: Long, input: ByteBuffer, output: ByteBuffer): Long
    private external fun nativeDestroy(handle: Long)

    companion object {
        init {
            System.loadLibrary("dav2_preprocess_native")
        }
    }
}
