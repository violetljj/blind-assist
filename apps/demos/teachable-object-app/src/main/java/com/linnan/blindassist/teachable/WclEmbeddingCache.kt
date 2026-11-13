package com.linnan.blindassist.teachable

import android.util.AtomicFile
import java.io.DataInputStream
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** One ordered set of raw FP32 WCL frame embeddings per completed teaching object. */
internal object WclEmbeddingCache {
    const val FILE_NAME = "wcl-embeddings.bin"
    private const val MAGIC = 0x57434C45 // WCLE
    private const val VERSION = 1
    private const val SHA_BYTES = 32
    private const val HEADER_BYTES = 2 * Int.SIZE_BYTES + SHA_BYTES

    fun identity(modelIdentity: String, frames: List<File>): ByteArray {
        val identity = MessageDigest.getInstance("SHA-256")
        identity.update("$modelIdentity|${WclTeachingStore.PREPROCESSING}|".toByteArray(Charsets.UTF_8))
        val jpegHash = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(64 * 1024)
        for (frame in frames) {
            frame.inputStream().use { input ->
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    jpegHash.update(buffer, 0, count)
                }
            }
            identity.update(jpegHash.digest())
        }
        return identity.digest()
    }

    // AtomicFile requires external synchronization. All teaching activities run in the same process.
    @Synchronized
    fun read(file: File, identity: ByteArray, frames: Int): List<FloatArray>? {
        val byteCount = HEADER_BYTES + frames * WclPipeline.D * Float.SIZE_BYTES + SHA_BYTES
        return try {
            AtomicFile(file).openRead().use { input ->
                if (input.channel.size() != byteCount.toLong()) return null
                val bytes = ByteArray(byteCount).also { DataInputStream(input).readFully(it) }
                val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                if (buffer.int != MAGIC || buffer.int != VERSION) return null
                if (!ByteArray(SHA_BYTES).also { buffer.get(it) }.contentEquals(identity)) return null
                val checksumOffset = bytes.size - SHA_BYTES
                if (!checksum(bytes, checksumOffset).contentEquals(bytes.copyOfRange(checksumOffset, bytes.size))) return null
                List(frames) { FloatArray(WclPipeline.D) { Float.fromBits(buffer.int) } }
            }
        } catch (_: IOException) {
            null
        }
    }

    @Synchronized
    fun write(file: File, identity: ByteArray, rows: List<FloatArray>) {
        if (rows.any { it.size != WclPipeline.D || it.any { value -> !value.isFinite() } }) return
        val buffer = ByteBuffer.allocate(HEADER_BYTES + rows.size * WclPipeline.D * Float.SIZE_BYTES + SHA_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(MAGIC).putInt(VERSION).put(identity)
        for (row in rows) for (value in row) buffer.putInt(value.toRawBits())
        buffer.put(checksum(buffer.array(), buffer.position()))
        val atomic = AtomicFile(file)
        try {
            val output = atomic.startWrite()
            try {
                output.write(buffer.array())
                atomic.finishWrite(output)
            } catch (error: IOException) {
                atomic.failWrite(output)
                throw error
            }
        } catch (_: IOException) {
            // Embeddings can be regenerated, so a cache failure must not block saving teaching.
        }
    }

    private fun checksum(bytes: ByteArray, length: Int): ByteArray = MessageDigest.getInstance("SHA-256")
        .apply { update(bytes, 0, length) }.digest()
}
