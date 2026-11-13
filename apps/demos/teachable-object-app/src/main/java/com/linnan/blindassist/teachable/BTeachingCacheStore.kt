package com.linnan.blindassist.teachable

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.DigestInputStream
import java.security.DigestOutputStream
import java.security.MessageDigest

/** Versioned B append state in the application's private files directory. */
internal object BTeachingCacheStore {
    private const val MAGIC = 0x42434150 // BCAP
    private const val VERSION = 1
    private const val SHA_BYTES = 32
    private const val HEADER_BYTES = 4 * Int.SIZE_BYTES + SHA_BYTES
    private const val NEGATIVE_CAP = 30_000

    class Restored(val features: TeachablePipeline.TeachingFeatures, val cache: Native.TeachingCache)

    fun save(file: File, identity: String, dim: Int, features: TeachablePipeline.TeachingFeatures, cache: Native.TeachingCache) {
        val identityBytes = identityBytes(identity)
        val parent = requireNotNull(file.absoluteFile.parentFile)
        val temporary = File.createTempFile("b-teaching-", ".tmp", parent)
        try {
            FileOutputStream(temporary).use { raw ->
                val hash = MessageDigest.getInstance("SHA-256")
                val output = DataOutputStream(DigestOutputStream(raw, hash))
                output.writeInt(MAGIC)
                output.writeInt(VERSION)
                output.writeInt(dim)
                output.write(identityBytes)
                output.writeInt(features.counts.size)
                for (count in features.counts) output.writeInt(count)
                val blocks = Blocks()
                blocks.writeFloats(output, features.patches)
                for (cls in features.cls) blocks.writeFloats(output, cls)
                blocks.writeFloats(output, cache.scores)
                blocks.writeInts(output, cache.witness)
                blocks.writeInts(output, cache.order)
                output.flush()
                raw.write(hash.digest())
                raw.fd.sync()
            }
            Files.move(temporary.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } finally {
            Files.deleteIfExists(temporary.toPath())
        }
    }

    fun load(file: File, identity: String, dim: Int): Restored = FileInputStream(file).use { raw ->
        val hash = MessageDigest.getInstance("SHA-256")
        val checked = DigestInputStream(raw, hash)
        val input = DataInputStream(checked)
        require(input.readInt() == MAGIC && input.readInt() == VERSION) { "Unsupported B teaching cache format" }
        require(input.readInt() == dim) { "B teaching cache feature width differs" }
        val savedIdentity = ByteArray(SHA_BYTES).also(input::readFully)
        require(savedIdentity.contentEquals(identityBytes(identity))) { "B teaching cache identity differs" }
        val objects = input.readInt()
        require(objects in 2..TeachablePipeline.OMAX) { "Invalid B teaching cache object count" }
        val patchesPerFrame = TeachablePipeline.GRID * TeachablePipeline.GRID
        val counts = IntArray(objects) { input.readInt() }
        val rowCount = counts.sumOf(Int::toLong)
        require(counts.all { it > 0 && it % patchesPerFrame == 0 } && rowCount <= Int.MAX_VALUE / dim) {
            "Invalid B teaching cache patch counts"
        }
        val rows = rowCount.toInt()
        val negativeColumns = counts.sumOf { minOf(NEGATIVE_CAP, rows - it) }
        val payloadWords = counts.size + rowCount * dim + counts.size.toLong() * dim + 4 * rowCount + negativeColumns
        require(raw.channel.size() == HEADER_BYTES + payloadWords * Int.SIZE_BYTES + SHA_BYTES) {
            "B teaching cache size differs from its shape"
        }

        val blocks = Blocks()
        val patches = FloatArray(rows * dim).also { blocks.readFloats(input, it) }
        val cls = List(objects) { FloatArray(dim).also { blocks.readFloats(input, it) } }
        val cache = Native.TeachingCache(counts)
        blocks.readFloats(input, cache.scores)
        blocks.readInts(input, cache.witness)
        blocks.readInts(input, cache.order)
        checked.on(false)
        val savedHash = ByteArray(SHA_BYTES).also(input::readFully)
        require(hash.digest().contentEquals(savedHash) && input.read() == -1) { "B teaching cache checksum differs" }
        validateCoverage(counts, rows, cache)
        Restored(TeachablePipeline.TeachingFeatures(patches, counts, cls), cache)
    }

    private fun validateCoverage(counts: IntArray, rows: Int, cache: Native.TeachingCache) {
        val position = IntArray(rows)
        var rowOffset = 0
        var orderOffset = 0
        for (count in counts) {
            position.fill(-1)
            val endRow = rowOffset + count
            val columns = minOf(NEGATIVE_CAP, rows - count)
            for (column in 0 until columns) {
                val row = cache.order[orderOffset + column]
                require(row in 0 until rows && row !in rowOffset until endRow && position[row] == -1) {
                    "Invalid B teaching cache traversal order"
                }
                position[row] = column
            }
            for (row in rowOffset until endRow) {
                val witness = cache.witness[2 * row]
                val covered = cache.witness[2 * row + 1]
                require(witness in 0 until rows && covered in 1..columns && position[witness] in 0 until covered) {
                    "Invalid B teaching cache witness or covered prefix"
                }
            }
            rowOffset = endRow
            orderOffset += columns
        }
    }

    private fun identityBytes(identity: String): ByteArray {
        require(identity.length == 2 * SHA_BYTES && identity.all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' }) {
            "Expected a hexadecimal SHA-256 teaching identity"
        }
        return ByteArray(SHA_BYTES) { index -> identity.substring(2 * index, 2 * index + 2).toInt(16).toByte() }
    }

    /** Transfer primitive words in bounded chunks, without constructing another feature-sized array. */
    private class Blocks {
        private val buffer = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN)
        private val bytes = buffer.array()
        private val words = buffer.asIntBuffer()

        fun writeFloats(output: DataOutputStream, values: FloatArray) = chunks(values.size) { offset, length ->
            for (index in 0 until length) words.put(index, values[offset + index].toRawBits())
            output.write(bytes, 0, length * Int.SIZE_BYTES)
        }

        fun readFloats(input: DataInputStream, values: FloatArray) = chunks(values.size) { offset, length ->
            input.readFully(bytes, 0, length * Int.SIZE_BYTES)
            for (index in 0 until length) values[offset + index] = Float.fromBits(words.get(index))
        }

        fun writeInts(output: DataOutputStream, values: IntArray) = chunks(values.size) { offset, length ->
            words.position(0)
            words.put(values, offset, length)
            output.write(bytes, 0, length * Int.SIZE_BYTES)
        }

        fun readInts(input: DataInputStream, values: IntArray) = chunks(values.size) { offset, length ->
            input.readFully(bytes, 0, length * Int.SIZE_BYTES)
            words.position(0)
            words.get(values, offset, length)
        }

        private inline fun chunks(size: Int, transfer: (offset: Int, length: Int) -> Unit) {
            var offset = 0
            while (offset < size) {
                val length = minOf(size - offset, words.capacity())
                transfer(offset, length)
                offset += length
            }
        }
    }
}
