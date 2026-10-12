package com.linnan.blindassist.capture

import java.io.InputStream
import java.net.SocketTimeoutException
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Bounded socket read-ahead, independent of decoder / journal fsync work.
 * Full queues apply backpressure; no received chunk is overwritten or dropped.
 * The clock is sampled immediately after read, never when the consumer catches up.
 */
internal class CaptureReadAhead(
    private val input: InputStream,
    private val receiptClock: () -> Long,
    threadName: String,
    capacity: Int = 8,
    chunkBytes: Int = 8192,
) : AutoCloseable {
    data class Chunk(val bytes: ByteArray, val receivedNs: Long)
    private data class Delivery(val chunk: Chunk? = null, val failure: Exception? = null)
    private val cancelled = AtomicBoolean()
    private val pending = ArrayBlockingQueue<Delivery>(capacity)
    private val reader: Thread

    init {
        require(chunkBytes > 0)
        reader = Thread({
            var failure: Exception? = null
            try {
                val buffer = ByteArray(chunkBytes)
                while (!cancelled.get()) {
                    val count = try { input.read(buffer) } catch (_: SocketTimeoutException) { continue }
                    if (count < 0) break
                    if (count == 0) continue
                    val receipt = receiptClock()
                    pending.put(Delivery(Chunk(buffer.copyOf(count), receipt)))
                }
            } catch (error: Exception) {
                if (!cancelled.get()) failure = error
            } finally {
                // EOF/error follows every accepted chunk, including a partial final line.
                if (!cancelled.get()) {
                    try { pending.put(Delivery(failure = failure)) }
                    catch (_: InterruptedException) { Thread.currentThread().interrupt() }
                }
            }
        }, threadName).apply { start() }
    }

    /** Returns null only after all queued chunks; transport failure is ordered the same way. */
    fun next(): Chunk? {
        val delivery = pending.take()
        delivery.failure?.let { throw it }
        return delivery.chunk
    }

    /**
     * Normal source stop closes its socket and drains next() through EOF/error first.
     * This abort path also releases a producer blocked on a full queue if the consumer
     * itself fails. That source reports ERROR; unsaved bytes are not called complete.
     */
    override fun close() {
        cancelled.set(true)
        runCatching { input.close() }
        reader.interrupt()
        if (Thread.currentThread() != reader) reader.join()
    }
}
