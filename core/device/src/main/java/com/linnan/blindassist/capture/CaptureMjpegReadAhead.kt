package com.linnan.blindassist.capture

import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean

/** Complete JPEGs keep their reader receipt clock while durable callbacks run independently. */
internal class CaptureMjpegReadAhead(
    private val readFrame: () -> AtomCameraCaptureFrame,
    private val shouldStop: () -> Boolean,
    private val preview: (AtomCameraCaptureFrame) -> Unit,
    private val clock: () -> Long,
    private val performance: CaptureCameraPerformance,
    capacity: Int = 8,
) : AutoCloseable {
    private data class Delivery(val frame: AtomCameraCaptureFrame? = null, val failure: Exception? = null)
    private val pending = ArrayBlockingQueue<Delivery>(capacity)
    private val cancelled = AtomicBoolean()
    private var ended = false // Only the serial callback worker consumes, including a reentrant close drain.
    private val reader = Thread({
        var failure: Exception? = null
        try {
            while (!cancelled.get() && !shouldStop()) {
                val startedNs = clock()
                val frame = readFrame()
                performance.received(frame, startedNs, clock())
                // Preview is latest-only; saved delivery remains FIFO and never overwrites.
                preview(frame)
                val queuedNs = clock()
                val delivery = Delivery(frame)
                val blocked = !pending.offer(delivery)
                if (blocked) {
                    performance.backpressure(capacity)
                    pending.put(delivery)
                }
                performance.queued(if (blocked) capacity else pending.size, queuedNs, clock())
            }
        } catch (error: Exception) {
            if (!cancelled.get() && !shouldStop()) failure = error
        } finally {
            // EOF and read failures follow every fully read JPEG. Normal stop drains this marker.
            if (!cancelled.get()) {
                try { pending.put(Delivery(failure = failure)) }
                catch (_: InterruptedException) { Thread.currentThread().interrupt() }
            }
        }
    }, "atom-jpeg-reader").apply { start() }

    fun next(): AtomCameraCaptureFrame? {
        if (ended) return null
        val delivery = pending.take()
        if (delivery.frame == null) ended = true
        delivery.failure?.let { throw it }
        return delivery.frame
    }

    /** Caller first disconnects HTTP to release a socket read; error paths also release a full queue. */
    override fun close() {
        cancelled.set(true)
        reader.interrupt()
        if (Thread.currentThread() != reader) reader.join()
        // Also includes a fully read producer frame interrupted while putting into a full queue.
        performance.deliveryEnded()
        pending.clear()
    }
}
