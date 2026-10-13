package com.linnan.blindassist.capture

import org.json.JSONObject

/** Bounded diagnostic samples, independent of controller/storage locks. Never changes delivery. */
internal class CaptureCameraPerformance(private val capacity: Int = 2048) {
    private val metrics = linkedMapOf<String, Samples>()
    private var frames = 0L
    private var callbacks = 0L
    private var callbacksStarted = 0L
    private var jpegBytes = 0L
    private var sequenceGapFrames = 0L
    private var sequenceRegressions = 0L
    private var bootChanges = 0L
    private var receiptRegressions = 0L
    private var deviceRegressions = 0L
    private var queueHighWatermark = 0
    private var queueBlockedCount = 0L
    private var abandonedFrames = 0L
    private var previous: AtomCameraCaptureFrame? = null

    init { require(capacity > 0) }

    @Synchronized
    fun reset() {
        metrics.clear()
        frames = 0; callbacks = 0; callbacksStarted = 0; jpegBytes = 0; sequenceGapFrames = 0
        sequenceRegressions = 0; bootChanges = 0; receiptRegressions = 0; deviceRegressions = 0
        queueHighWatermark = 0; queueBlockedCount = 0; abandonedFrames = 0
        previous = null
    }

    @Synchronized
    fun received(frame: AtomCameraCaptureFrame, readStartedNs: Long, callbackStartedNs: Long) {
        frames++
        jpegBytes += frame.jpeg.size
        add("jpeg_bytes", frame.jpeg.size.toLong())
        elapsed("frame_read_ms", readStartedNs, callbackStartedNs)
        frame.readTiming?.let { timing ->
            elapsed("boundary_header_wait_ms", readStartedNs, timing.bodyReadStartNs)
            elapsed("jpeg_body_read_ms", timing.bodyReadStartNs, frame.receivedElapsedNs)
            elapsed("first_body_byte_wait_ms", timing.bodyReadStartNs, timing.firstByteNs)
            add("body_read_calls", timing.bodyReadCalls.toLong())
            add("max_body_read_gap_ms", timing.maxBodyReadGapNs)
        }
        val captureUs = frame.deviceCaptureTimestampUs
        val readyUs = frame.deviceJpegReadyTimestampUs
        if (captureUs != null && readyUs != null && readyUs >= captureUs) {
            add("device_capture_to_ready_ms", readyUs - captureUs)
        }
        previous?.let { before ->
            if (frame.receivedElapsedNs >= before.receivedElapsedNs) {
                elapsed("receipt_interval_ms", before.receivedElapsedNs, frame.receivedElapsedNs)
            } else receiptRegressions++
            val sameBoot = frame.bootId != null && frame.bootId == before.bootId
            if (frame.bootId != null && before.bootId != null && !sameBoot) bootChanges++
            if (sameBoot) {
                val oldSequence = before.sequence
                val sequence = frame.sequence
                if (sequence != null && oldSequence != null) {
                    if (sequence > oldSequence) sequenceGapFrames += sequence - oldSequence - 1
                    else sequenceRegressions++
                }
                val oldCaptureUs = before.deviceCaptureTimestampUs
                if (captureUs != null && oldCaptureUs != null) {
                    if (captureUs >= oldCaptureUs) add("device_capture_interval_ms", captureUs - oldCaptureUs)
                    else deviceRegressions++
                }
            }
        }
        // Retain metadata only: diagnostics must not keep a previous multi-megabyte JPEG alive.
        previous = frame.copy(jpeg = EMPTY_BYTES, headers = emptyMap())
    }

    @Synchronized
    fun callback(startedNs: Long, endedNs: Long) {
        callbacks++
        elapsed("callback_ms", startedNs, endedNs)
    }

    @Synchronized
    fun queued(depth: Int, startedNs: Long, endedNs: Long) {
        queueHighWatermark = maxOf(queueHighWatermark, depth)
        elapsed("queue_enqueue_wait_ms", startedNs, endedNs)
    }

    @Synchronized
    fun backpressure(capacity: Int) {
        queueBlockedCount++
        queueHighWatermark = maxOf(queueHighWatermark, capacity)
    }

    @Synchronized
    fun delivering(frame: AtomCameraCaptureFrame, startedNs: Long) {
        callbacksStarted++
        elapsed("receipt_to_callback_start_ms", frame.receivedElapsedNs, startedNs)
    }

    @Synchronized
    fun deliveryEnded() { abandonedFrames = (frames - callbacksStarted).coerceAtLeast(0) }

    @Synchronized
    fun receipt(): JSONObject = JSONObject().apply {
        put("supported", true)
        put("scope", "since_reset; complete JPEG receipt before callback; callback may include durable recording")
        put("sample_limit_per_metric", capacity)
        put("frames_received", frames)
        put("callbacks_started", callbacksStarted)
        put("callbacks_completed", callbacks)
        put("jpeg_bytes_total", jpegBytes)
        put("source_sequence_gap_frames", sequenceGapFrames)
        put("source_sequence_regressions", sequenceRegressions)
        put("boot_changes", bootChanges)
        put("receipt_clock_regressions", receiptRegressions)
        put("device_capture_clock_regressions", deviceRegressions)
        put("pending_queue_high_watermark", queueHighWatermark)
        put("pending_queue_backpressure_count", queueBlockedCount)
        put("queued_frames_abandoned_on_callback_failure", abandonedFrames)
        put("queue_capacity_frames", 8)
        put("boundary", "Android read/callback intervals are not exposure or display latency. " +
            "Body reads may use already buffered bytes. Device capture-to-ready is reported framebuffer " +
            "timestamp to esp_camera_fb_get return, including any residence; not isolated JPEG encoding. " +
            "HTTP sequence gaps do not count all sensor exposures. No device/Android clock mapping.")
        for ((name, samples) in metrics) {
            val scale = when {
                name.startsWith("device_") -> 1_000.0
                name.endsWith("_ms") -> 1_000_000.0
                else -> 1.0
            }
            put(name, samples.receipt(scale))
        }
    }

    private fun elapsed(name: String, start: Long, end: Long) {
        if (end >= start) add(name, end - start)
    }

    private fun add(name: String, value: Long) {
        if (value >= 0) metrics.getOrPut(name) { Samples(capacity) }.add(value)
    }

    private class Samples(capacity: Int) {
        private val values = LongArray(capacity)
        private var next = 0
        private var count = 0L
        private var sum = 0.0
        private var max = 0L

        fun add(value: Long) {
            values[next] = value
            next = (next + 1) % values.size
            count++
            sum += value
            max = maxOf(max, value)
        }

        fun receipt(scale: Double): JSONObject {
            val sorted = values.copyOf(minOf(count, values.size.toLong()).toInt()).sorted()
            fun percentile(p: Double): Double {
                val position = (sorted.size - 1) * p
                val low = position.toInt()
                val high = minOf(low + 1, sorted.lastIndex)
                return (sorted[low] + (sorted[high] - sorted[low]) * (position - low)) / scale
            }
            return JSONObject().apply {
                put("count", count)
                put("retained_samples", sorted.size)
                put("recent_p50", percentile(0.50))
                put("recent_p95", percentile(0.95))
                put("all_max", max / scale)
                put("all_mean", sum / count / scale)
            }
        }
    }

    private companion object { val EMPTY_BYTES = ByteArray(0) }
}
