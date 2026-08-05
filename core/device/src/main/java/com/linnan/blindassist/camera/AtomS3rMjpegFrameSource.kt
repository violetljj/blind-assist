package com.linnan.blindassist.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.os.Trace
import androidx.camera.view.PreviewView
import com.linnan.blindassist.util.FatalThrowables
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.ExternalFrameTiming
import com.linnan.blindassist.vision.ExternalFrameTransportDiagnostics
import com.linnan.blindassist.vision.FrameStamp
import com.linnan.blindassist.vision.RangingSample
import com.linnan.blindassist.vision.VisionFrame
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.EOFException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/** Reads AtomS3R MJPEG as a latest-only external frame source. */
class AtomS3rMjpegFrameSource(
    endpoint: String,
    private val connectionFactory: (String) -> HttpURLConnection = ::openConnection,
    private val executor: ExecutorService = Executors.newFixedThreadPool(3),
    private val clockSynchronizer: AtomS3rClockSynchronizer = AtomS3rClockSynchronizer(endpoint)
) : FrameSource {
    private val baseEndpoint = endpoint.trim().trimEnd('/')
    private val streamUrl = baseEndpoint.replace(Regex(":\\d+$"), "") + ":81/stream"
    private val lifecycleLock = java.lang.Object()
    private var generation = 0L
    private var running = false
    private var shutdownRequested = false
    private var activeConnection: HttpURLConnection? = null
    private var latestPacket: MjpegPacket? = null
    @Volatile private var clockMapping: AtomS3rClockMapping? = null
    private val packetsRead = AtomicLong()
    private val latestPacketOverwrites = AtomicLong()
    private val reconnects = AtomicLong()
    private val streamErrors = AtomicLong()
    private val clockSyncSuccesses = AtomicLong()
    private val clockSyncFailures = AtomicLong()

    init {
        require(baseEndpoint.startsWith("http://") || baseEndpoint.startsWith("https://"))
    }

    override fun start(
        previewView: PreviewView?,
        onFrame: (VisionFrame) -> Unit,
        onStarted: () -> Unit,
        onError: (Throwable) -> Unit,
        onPreviewBitmap: ((Bitmap) -> Unit)?
    ) {
        val session = synchronized(lifecycleLock) {
            if (shutdownRequested || running) return
            running = true
            generation += 1L
            generation
        }
        executor.execute { readLoop(session, onError) }
        executor.execute { consumeLoop(session, onFrame, onStarted, onError, onPreviewBitmap) }
        executor.execute { clockSyncLoop(session) }
    }

    override fun stop() {
        val connection = synchronized(lifecycleLock) {
            generation += 1L
            running = false
            latestPacket = null
            lifecycleLock.notifyAll()
            activeConnection.also { activeConnection = null }
        }
        connection?.disconnect()
    }

    override fun shutdown() {
        val shouldShutdown = synchronized(lifecycleLock) {
            if (shutdownRequested) false else {
                shutdownRequested = true
                true
            }
        }
        stop()
        if (shouldShutdown) {
            executor.shutdown()
            if (!executor.awaitTermination(1L, TimeUnit.SECONDS)) executor.shutdownNow()
        }
    }

    private fun readLoop(session: Long, onError: (Throwable) -> Unit) {
        var deliveredPacket = false
        while (isCurrent(session)) {
            try {
                if (clockMapping == null) {
                    clockMapping = synchronizeClock()
                }
                val connection = connectionFactory(streamUrl)
                synchronized(lifecycleLock) {
                    if (!isCurrentLocked(session)) {
                        connection.disconnect()
                        return
                    }
                    activeConnection = connection
                }
                require(connection.responseCode == HttpURLConnection.HTTP_OK) {
                    "HTTP ${connection.responseCode} from $streamUrl"
                }
                require(connection.contentType.orEmpty().startsWith("multipart/x-mixed-replace")) {
                    "Unexpected MJPEG content type: ${connection.contentType}"
                }
                BufferedInputStream(connection.inputStream, BUFFER_SIZE).use { input ->
                    while (isCurrent(session)) {
                        val packet = MjpegPartReader.readPacket(input, clockMapping = clockMapping)
                        packetsRead.incrementAndGet()
                        synchronized(lifecycleLock) {
                            if (!isCurrentLocked(session)) return
                            if (latestPacket != null) latestPacketOverwrites.incrementAndGet()
                            latestPacket = packet
                            deliveredPacket = true
                            lifecycleLock.notifyAll()
                        }
                    }
                }
            } catch (error: Throwable) {
                FatalThrowables.rethrowIfFatal(error)
                if (!deliveredPacket && isCurrent(session)) {
                    streamErrors.incrementAndGet()
                    onError(error)
                    return
                }
                if (isCurrent(session)) {
                    reconnects.incrementAndGet()
                    try {
                        Thread.sleep(RECONNECT_DELAY_MS)
                    } catch (_: InterruptedException) {
                        Thread.currentThread().interrupt()
                        return
                    }
                }
            } finally {
                synchronized(lifecycleLock) {
                    activeConnection?.disconnect()
                    activeConnection = null
                }
            }
        }
    }

    private fun clockSyncLoop(session: Long) {
        while (isCurrent(session)) {
            try {
                Thread.sleep(CLOCK_SYNC_INTERVAL_MS)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return
            }
            if (isCurrent(session)) synchronizeClock()?.let { clockMapping = it }
        }
    }

    private fun synchronizeClock(): AtomS3rClockMapping? =
        runCatching(clockSynchronizer::synchronize)
            .onSuccess { clockSyncSuccesses.incrementAndGet() }
            .onFailure { clockSyncFailures.incrementAndGet() }
            .getOrNull()

    fun diagnostics(): Diagnostics = Diagnostics(
        packetsRead = packetsRead.get(),
        latestPacketOverwrites = latestPacketOverwrites.get(),
        reconnects = reconnects.get(),
        streamErrors = streamErrors.get(),
        clockSyncSuccesses = clockSyncSuccesses.get(),
        clockSyncFailures = clockSyncFailures.get(),
        currentClockMapping = clockMapping
    )

    data class Diagnostics(
        val packetsRead: Long,
        val latestPacketOverwrites: Long,
        val reconnects: Long,
        val streamErrors: Long,
        val clockSyncSuccesses: Long,
        val clockSyncFailures: Long,
        val currentClockMapping: AtomS3rClockMapping?
    )

    private fun consumeLoop(
        session: Long,
        onFrame: (VisionFrame) -> Unit,
        onStarted: () -> Unit,
        onError: (Throwable) -> Unit,
        onPreviewBitmap: ((Bitmap) -> Unit)?
    ) {
        var started = false
        while (isCurrent(session)) {
            val packet = synchronized(lifecycleLock) {
                while (isCurrentLocked(session) && latestPacket == null) lifecycleLock.wait()
                latestPacket.also { latestPacket = null }
            } ?: continue
            try {
                val decodeStartNs = SystemClock.elapsedRealtimeNanos()
                val bitmap = traced(TRACE_JPEG_DECODE) {
                    requireNotNull(BitmapFactory.decodeByteArray(packet.jpeg, 0, packet.jpeg.size)) {
                        "Unable to decode AtomS3R JPEG"
                    }
                }
                val decodeCompleteNs = SystemClock.elapsedRealtimeNanos()
                val metadata = packet.metadata(decodeStartNs, decodeCompleteNs)
                val frame = OwnedBitmapVisionFrame(
                    bitmap = bitmap,
                    frameStamp = metadata.frameStamp,
                    rangingSample = metadata.rangingSample,
                    externalTiming = metadata.externalTimingSeed.withRgbaComplete(decodeCompleteNs),
                    externalTransportDiagnostics = metadata.transportDiagnostics
                )
                try {
                    if (!started) {
                        started = true
                        onStarted()
                    }
                    onPreviewBitmap?.invoke(bitmap)
                    onFrame(frame)
                } catch (error: Throwable) {
                    frame.close()
                    throw error
                }
            } catch (error: Throwable) {
                FatalThrowables.rethrowIfFatal(error)
                if (isCurrent(session)) onError(error)
                return
            }
        }
    }

    private fun isCurrent(session: Long): Boolean = synchronized(lifecycleLock) {
        isCurrentLocked(session)
    }

    private fun isCurrentLocked(session: Long): Boolean =
        !shutdownRequested && running && generation == session

    internal data class PacketMetadata(
        val frameStamp: FrameStamp,
        val rangingSample: RangingSample?,
        val externalTimingSeed: ExternalTimingSeed,
        val transportDiagnostics: ExternalFrameTransportDiagnostics
    )

    internal data class ExternalTimingSeed(
        val deviceCaptureNs: Long,
        val deviceJpegReadyNs: Long,
        val deviceSendStartNs: Long?,
        val androidReadStartNs: Long,
        val androidFirstByteNs: Long,
        val androidJpegCompleteNs: Long,
        val androidDecodeStartNs: Long,
        val androidDecodeCompleteNs: Long,
        val clockMapping: AtomS3rClockMapping?
    ) {
        fun withRgbaComplete(androidRgbaCompleteNs: Long) = ExternalFrameTiming(
            deviceCaptureNs = deviceCaptureNs,
            deviceJpegReadyNs = deviceJpegReadyNs,
            deviceSendStartNs = deviceSendStartNs,
            androidReadStartNs = androidReadStartNs,
            androidFirstByteNs = androidFirstByteNs,
            androidJpegCompleteNs = androidJpegCompleteNs,
            androidDecodeStartNs = androidDecodeStartNs,
            androidDecodeCompleteNs = androidDecodeCompleteNs,
            androidRgbaCompleteNs = androidRgbaCompleteNs,
            deviceMinusAndroidNs = clockMapping?.deviceMinusAndroidNs,
            clockSyncRttNs = clockMapping?.roundTripNs,
            clockSyncErrorBoundNs = clockMapping?.errorBoundNs
        )
    }

    internal data class MjpegPacket(
        val headers: Map<String, String>,
        val jpeg: ByteArray,
        val readStartNs: Long,
        val firstByteNs: Long,
        val jpegCompleteNs: Long,
        val bodyReadCalls: Int,
        val maxBodyReadGapNs: Long,
        val clockMapping: AtomS3rClockMapping?
    ) {
        fun metadata(
            decodeStartNs: Long = jpegCompleteNs,
            decodeCompleteNs: Long = decodeStartNs
        ): PacketMetadata {
            val frameSequence = headers.requiredLong("x-frame-sequence")
            val captureNs = headers.requiredLong("x-capture-timestamp-us") * NANOS_PER_MICROSECOND
            val jpegReadyNs = headers.requiredLong("x-jpeg-ready-timestamp-us") * NANOS_PER_MICROSECOND
            val sendStartNs = headers["x-device-send-start-timestamp-us"]
                ?.toLongOrNull()?.times(NANOS_PER_MICROSECOND)
            val mappedCaptureNs = clockMapping?.deviceToAndroidNs(captureNs)
            val frameStamp = FrameStamp(
                frameId = frameSequence,
                capturedAtNs = mappedCaptureNs ?: captureNs,
                receivedAtNs = jpegCompleteNs,
                sourceId = "atoms3r-m12:${headers["x-sequence-id"] ?: "unknown"}",
                coordinateFrame = "atoms3r-m12:camera",
                clockDomain = if (clockMapping == null) {
                    FrameClockDomain.EXTERNAL_DEVICE_MONOTONIC_UNMAPPED
                } else {
                    FrameClockDomain.EXTERNAL_DEVICE_MONOTONIC_MAPPED_TO_ANDROID
                }
            )
            val tofTimestampUs = headers["x-tof-timestamp-us"]?.toLongOrNull() ?: 0L
            val ranging = if (tofTimestampUs > 0L) {
                val valid = headers["x-tof-valid"]?.toBooleanStrictOrNull() ?: false
                RangingSample(
                    sampledAtNs = tofTimestampUs * NANOS_PER_MICROSECOND,
                    valid = valid,
                    rangeMm = headers["x-tof-range-mm"]?.toIntOrNull()?.takeIf { valid },
                    ageAtFrameReadyNs = headers["x-tof-age-at-jpeg-ready-us"]
                        ?.toLongOrNull()?.times(NANOS_PER_MICROSECOND),
                    clockDomain = FrameClockDomain.EXTERNAL_DEVICE_MONOTONIC_UNMAPPED
                )
            } else null
            return PacketMetadata(
                frameStamp,
                ranging,
                ExternalTimingSeed(
                    deviceCaptureNs = captureNs,
                    deviceJpegReadyNs = jpegReadyNs,
                    deviceSendStartNs = sendStartNs,
                    androidReadStartNs = readStartNs,
                    androidFirstByteNs = firstByteNs,
                    androidJpegCompleteNs = jpegCompleteNs,
                    androidDecodeStartNs = decodeStartNs,
                    androidDecodeCompleteNs = decodeCompleteNs,
                    clockMapping = clockMapping
                ),
                ExternalFrameTransportDiagnostics(
                    jpegSizeBytes = jpeg.size,
                    wifiRssiDbm = headers["x-wifi-rssi-dbm"]?.toIntOrNull(),
                    previousFrameSequence = if (
                        headers["x-previous-response-write-valid"]?.toBooleanStrictOrNull() == true
                    ) headers["x-previous-frame-sequence"]?.toLongOrNull() else null,
                    previousResponseWriteDurationNs = if (
                        headers["x-previous-response-write-valid"]?.toBooleanStrictOrNull() == true
                    ) headers["x-previous-response-write-duration-us"]
                        ?.toLongOrNull()?.times(NANOS_PER_MICROSECOND) else null,
                    androidBodyReadCalls = bodyReadCalls,
                    androidMaxBodyReadGapNs = maxBodyReadGapNs
                )
            )
        }
    }

    internal object MjpegPartReader {
        fun readPacket(
            input: BufferedInputStream,
            receivedAtNs: Long? = null,
            clockMapping: AtomS3rClockMapping? = null
        ): MjpegPacket {
            var line: String
            do {
                line = readLine(input)
            } while (!line.startsWith("--"))
            val headers = linkedMapOf<String, String>()
            while (true) {
                line = readLine(input)
                if (line.isEmpty()) break
                val separator = line.indexOf(':')
                require(separator > 0) { "Malformed MJPEG header: $line" }
                headers[line.substring(0, separator).trim().lowercase(Locale.US)] =
                    line.substring(separator + 1).trim()
            }
            val length = headers["content-length"]?.toIntOrNull()
                ?: error("Missing Content-Length")
            require(length in 1..MAX_JPEG_BYTES) { "Invalid JPEG length: $length" }
            val jpeg = ByteArray(length)
            val readStartNs = receivedAtNs ?: SystemClock.elapsedRealtimeNanos()
            val first = input.read()
            if (first < 0) throw EOFException("MJPEG ended before JPEG")
            jpeg[0] = first.toByte()
            val firstByteNs = receivedAtNs ?: SystemClock.elapsedRealtimeNanos()
            var offset = 1
            var bodyReadCalls = 1
            var previousReadCompleteNs = firstByteNs
            var maxBodyReadGapNs = 0L
            while (offset < length) {
                val read = input.read(jpeg, offset, length - offset)
                if (read < 0) throw EOFException("MJPEG ended inside JPEG")
                val readCompleteNs = receivedAtNs ?: SystemClock.elapsedRealtimeNanos()
                maxBodyReadGapNs = maxOf(maxBodyReadGapNs, readCompleteNs - previousReadCompleteNs)
                previousReadCompleteNs = readCompleteNs
                bodyReadCalls += 1
                offset += read
            }
            val jpegCompleteNs = receivedAtNs ?: SystemClock.elapsedRealtimeNanos()
            return MjpegPacket(
                headers,
                jpeg,
                readStartNs,
                firstByteNs,
                jpegCompleteNs,
                bodyReadCalls,
                maxBodyReadGapNs,
                clockMapping
            )
        }

        private fun readLine(input: BufferedInputStream): String {
            val bytes = ByteArrayOutputStream(128)
            while (bytes.size() <= MAX_HEADER_LINE_BYTES) {
                val value = input.read()
                if (value < 0) throw EOFException("MJPEG stream ended")
                if (value == '\n'.code) break
                if (value != '\r'.code) bytes.write(value)
            }
            require(bytes.size() <= MAX_HEADER_LINE_BYTES) { "MJPEG header line too long" }
            return bytes.toString(StandardCharsets.US_ASCII.name())
        }
    }

    companion object {
        private const val TRACE_JPEG_DECODE = "BlindAssist.AtomS3rJpegDecode"
        private const val BUFFER_SIZE = 64 * 1024
        private const val MAX_JPEG_BYTES = 2 * 1024 * 1024
        private const val MAX_HEADER_LINE_BYTES = 2048
        private const val RECONNECT_DELAY_MS = 500L
        private const val CLOCK_SYNC_INTERVAL_MS = 30_000L
        private const val NANOS_PER_MICROSECOND = 1_000L

        private fun openConnection(url: String): HttpURLConnection =
            (URL(url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 3_000
                readTimeout = 5_000
                useCaches = false
                setRequestProperty("Connection", "close")
            }

        private fun Map<String, String>.requiredLong(name: String): Long =
            get(name)?.toLongOrNull() ?: error("Missing or invalid $name")
    }

    private inline fun <T> traced(name: String, block: () -> T): T {
        val tracing = try {
            Trace.beginSection(name)
            true
        } catch (_: RuntimeException) {
            false
        }
        return try {
            block()
        } finally {
            if (tracing) {
                try {
                    Trace.endSection()
                } catch (_: RuntimeException) {
                    // Android Trace is unavailable in local JVM tests.
                }
            }
        }
    }
}
