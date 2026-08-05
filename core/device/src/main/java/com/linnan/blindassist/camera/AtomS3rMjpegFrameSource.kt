package com.linnan.blindassist.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.camera.view.PreviewView
import com.linnan.blindassist.util.FatalThrowables
import com.linnan.blindassist.vision.FrameClockDomain
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

/** Reads AtomS3R MJPEG as a latest-only external frame source. */
class AtomS3rMjpegFrameSource(
    endpoint: String,
    private val connectionFactory: (String) -> HttpURLConnection = ::openConnection,
    private val executor: ExecutorService = Executors.newFixedThreadPool(2)
) : FrameSource {
    private val baseEndpoint = endpoint.trim().trimEnd('/')
    private val streamUrl = baseEndpoint.replace(Regex(":\\d+$"), "") + ":81/stream"
    private val lifecycleLock = java.lang.Object()
    private var generation = 0L
    private var running = false
    private var shutdownRequested = false
    private var activeConnection: HttpURLConnection? = null
    private var latestPacket: MjpegPacket? = null

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
                        val packet = MjpegPartReader.readPacket(input)
                        synchronized(lifecycleLock) {
                            if (!isCurrentLocked(session)) return
                            latestPacket = packet
                            deliveredPacket = true
                            lifecycleLock.notifyAll()
                        }
                    }
                }
            } catch (error: Throwable) {
                FatalThrowables.rethrowIfFatal(error)
                if (!deliveredPacket && isCurrent(session)) {
                    onError(error)
                    return
                }
                if (isCurrent(session)) {
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
                val bitmap = requireNotNull(BitmapFactory.decodeByteArray(packet.jpeg, 0, packet.jpeg.size)) {
                    "Unable to decode AtomS3R JPEG"
                }
                try {
                    val metadata = packet.metadata()
                    val frame = BitmapRgbaVisionFrame.from(
                        bitmap = bitmap,
                        frameStamp = metadata.frameStamp,
                        rangingSample = metadata.rangingSample
                    )
                    if (!started) {
                        started = true
                        onStarted()
                    }
                    onPreviewBitmap?.invoke(bitmap)
                    try {
                        onFrame(frame)
                    } catch (error: Throwable) {
                        frame.close()
                        throw error
                    }
                } finally {
                    bitmap.recycle()
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
        val rangingSample: RangingSample?
    )

    internal data class MjpegPacket(
        val headers: Map<String, String>,
        val jpeg: ByteArray,
        val receivedAtNs: Long
    ) {
        fun metadata(): PacketMetadata {
            val frameSequence = headers.requiredLong("x-frame-sequence")
            val captureNs = headers.requiredLong("x-capture-timestamp-us") * NANOS_PER_MICROSECOND
            val frameStamp = FrameStamp(
                frameId = frameSequence,
                capturedAtNs = captureNs,
                receivedAtNs = receivedAtNs,
                sourceId = "atoms3r-m12:${headers["x-sequence-id"] ?: "unknown"}",
                coordinateFrame = "atoms3r-m12:camera",
                clockDomain = FrameClockDomain.EXTERNAL_DEVICE_MONOTONIC_UNMAPPED
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
            return PacketMetadata(frameStamp, ranging)
        }
    }

    internal object MjpegPartReader {
        fun readPacket(
            input: BufferedInputStream,
            receivedAtNs: Long = SystemClock.elapsedRealtimeNanos()
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
            var offset = 0
            while (offset < length) {
                val read = input.read(jpeg, offset, length - offset)
                if (read < 0) throw EOFException("MJPEG ended inside JPEG")
                offset += read
            }
            return MjpegPacket(headers, jpeg, receivedAtNs)
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
        private const val BUFFER_SIZE = 64 * 1024
        private const val MAX_JPEG_BYTES = 2 * 1024 * 1024
        private const val MAX_HEADER_LINE_BYTES = 2048
        private const val RECONNECT_DELAY_MS = 500L
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
}
