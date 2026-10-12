package com.linnan.blindassist.capture

import com.linnan.blindassist.camera.AtomS3rMjpegFrameSource
import java.io.BufferedInputStream
import java.io.EOFException
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

interface AtomCameraCaptureSource : AutoCloseable {
    val transport: String
    fun start(onFrame: (AtomCameraCaptureFrame) -> Unit, onStatus: (CnhCaptureStatus) -> Unit)
}

/** Original JPEG and reported Atom timestamps. Receipt is not a synthetic exposure timestamp. */
data class AtomCameraCaptureFrame(
    val jpeg: ByteArray,
    val receivedElapsedNs: Long,
    val width: Int,
    val height: Int,
    val sequence: Long?,
    val bootId: String?,
    val deviceCaptureTimestampUs: Long?,
    val deviceJpegReadyTimestampUs: Long?,
    val headers: Map<String, String>,
    val sourceId: String,
)

/** Reuses the existing bounded Content-Length reader without starting its clock synchronizer. */
internal object AtomCameraCaptureReader {
    fun read(input: BufferedInputStream, endpoint: String, receivedAtNs: Long? = null): AtomCameraCaptureFrame {
        val packet = AtomS3rMjpegFrameSource.MjpegPartReader.readPacket(input, receivedAtNs, clockMapping = null)
        val (width, height) = dimensions(packet.jpeg)
        val boot = packet.headers["x-sequence-id"]
        fun timestamp(name: String) = packet.headers[name]?.toLongOrNull()?.takeIf { it >= 0 }
        return AtomCameraCaptureFrame(packet.jpeg, packet.jpegCompleteNs, width, height,
            timestamp("x-frame-sequence"), boot, timestamp("x-capture-timestamp-us"),
            timestamp("x-jpeg-ready-timestamp-us"), packet.headers.toMap(),
            "atom-mjpeg:$endpoint:boot:${boot ?: "unknown"}")
    }

    /** Read native JPEG SOF dimensions; pixels are decoded only by the preview UI. */
    private fun dimensions(jpeg: ByteArray): Pair<Int, Int> {
        fun u(index: Int) = jpeg[index].toInt() and 255
        require(jpeg.size >= 4 && u(0) == 255 && u(1) == 216 &&
            u(jpeg.lastIndex - 1) == 255 && u(jpeg.lastIndex) == 217) { "Incomplete Atom JPEG markers" }
        var position = 2
        while (position < jpeg.size - 2) {
            require(u(position++) == 255) { "Invalid JPEG segment marker" }
            while (position < jpeg.size && u(position) == 255) position++
            require(position < jpeg.size) { "Incomplete JPEG marker" }
            val marker = u(position++)
            if (marker == 0xd9 || marker == 0xda) break
            if (marker == 0x01 || marker in 0xd0..0xd8) continue
            require(position + 1 < jpeg.size) { "Incomplete JPEG segment length" }
            val length = u(position) * 256 + u(position + 1)
            require(length >= 2 && position + length <= jpeg.size) { "Incomplete JPEG segment" }
            if (marker in 0xc0..0xcf && marker != 0xc4 && marker != 0xc8 && marker != 0xcc) {
                require(length >= 8) { "Invalid JPEG SOF" }
                val height = u(position + 3) * 256 + u(position + 4)
                val width = u(position + 5) * 256 + u(position + 6)
                require(width > 0 && height > 0) { "Invalid JPEG dimensions" }
                return width to height
            }
            position += length
        }
        error("JPEG has no supported SOF dimensions")
    }
}

/** One explicit HTTP stream, no auto-reconnect and no device-clock mapping. */
class AtomMjpegCaptureSource internal constructor(
    host: String,
    private val port: Int,
    private val openConnection: (URL) -> HttpURLConnection,
    private val readFrame: (BufferedInputStream, String) -> AtomCameraCaptureFrame = { input, endpoint ->
        AtomCameraCaptureReader.read(input, endpoint)
    },
) : AtomCameraCaptureSource {
    constructor(host: String, port: Int = 81) : this(host, port, { it.openConnection() as HttpURLConnection })
    private val host = try { captureHost(host) } catch (error: IllegalArgumentException) {
        throw IllegalArgumentException("请输入 Atom 相机的局域网 IPv4 地址", error)
    }
    override val transport = "http://${this.host}:$port/stream"
    private val closed = AtomicBoolean()
    private val started = AtomicBoolean()
    private val worker = Executors.newSingleThreadExecutor { task ->
        Thread(task, "atom-jpeg-capture").also { workerThread = it }
    }
    @Volatile private var workerThread: Thread? = null
    @Volatile private var connection: HttpURLConnection? = null
    @Volatile private var input: BufferedInputStream? = null

    init { require(port in 1..65535) }

    override fun start(onFrame: (AtomCameraCaptureFrame) -> Unit, onStatus: (CnhCaptureStatus) -> Unit) {
        check(!closed.get() && started.compareAndSet(false, true)) { "Atom camera source already started or closed" }
        worker.execute {
            var terminal: CnhCaptureStatus? = null
            try {
                onStatus(CnhCaptureStatus(CnhConnectionState.CONNECTING, "连接 Atom 相机 $host:$port"))
                val http = openConnection(URL(transport)).apply {
                    connectTimeout = 2500
                    // Preview freshness is not a transport deadline. Allow a short
                    // network stall; the controller records/limits missing input.
                    readTimeout = 10000
                    useCaches = false
                    instanceFollowRedirects = false
                    setRequestProperty("Connection", "close")
                }
                connection = http
                if (!closed.get()) {
                    require(http.responseCode == HttpURLConnection.HTTP_OK) { "Atom HTTP ${http.responseCode}" }
                    require(http.contentType.orEmpty().startsWith("multipart/x-mixed-replace")) { "Atom is not an MJPEG stream" }
                    BufferedInputStream(http.inputStream, 64 * 1024).use { stream ->
                        input = stream
                        onStatus(CnhCaptureStatus(CnhConnectionState.CONNECTED, "Atom 相机已连接"))
                        while (!closed.get()) onFrame(readFrame(stream, transport))
                    }
                }
            } catch (error: Exception) {
                if (!closed.get()) terminal = CnhCaptureStatus(
                    if (error is EOFException) CnhConnectionState.DISCONNECTED else CnhConnectionState.ERROR,
                    "Atom 相机中断：${error.message ?: error.javaClass.simpleName}", captureFailureReason(error))
            } finally {
                input = null
                connection?.disconnect()
                connection = null
                val status = if (closed.get()) CnhCaptureStatus(CnhConnectionState.STOPPED, "Atom 相机已停止") else terminal
                if (status != null) runCatching { onStatus(status) }
                worker.shutdown()
            }
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        // Disconnect first: closing a BufferedInputStream can wait on the monitor
        // held by its blocking read. Abort the HTTP socket before taking that lock.
        runCatching { connection?.disconnect() }
        runCatching { input?.close() }
        // Closing the input cancels its read. A terminal callback may close other
        // sources on this worker, so do not interrupt that callback with shutdownNow.
        worker.shutdown()
        // A fully read JPEG may still be waiting for the controller's storage
        // lock. Drain its callback before the controller advances generation and
        // finalizes the session, even when another stream's fsync takes > 1.5 s.
        if (Thread.currentThread() != workerThread) worker.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)
    }
}
