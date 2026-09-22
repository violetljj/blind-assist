package com.linnan.blindassist.device.glasses

import android.graphics.Bitmap
import android.os.SystemClock
import com.linnan.blindassist.camera.AtomS3rClockSynchronizer
import com.linnan.blindassist.camera.AtomS3rMjpegFrameSource
import com.linnan.blindassist.risk.DemoTofCell
import com.linnan.blindassist.risk.TofCorridorDemo
import com.linnan.blindassist.vision.NativeImageVisionFrame
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/** Latest-only camera + independent ToF acquisition over the phone's private LAN/hotspot.
 * All sensor work is off the UI thread. No image alignment or simultaneous-exposure claim.
 * A copied bitmap is GC-owned so old Compose snapshots never reference recycled pool images.
 */
class HardwareWifiDemoClient(cameraEndpoint: String, tofEndpoint: String) : AutoCloseable {
    private val cameraEndpoint = HardwareWifiEndpoint.normalize(cameraEndpoint)
    private val tofEndpoint = HardwareWifiEndpoint.normalize(tofEndpoint)
    private val running = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private val workers = Executors.newFixedThreadPool(2)
    private val cameraGeneration = AtomicLong()
    private val connections = mutableSetOf<HttpURLConnection>()
    @Volatile private var source: AtomS3rMjpegFrameSource? = null
    @Volatile private var camera: CameraSample? = null
    @Volatile private var tof: TofSample? = null
    @Volatile private var cameraProblem = "等待相机握手"
    @Volatile private var tofProblem = "等待 8×8 ToF"
    @Volatile private var restartCamera = false
    @Volatile private var cameraStatusValid = false

    val diagnostics: HardwareWifiDiagnostics
        get() {
            val now = SystemClock.elapsedRealtime()
            val c = camera
            val t = tof
            return HardwareWifiDiagnostics(
                c?.let { it.ageAtReceiveMs + now - it.receivedMs },
                t?.let { it.ageAtReceiveMs + now - it.receivedMs },
                c?.clockErrorMs, c?.sequence, t?.sequence
            )
        }

    fun start() {
        check(!closed.get()) { "Client is closed" }
        if (!running.compareAndSet(false, true)) return
        workers.execute(::cameraSupervisor)
        workers.execute(::tofLoop)
    }

    /** Non-blocking immutable view; stale pairs clear both image and algorithm support. */
    fun poll(): HardwareDemoSnapshot {
        val now = SystemClock.elapsedRealtime()
        val c = camera
        val t = tof
        val freshCamera = c != null && HardwareWifiFreshness.fresh(c.receivedMs, c.ageAtReceiveMs, now)
        val freshTof = t != null && HardwareWifiFreshness.fresh(t.receivedMs, t.ageAtReceiveMs, now)
        val usable = running.get() && !closed.get() && cameraStatusValid && freshCamera && freshTof
        val cells = if (usable) t!!.cells else emptyList()
        val decision = TofCorridorDemo.evaluate(8, 8, cells, usable)
        val status = if (usable) {
            "无线直连 · 相机 ≤${c!!.ageAtReceiveMs + now - c.receivedMs} ms · " +
                "ToF ≤${t!!.ageAtReceiveMs + now - t.receivedMs} ms · 独立采样"
        } else {
            "UNKNOWN · " + when {
                !running.get() -> "无线连接已停止"
                !cameraStatusValid || !freshCamera -> cameraProblem.ifBlank { "相机数据过期" }
                else -> tofProblem.ifBlank { "ToF 数据过期" }
            }
        }
        return HardwareDemoSnapshot("live", "wifi:${c?.boot ?: "?"}:${t?.boot ?: "?"}",
            c?.sequence, t?.sequence, if (usable) c?.bitmap else null, cells, 8, 8,
            decision, usable, status)
    }

    private fun cameraSupervisor() {
        var boot: String? = null
        try {
            while (running.get()) {
                try {
                    val status = readJson(cameraEndpoint, "/api/status")
                    require(status.getString("role") == "camera")
                    val nextBoot = status.getString("boot_id").also { require(it.isNotBlank()) }
                    cameraStatusValid = true
                    if (boot != nextBoot || restartCamera || source == null) {
                        cameraGeneration.incrementAndGet()
                        camera = null
                        shutdownCamera(source)
                        if (!running.get()) break
                        boot = nextBoot
                        restartCamera = false
                        cameraProblem = "相机连接中 · 等待时钟与新帧"
                        startCamera(nextBoot)
                    }
                } catch (_: Exception) {
                    cameraStatusValid = false
                    camera = null
                    cameraProblem = "相机未连接 · 检查热点与设备电源"
                }
                if (!pause(750L)) break
            }
        } finally {
            shutdownCamera(source)
            source = null
        }
    }

    private fun startCamera(boot: String) {
        val generation = cameraGeneration.get()
        val progression = HardwareWifiProgress()
        val next = AtomS3rMjpegFrameSource(cameraEndpoint,
            clockSynchronizer = AtomS3rClockSynchronizer(cameraEndpoint, attempts = 2),
            maxFrameAgeMs = HardwareWifiFreshness.MAX_AGE_MS,
            shouldDecode = { running.get() && cameraGeneration.get() == generation })
        source = next
        next.start(previewView = null, onStarted = {}, onPreviewBitmap = null,
            onError = {
                if (cameraGeneration.get() == generation) {
                    camera = null
                    cameraProblem = "相机流中断 · 正在重连"
                    restartCamera = true
                }
            }, onFrame = { frame ->
                try {
                    if (running.get() && cameraGeneration.get() == generation) {
                        val stamp = requireNotNull(frame.frameStamp)
                        require(stamp.sourceId == "atoms3r-m12:$boot")
                        val timing = requireNotNull(frame.externalTiming)
                        val offset = timing.deviceMinusAndroidNs
                        val error = timing.clockSyncErrorBoundNs
                        val nowNs = SystemClock.elapsedRealtimeNanos()
                        // Without a mapping, TCP receipt alone cannot bound camera capture age.
                        if (offset == null || error == null) {
                            camera = null
                            cameraProblem = "相机时钟校验中 · UNKNOWN"
                        } else {
                            val capturedNs = timing.deviceCaptureNs - offset
                            val ageNs = nowNs - capturedNs
                            val upperNs = ageNs + error
                            require(ageNs >= -error && upperNs in 0..750_000_000L)
                            if (progression.accept(boot, stamp.frameId, timing.deviceCaptureNs / 1000L)) {
                                val bitmap = (frame as NativeImageVisionFrame).nativeImage as Bitmap
                                val copy = requireNotNull(bitmap.copy(Bitmap.Config.ARGB_8888, false))
                                val doneNs = SystemClock.elapsedRealtimeNanos()
                                val ageMs = (doneNs - capturedNs + error + 999_999L) / 1_000_000L
                                if (ageMs in 0..HardwareWifiFreshness.MAX_AGE_MS && running.get() &&
                                    cameraGeneration.get() == generation) {
                                    camera = CameraSample(boot, stamp.frameId, copy, doneNs / 1_000_000L,
                                        ageMs, (error + 999_999L) / 1_000_000L)
                                    cameraProblem = "相机数据过期 · 等待新帧"
                                }
                            }
                        }
                    }
                } catch (_: Exception) {
                    camera = null
                    cameraProblem = "相机时间或数据无效 · UNKNOWN"
                } finally { frame.close() }
            })
    }

    private fun tofLoop() {
        val progression = HardwareWifiProgress()
        var boot: String? = null
        while (running.get()) {
            try {
                val requested = SystemClock.elapsedRealtime()
                val json = readJson(tofEndpoint, "/api/tof")
                val received = SystemClock.elapsedRealtime()
                val nextBoot = json.getString("boot_id").also { require(it.isNotBlank()) }
                if (boot != nextBoot) { tof = null; boot = nextBoot }
                val sequence = json.getLong("seq")
                val sampled = json.getLong("sampled_us")
                val send = json.getLong("send_us")
                require(json.getInt("rows") == 8 && json.getInt("cols") == 8)
                val distances = json.getJSONArray("distance_mm")
                val statuses = json.getJSONArray("target_status")
                val targets = json.getJSONArray("nb_target")
                require(distances.length() == 64 && statuses.length() == 64 && targets.length() == 64)
                val cells = (0 until 64).map { zone ->
                    val range = distances.getDouble(zone).also { require(it.isFinite()) }
                    val status = statuses.getInt(zone).also { require(it in 0..255) }
                    val count = targets.getInt(zone).also { require(it in 0..255) }
                    DemoTofCell(zone, range, status, count, "KNOWN")
                }
                val age = requireNotNull(HardwareWifiFreshness.requestAgeUpperMs(sampled, send, requested, received))
                if (progression.accept(nextBoot, sequence, sampled)) {
                    if (running.get()) tof = TofSample(nextBoot, sequence, cells, received, age)
                    tofProblem = "ToF 数据过期 · 等待新测距"
                }
            } catch (_: Exception) {
                tof = null
                tofProblem = "ToF 未连接或读数失效 · UNKNOWN"
            }
            if (!pause(50L)) break
        }
    }

    private fun readJson(endpoint: String, path: String): JSONObject {
        val connection = URL(endpoint + path).openConnection() as HttpURLConnection
        synchronized(connections) {
            check(running.get())
            connections.add(connection)
        }
        return try {
            connection.connectTimeout = 500
            connection.readTimeout = 500
            connection.instanceFollowRedirects = false
            connection.useCaches = false
            connection.setRequestProperty("Cache-Control", "no-cache")
            require(connection.responseCode == 200)
            val bytes = connection.inputStream.use { it.readBytesBounded(24_000) }
            JSONObject(bytes.toString(Charsets.UTF_8))
        } finally {
            synchronized(connections) { connections.remove(connection) }
            connection.disconnect()
        }
    }

    private fun java.io.InputStream.readBytesBounded(limit: Int): ByteArray {
        val result = java.io.ByteArrayOutputStream()
        val buffer = ByteArray(2048)
        while (true) {
            val count = read(buffer)
            if (count < 0) break
            require(result.size() + count <= limit)
            result.write(buffer, 0, count)
        }
        return result.toByteArray()
    }

    private fun pause(ms: Long): Boolean = try {
        Thread.sleep(ms)
        running.get()
    } catch (_: InterruptedException) { Thread.currentThread().interrupt(); false }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        running.set(false)
        cameraGeneration.incrementAndGet()
        camera = null
        tof = null
        workers.shutdownNow()
        // Lifecycle callers may be on Android's main thread. Teardown is bounded but asynchronous.
        Thread({
            val active = synchronized(connections) { connections.toList() }
            active.forEach(HttpURLConnection::disconnect)
            source?.stop()
            try { workers.awaitTermination(2, TimeUnit.SECONDS) }
            catch (_: InterruptedException) { Thread.currentThread().interrupt() }
            shutdownCamera(source)
            source = null
        }, "hardware-wifi-close").apply { isDaemon = true; start() }
    }

    private fun shutdownCamera(value: AtomS3rMjpegFrameSource?) {
        // A cancelled supervisor still releases its decoder pool and camera worker executor.
        val interrupted = Thread.interrupted()
        try { value?.shutdown() } finally { if (interrupted) Thread.currentThread().interrupt() }
    }

    private data class CameraSample(val boot: String, val sequence: Long, val bitmap: Bitmap,
        val receivedMs: Long, val ageAtReceiveMs: Long, val clockErrorMs: Long)
    private data class TofSample(val boot: String, val sequence: Long, val cells: List<DemoTofCell>,
        val receivedMs: Long, val ageAtReceiveMs: Long)
}

data class HardwareWifiDiagnostics(val cameraAgeUpperMs: Long?, val tofAgeUpperMs: Long?,
    val cameraClockErrorMs: Long?, val cameraSequence: Long?, val tofSequence: Long?)
