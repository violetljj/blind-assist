package com.linnan.blindassist

import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import com.linnan.blindassist.capture.AtomMjpegCaptureSource
import com.linnan.blindassist.capture.PhoneCaptureController
import com.linnan.blindassist.capture.PhoneCaptureState
import com.linnan.blindassist.capture.PhoneCaptureStore
import com.linnan.blindassist.capture.PhoneCaptureSummary
import com.linnan.blindassist.capture.TcpCnhCaptureSource
import com.linnan.blindassist.capture.TcpImuCaptureSource
import com.linnan.blindassist.device.glasses.HardwareWifiDiscovery
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import org.json.JSONObject
import java.net.URI
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.ceil

@Immutable
internal data class AtomPreviewFrame(val image: ImageBitmap, val receiptNs: Long, val sequence: Long?)

class PhoneCaptureActivity : ComponentActivity() {
    private lateinit var capture: PhoneCaptureController
    private val previewState = mutableStateOf<AtomPreviewFrame?>(null)
    private val previewProvider: () -> AtomPreviewFrame? = { previewState.value }
    private val previewMetrics = CapturePreviewMetrics()
    private val onPreviewDraw: (Long) -> Unit = { previewMetrics.drawn(it, SystemClock.elapsedRealtimeNanos()) }
    private val previewGeneration = AtomicLong()
    private val latestDecoded = AtomicReference<Pair<Long, AtomPreviewFrame>?>(null)
    private val previewPostQueued = AtomicBoolean()
    private val handler = Handler(Looper.getMainLooper())
    private val io = Executors.newSingleThreadExecutor()
    private val controls = Executors.newSingleThreadExecutor { Thread(it, "capture-controls") }
    private val previewWorker = Executors.newSingleThreadScheduledExecutor { Thread(it, "capture-preview") }
    private val hudWorker = Executors.newSingleThreadScheduledExecutor { Thread(it, "capture-hud") }
    private val controlGeneration = AtomicLong()
    private val pendingControls = AtomicInteger()
    private var previewTask: ScheduledFuture<*>? = null
    private var hudTask: ScheduledFuture<*>? = null
    private var state by mutableStateOf(PhoneCaptureState())
    private var sessions by mutableStateOf(emptyList<PhoneCaptureSummary>())
    private var host by mutableStateOf("192.168.5.10")
    private var atomHost by mutableStateOf("192.168.5.9")
    private var note by mutableStateOf("")
    private var busy by mutableStateOf(false)
    private var controlBusy by mutableStateOf(false)
    private var discovering by mutableStateOf(false)
    @Volatile private var foreground = false
    private var exportMessage by mutableStateOf("")
    private var exportSessionId: String? = null

    private val export = registerForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        val sessionId = exportSessionId
        exportSessionId = null
        if (uri != null && sessionId != null) {
            busy = true
            exportMessage = "正在导出…"
            io.execute {
                val startNs = SystemClock.elapsedRealtimeNanos()
                val result = runCatching {
                    val session = PhoneCaptureStore.listSessions(this).single { it.id == sessionId }
                    requireNotNull(contentResolver.openOutputStream(uri, "wt")).use { output ->
                        PhoneCaptureStore.exportZip(session, output) { written, total ->
                            handler.post {
                                if (!isDestroyed) exportMessage = "正在导出 ${if (total > 0) written * 100 / total else 100}% · ${written / 1024 / 1024} / ${total / 1024 / 1024} MB"
                            }
                        }
                    }
                }
                handler.post {
                    if (!isDestroyed) {
                        busy = false
                        val seconds = (SystemClock.elapsedRealtimeNanos() - startNs) / 1e9
                        exportMessage = result.fold({ "导出完成：$sessionId.zip（${"%.1f".format(seconds)} 秒）" }, { "导出失败，请重新保存：${it.message}" })
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        capture = PhoneCaptureController(this)
        val preferences = getSharedPreferences("phone_capture", MODE_PRIVATE)
        host = preferences.getString("host", "192.168.5.10")!!
        atomHost = preferences.getString("atom_host", preferences.getString("imu_host", "192.168.5.9"))!!
        note = savedInstanceState?.getString("capture_note") ?: ""
        exportSessionId = savedInstanceState?.getString("export_session_id")
        capture.setImuEnabled(preferences.getBoolean("imu_enabled", true))
        state = capture.snapshot()
        refreshSessions()
        setContent {
            BlindAssistTheme {
                PhoneCaptureScreen(state, previewProvider, onPreviewDraw, host, atomHost, note, sessions,
                    busy || controlBusy || discovering, exportMessage,
                    onHostChange = { host = it }, onAtomHostChange = { atomHost = it }, onNoteChange = { note = it },
                    onImuEnabledChange = { enabled ->
                        runControl("IMU 设置失败") {
                            capture.setImuEnabled(enabled)
                            preferences.edit().putBoolean("imu_enabled", enabled).apply()
                        }
                    },
                    onConnect = ::connectDevices, onDiscover = ::discoverDevices,
                    onDisconnect = { invalidatePreview(); runControl("断开失败") { capture.disconnect() } },
                    onStart = {
                        val sceneNote = note
                        runControl("无法开始") { capture.startRecording(sceneNote) }
                    },
                    onStop = {
                        invalidatePreview()
                        val tofAddress = host.trim(); val atomAddress = atomHost.trim(); val imuEnabled = state.imuEnabled
                        runControl("停止失败", onComplete = { refreshSessions() }) { generation ->
                            capture.stopRecording()
                            if (isCurrent(generation)) connectSources(generation, tofAddress, atomAddress, imuEnabled)
                        }
                    },
                    onExport = { session -> exportSessionId = session.id; export.launch("${session.id}.zip") },
                    onBack = { finish() })
            }
        }
    }

    override fun onResume() {
        super.onResume()
        foreground = true
        hudTask?.cancel(false)
        hudTask = hudWorker.scheduleWithFixedDelay({
            if (!foreground || pendingControls.get() != 0) return@scheduleWithFixedDelay
            val generation = controlGeneration.get()
            val next = runCatching { capture.snapshot() }.getOrNull() ?: return@scheduleWithFixedDelay
            handler.post {
                if (isCurrent(generation)) {
                    val wasRecording = state.recording
                    state = next
                    if (wasRecording && !next.recording) refreshSessions()
                }
            }
        }, 0, 200, TimeUnit.MILLISECONDS)
        previewTask?.cancel(false)
        var decodedReceiptNs = 0L
        previewTask = previewWorker.scheduleAtFixedRate({
            if (!foreground) return@scheduleAtFixedRate
            val generation = previewGeneration.get()
            val frame = capture.peekCameraFrame()
            if (frame == null) {
                if (previewState.value != null) handler.post {
                    if (foreground && generation == previewGeneration.get()) previewState.value = null
                }
                return@scheduleAtFixedRate
            }
            if (frame.receivedElapsedNs == decodedReceiptNs) return@scheduleAtFixedRate
            decodedReceiptNs = frame.receivedElapsedNs
            val decodeStart = SystemClock.elapsedRealtimeNanos()
            val image = runCatching { BitmapFactory.decodeByteArray(frame.jpeg, 0, frame.jpeg.size)?.asImageBitmap() }.getOrNull()
            previewMetrics.decoded(SystemClock.elapsedRealtimeNanos() - decodeStart)
            if (image != null && foreground && generation == previewGeneration.get()) {
                latestDecoded.set(generation to AtomPreviewFrame(image, frame.receivedElapsedNs, frame.sequence))
                queuePreviewPost()
            }
        // Check at display cadence, but decode only a new receipt. A 33ms poll
        // can skip a source frame when JPEG decode and Wi-Fi delivery take a slot.
        }, 0, 16, TimeUnit.MILLISECONDS)
    }

    private fun queuePreviewPost() {
        if (!previewPostQueued.compareAndSet(false, true)) return
        handler.post {
            previewPostQueued.set(false)
            val latest = latestDecoded.getAndSet(null)
            if (latest != null && foreground && latest.first == previewGeneration.get() && !isDestroyed) {
                previewState.value = latest.second
            }
            if (latestDecoded.get() != null) queuePreviewPost()
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString("capture_note", note)
        outState.putString("export_session_id", exportSessionId)
        super.onSaveInstanceState(outState)
    }

    override fun onStop() {
        foreground = false
        controlGeneration.incrementAndGet()
        discovering = false
        controlBusy = false
        hudTask?.cancel(false)
        previewTask?.cancel(false)
        invalidatePreview()
        enqueueControl { capture.stopForeground() }
        refreshSessions()
        super.onStop()
    }

    override fun onDestroy() {
        foreground = false
        controlGeneration.incrementAndGet()
        enqueueControl { capture.shutdown() }
        controls.shutdown()
        hudWorker.shutdown()
        previewWorker.shutdown()
        io.shutdown()
        super.onDestroy()
    }

    private fun isCurrent(generation: Long) = foreground && !isDestroyed && generation == controlGeneration.get()

    private fun enqueueControl(action: () -> Unit) {
        pendingControls.incrementAndGet()
        controls.execute {
            try { action() } finally { pendingControls.decrementAndGet() }
        }
    }

    private fun runControl(message: String, onComplete: () -> Unit = {}, action: (Long) -> Unit) {
        val generation = controlGeneration.incrementAndGet()
        controlBusy = true
        enqueueControl {
            if (!isCurrent(generation)) return@enqueueControl
            val result = runCatching { action(generation) }
            val next = runCatching { capture.snapshot() }.getOrNull()
            handler.post {
                if (isCurrent(generation)) {
                    controlBusy = false
                    next?.let { state = it }
                    result.onFailure { exportMessage = "$message：${it.message}" }
                    onComplete()
                }
            }
        }
    }

    private fun discoverDevices() {
        val generation = controlGeneration.incrementAndGet()
        discovering = true
        invalidatePreview()
        exportMessage = "正在寻找同一网络中的设备…"
        enqueueControl {
            capture.disconnect()
            if (!isCurrent(generation)) return@enqueueControl
            io.execute {
                val result = runCatching { HardwareWifiDiscovery.discover() }
                handler.post {
                    if (!isCurrent(generation)) return@post
                    discovering = false
                    result.onSuccess { found ->
                        val tofs = found.filter { it.role == "tof" && it.firmware == "xiao-cnh-h3-wifi-v1" }
                        val atoms = found.filter { it.role == "camera" && it.firmware == "atom-camera-wifi-v1" }
                        when {
                            tofs.isEmpty() || atoms.isEmpty() -> exportMessage =
                                "没有找到 ${if (tofs.isEmpty()) "ToF" else "Atom"}，请确认设备与手机连接同一网络，或手动填写地址。"
                            tofs.size == 1 && atoms.size == 1 -> {
                                host = URI(tofs.single().endpoint).host
                                atomHost = URI(atoms.single().endpoint).host
                                exportMessage = "已找到 Atom 和 ToF"
                                connectDevices()
                            }
                            else -> exportMessage = "找到多个设备，请手动填写 Atom 和 ToF 地址后连接。"
                        }
                    }.onFailure { exportMessage = "寻找失败，请手动填写地址：${it.message}" }
                }
            }
        }
    }

    private fun connectDevices() {
        val tofAddress = host.trim(); val atomAddress = atomHost.trim(); val imuEnabled = state.imuEnabled
        invalidatePreview()
        runControl("连接失败") { generation -> connectSources(generation, tofAddress, atomAddress, imuEnabled) }
    }

    private fun connectSources(generation: Long, tofAddress: String, atomAddress: String, imuEnabled: Boolean) {
        try {
            capture.disconnect()
            if (!isCurrent(generation)) return
            getSharedPreferences("phone_capture", MODE_PRIVATE).edit().putString("host", tofAddress)
                .putString("atom_host", atomAddress).putString("imu_host", atomAddress).apply()
            capture.connectCamera(AtomMjpegCaptureSource(atomAddress))
            if (!isCurrent(generation)) return
            capture.connect(TcpCnhCaptureSource(tofAddress))
            if (imuEnabled && isCurrent(generation)) capture.connectImu(TcpImuCaptureSource(atomAddress))
        } finally {
            if (!isCurrent(generation)) capture.disconnect()
        }
    }

    private fun invalidatePreview() {
        previewGeneration.incrementAndGet()
        latestDecoded.set(null)
        previewState.value = null
    }

    private fun refreshSessions() {
        if (io.isShutdown) return
        io.execute {
            val result = runCatching { PhoneCaptureStore.listSessions(this) }
            handler.post {
                if (!isDestroyed) result.onSuccess { sessions = it }
                    .onFailure { exportMessage = "记录目录不可用：${it.message}" }
            }
        }
    }

    fun resetPreviewPerformance() = previewMetrics.reset()
    fun previewPerformanceReceipt(): JSONObject = previewMetrics.receipt()
}

private class CapturePreviewMetrics {
    private var startedNs = SystemClock.elapsedRealtimeNanos()
    private var decodedCount = 0L
    private var drawnCount = 0L
    private var previousReceiptNs = 0L
    private val ages = ArrayDeque<Long>()
    private val decodeTimes = ArrayDeque<Long>()
    @Synchronized fun reset() {
        startedNs = SystemClock.elapsedRealtimeNanos(); decodedCount = 0; drawnCount = 0; previousReceiptNs = 0
        ages.clear(); decodeTimes.clear()
    }
    @Synchronized fun decoded(durationNs: Long) {
        decodedCount++
        if (decodeTimes.size == 1024) decodeTimes.removeFirst()
        decodeTimes.addLast(durationNs)
    }
    @Synchronized fun drawn(receiptNs: Long, nowNs: Long) {
        if (receiptNs == previousReceiptNs) return
        previousReceiptNs = receiptNs; drawnCount++
        if (ages.size == 1024) ages.removeFirst()
        ages.addLast((nowNs - receiptNs).coerceAtLeast(0))
    }
    @Synchronized fun receipt(): JSONObject {
        val elapsedNs = SystemClock.elapsedRealtimeNanos() - startedNs
        return JSONObject().put("schema", "blindassist-atom-preview-performance-v1")
            .put("elapsed_ns", elapsedNs).put("decoded_frames", decodedCount).put("drawn_unique_frames", drawnCount)
            .put("drawn_fps", drawnCount * 1e9 / elapsedNs.coerceAtLeast(1))
            .put("receipt_to_canvas_draw_ms", distribution(ages)).put("decode_ms", distribution(decodeTimes))
            .put("boundary", "Latest-only UI preview; Android receipt to Canvas submission, not exposure or physical display latency; recording stream unchanged")
    }
    private fun distribution(values: Collection<Long>): Any {
        if (values.isEmpty()) return JSONObject.NULL
        val sorted = values.sorted(); val middle = sorted.size / 2
        return JSONObject().put("min", sorted.first() / 1e6)
            .put("median", if (sorted.size % 2 == 0) (sorted[middle - 1] + sorted[middle]) / 2e6 else sorted[middle] / 1e6)
            .put("p95", sorted[ceil(sorted.size * .95).toInt() - 1] / 1e6).put("max", sorted.last() / 1e6)
            .put("count", sorted.size)
    }
}
