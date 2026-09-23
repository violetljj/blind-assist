package com.linnan.blindassist

import android.os.Bundle
import android.content.Intent
import android.provider.Settings
import android.os.SystemClock
import android.os.Handler
import android.os.Looper
import android.os.Vibrator
import android.os.VibrationEffect
import android.view.Choreographer
import android.speech.tts.TextToSpeech
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.TextButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import com.linnan.blindassist.device.glasses.HardwareDemoClient
import com.linnan.blindassist.device.glasses.HardwareDemoSnapshot
import com.linnan.blindassist.device.glasses.HardwareLocalRuntime
import com.linnan.blindassist.risk.HardwareLocalModel
import com.linnan.blindassist.device.glasses.HardwareEvidenceRecorder
import com.linnan.blindassist.device.glasses.HardwareEvidenceStore
import com.linnan.blindassist.device.glasses.HardwareEvidenceSummary
import com.linnan.blindassist.device.glasses.HardwareEvidenceClip
import com.linnan.blindassist.device.glasses.HardwareEvidenceEvent
import com.linnan.blindassist.device.glasses.HardwareWifiDemoClient
import com.linnan.blindassist.feedback.HardwareDemoFeedbackPolicy
import com.linnan.blindassist.feedback.HardwareDemoFeedbackEvent
import com.linnan.blindassist.device.glasses.HardwareWifiDiscovery
import com.linnan.blindassist.device.glasses.HardwareWifiEndpoint
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import java.util.Locale
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/** Explicit, foreground-only hardware demonstration; never runs the normal assist session. */
class HardwareDemoActivity : ComponentActivity() {
    private var snapshot by mutableStateOf<HardwareDemoSnapshot?>(null)
    private var connectionStatus by mutableStateOf("等待无线硬件连接")
    private var wireless by mutableStateOf(true)
    private var cameraEndpoint by mutableStateOf("")
    private var tofEndpoint by mutableStateOf("")
    private var wifiClient: HardwareWifiDemoClient? = null
    private var connectionGeneration = 0
    private var speechEnabled by mutableStateOf(true)
    private var localEnabled by mutableStateOf(true)
    private var speechStatus by mutableStateOf("语音初始化中")
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var foreground = false
    private var polling: Job? = null
    private var watchdog: Job? = null
    private var lastResultAt = 0L
    private var vibrationEnabled by mutableStateOf(true)
    private val vibrator by lazy { getSystemService(Vibrator::class.java) }
    private val feedbackPolicy = HardwareDemoFeedbackPolicy()
    private var obstacleSpeech = false
    private var previousCameraUsable: Boolean? = null
    private val recorder by lazy { HardwareEvidenceRecorder(BuildConfig.VERSION_NAME) }
    private val evidenceStore by lazy { HardwareEvidenceStore(File(noBackupFilesDir, "hardware-evidence")) }
    private var recordImages by mutableStateOf(false)
    private var evidenceBusy by mutableStateOf(false)
    private var evidenceStatus by mutableStateOf("最近 20 秒循环暂存 · 默认不记录画面")
    private var recordings by mutableStateOf<List<HardwareEvidenceSummary>>(emptyList())
    private var replayClip by mutableStateOf<HardwareEvidenceClip?>(null)
    private var replayId by mutableStateOf<String?>(null)
    private var replayPosition by mutableStateOf(0L)
    private var replayPlaying by mutableStateOf(false)
    private var replayJob: Job? = null
    private var replayGeneration = 0
    private var pendingExport: String? = null
    private var replayLoading = false
    private var replayLoadGeneration = 0
    private var pendingReplayId: String? = null
    private val exportDocument = registerForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        val id = pendingExport
        pendingExport = null
        if (uri != null && id != null) lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val bytes = evidenceStore.readBytes(id)
                    requireNotNull(contentResolver.openOutputStream(uri, "wt")).use { it.write(bytes) }
                }
                evidenceStatus = "记录已导出"
            } catch (_: Exception) { evidenceStatus = "导出失败，请重新选择保存位置" }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val preferences = getSharedPreferences("hardware_demo", MODE_PRIVATE)
        cameraEndpoint = preferences.getString("camera", "") ?: ""
        tofEndpoint = preferences.getString("tof", "") ?: ""
        wireless = preferences.getBoolean("wireless", true)
        speechEnabled = preferences.getBoolean("speech", true)
        localEnabled = preferences.getBoolean("a_local", true)
        vibrationEnabled = preferences.getBoolean("vibration", true)
        pendingExport = savedInstanceState?.getString("hardware_export_id")
        refreshRecordings()
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.SIMPLIFIED_CHINESE)
                ttsReady = result != null && result >= TextToSpeech.LANG_AVAILABLE
                if (ttsReady) feedbackPolicy.reset()
            }
            speechStatus = if (ttsReady) "中文语音可用" else "中文语音不可用，请查看屏幕"
        }
        setContent {
            BlindAssistTheme {
                HardwareDemoScreen(
                    snapshot = snapshot,
                    connectionStatus = connectionStatus,
                    speechEnabled = speechEnabled,
                    localEnabled = localEnabled,
                    onLocalChanged = {
                        localEnabled = it
                        preferences.edit().putBoolean("a_local", it).apply()
                        feedbackPolicy.reset()
                        beginConnection(discover = false)
                    },
                    speechStatus = speechStatus,
                    vibrationEnabled = vibrationEnabled,
                    wireless = wireless,
                    cameraEndpoint = cameraEndpoint,
                    tofEndpoint = tofEndpoint,
                    evidenceContent = {
                        HardwareEvidencePanel(recordImages, evidenceBusy, evidenceStatus, recordings,
                            replayId, replayClip, replayPosition, replayPlaying,
                            onImages = {
                                recordImages = it; recorder.captureImages = it
                                evidenceStatus = if (it) "正在暂存测距和 2 Hz 画面抽帧，仅保存后保留" else "只暂存测距；未保存画面已清除"
                            },
                            onSave = ::saveEvidence, onReplay = { openReplay(it) },
                            onExport = { id -> pendingExport = id; exportDocument.launch(id) },
                            onDelete = ::deleteEvidence,
                            onSeek = { showReplay(it, false) },
                            onPlay = { showReplay(if (replayPosition >= (replayClip?.durationMs ?: 0)) 0 else replayPosition, !replayPlaying) },
                            onLive = ::returnToLive)
                    },
                    onCameraChanged = { cameraEndpoint = it },
                    onTofChanged = { tofEndpoint = it },
                    onConnect = { beginConnection(discover = false) },
                    onDiscover = { beginConnection(discover = true) },
                    onModeChanged = {
                        wireless = it
                        getSharedPreferences("hardware_demo", MODE_PRIVATE).edit()
                            .putBoolean("wireless", it).apply()
                        beginConnection(discover = it)
                    },
                    onHotspotSettings = {
                        runCatching { startActivity(Intent(Settings.ACTION_WIRELESS_SETTINGS)) }
                    },
                    onSpeechChanged = {
                        speechEnabled = it
                        preferences.edit().putBoolean("speech", it).apply()
                        feedbackPolicy.reset()
                        if (!it) tts?.stop()
                    },
                    onVibrationChanged = {
                        vibrationEnabled = it
                        preferences.edit().putBoolean("vibration", it).apply()
                        feedbackPolicy.reset()
                        if (!it) vibrator?.cancel()
                    },
                    onMore = {
                        startActivity(Intent(this, MainActivity::class.java).apply {
                            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                        })
                    }
                )
            }
        }
        savedInstanceState?.getString("hardware_replay_id")?.let {
            openReplay(it, savedInstanceState.getLong("hardware_replay_position", 0))
        }
    }

    override fun onStart() {
        super.onStart()
        foreground = true
        feedbackPolicy.reset()
        previousCameraUsable = null
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (!replayLoading) {
            if (replayClip == null) beginConnection(discover = wireless) else showReplay(replayPosition, false)
        }
        // Independent clock: a blocked request must never keep an alert alive.
        watchdog = lifecycleScope.launch {
            while (isActive) {
                delay(100)
                if (replayClip == null && snapshot != null && SystemClock.elapsedRealtime() - lastResultAt > 1_500) {
                    invalidate("连接失效 · 超过 1.5 秒未收到结果", notifyLoss = true)
                }
            }
        }
    }

    private fun beginConnection(discover: Boolean) {
        if (!foreground || replayClip != null || replayLoading) return
        val generation = ++connectionGeneration
        polling?.cancel()
        wifiClient?.close()
        wifiClient = null
        invalidate(if (wireless) "正在连接无线硬件…" else "等待 USB 电脑中转")
        lastResultAt = SystemClock.elapsedRealtime()
        polling = lifecycleScope.launch {
            val useWifi = wireless
            val useLocal = localEnabled
            var localProblem: String? = null
            val localRuntime = if (useLocal) try {
                withContext(Dispatchers.Default) {
                    assets.open("hardware_local/local-v1.bin").use { HardwareLocalRuntime(HardwareLocalModel.read(it)) }
                }
            } catch (cancelled: CancellationException) { throw cancelled }
            catch (_: Exception) { localProblem = "模型加载失败，已回退基础 ToF"; null }
            else null
            var ownedWifi: HardwareWifiDemoClient? = null
            try {
                if (useWifi) {
                    if (discover) {
                        connectionStatus = "正在发现热点内的相机和 ToF…"
                        val devices = withContext(Dispatchers.IO) { HardwareWifiDiscovery.discover() }
                        if (generation != connectionGeneration) return@launch
                        devices.firstOrNull { it.role == "camera" }?.let { cameraEndpoint = it.endpoint }
                        devices.firstOrNull { it.role == "tof" }?.let { tofEndpoint = it.endpoint }
                    }
                    val camera = HardwareWifiEndpoint.normalize(cameraEndpoint)
                    val tof = HardwareWifiEndpoint.normalize(tofEndpoint)
                    ownedWifi = HardwareWifiDemoClient(camera, tof)
                    wifiClient = ownedWifi
                    ownedWifi.start()
                    getSharedPreferences("hardware_demo", MODE_PRIVATE).edit()
                        .putString("camera", camera).putString("tof", tof).apply()
                }
            } catch (cancelled: CancellationException) {
                ownedWifi?.close()
                throw cancelled
            } catch (_: Exception) {
                ownedWifi?.close()
                invalidate("未连接 · 请开启手机热点并给两块硬件供电，再点自动发现")
                return@launch
            }
            val usbClient = if (useWifi) null else HardwareDemoClient()
            try {
            while (isActive) {
                try {
                    val raw = ownedWifi?.poll() ?: withContext(Dispatchers.IO) { requireNotNull(usbClient).poll() }
                    val processingStart = SystemClock.elapsedRealtime()
                    var result = if (localRuntime != null) {
                        try { withContext(Dispatchers.Default) { localRuntime.evaluate(raw, processingStart) } }
                        catch (cancelled: CancellationException) { throw cancelled }
                        catch (_: Exception) { raw.copy(status = "A+LOCAL 输入不兼容，已回退基础 ToF · ${raw.status}") }
                    } else if (useLocal) raw.copy(status = "A+LOCAL ${localProblem.orEmpty()} · ${raw.status}") else raw
                    val processingMs = SystemClock.elapsedRealtime() - processingStart
                    result = HardwareLocalRuntime.afterProcessing(raw, result, processingMs)
                    if (generation != connectionGeneration) break
                    lastResultAt = SystemClock.elapsedRealtime()
                    snapshot = result
                    connectionStatus = result.status
                    recorder.offer(result, lastResultAt)
                    updateFeedback(result)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (_: Exception) {
                    invalidate(if (useWifi) "无线连接失效 · 请检查热点和硬件供电" else "连接失效 · 请检查电脑中转与 USB", notifyLoss = true)
                }
                if (useWifi) awaitDisplayFrame() else delay(200)
            }
            } finally {
                ownedWifi?.close()
            }
        }
    }

    private fun updateFeedback(result: HardwareDemoSnapshot) {
        if (!foreground) return
        if (result.mode != "live" || (!result.tofUsable || !result.decision.alert) && obstacleSpeech) {
            tts?.stop()
            vibrator?.cancel()
            obstacleSpeech = false
        }
        val event = feedbackPolicy.update(SystemClock.elapsedRealtime(), result.mode == "live",
            result.tofUsable, result.decision.alert, result.decision.nearestMm)
        val cameraEvent = if (result.mode == "live" && result.tofUsable) when {
            !result.cameraUsable && previousCameraUsable != false -> HardwareDemoFeedbackEvent.CAMERA_ONLY
            result.cameraUsable && previousCameraUsable == false -> HardwareDemoFeedbackEvent.CAMERA_RECOVERED
            else -> null
        } else null
        if (result.mode == "live" && result.tofUsable) previousCameraUsable = result.cameraUsable
        if (event != null) emitFeedback(event, cameraEvent?.spokenText?.plus("，") ?: "")
        else cameraEvent?.let { emitFeedback(it) }
    }

    private fun emitFeedback(event: HardwareDemoFeedbackEvent, prefix: String = "") {
        if (!foreground) return
        obstacleSpeech = event !in setOf(HardwareDemoFeedbackEvent.CONNECTION_LOST,
            HardwareDemoFeedbackEvent.RECOVERED, HardwareDemoFeedbackEvent.CAMERA_ONLY,
            HardwareDemoFeedbackEvent.CAMERA_RECOVERED)
        val text = prefix + event.spokenText
        var speechQueued = false
        var vibrationRequested = false
        if (speechEnabled && ttsReady) {
            val result = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "hardware-${event.name}")
            speechQueued = result == TextToSpeech.SUCCESS
            if (result != TextToSpeech.SUCCESS) speechStatus = "语音播放失败，请查看屏幕或震动"
        }
        if (vibrationEnabled && vibrator?.hasVibrator() == true) {
            val pattern = when (event) {
                HardwareDemoFeedbackEvent.CONNECTION_LOST -> longArrayOf(0, 250, 120, 250)
                HardwareDemoFeedbackEvent.CLOSER -> longArrayOf(0, 120, 80, 120)
                HardwareDemoFeedbackEvent.RECOVERED -> longArrayOf(0, 80)
                else -> longArrayOf(0, 180)
            }
            vibrator?.vibrate(VibrationEffect.createWaveform(pattern, -1))
            vibrationRequested = true
        }
        recorder.recordEvent(HardwareEvidenceEvent(SystemClock.elapsedRealtime(), event.name, text, speechQueued, vibrationRequested))
    }

    private suspend fun awaitDisplayFrame(): Unit = suspendCancellableCoroutine { continuation ->
        // Activity lifecycleScope has no Compose MonotonicFrameClock. Use the main looper's
        // real display callback, cancelling it when the page stops or reconnects.
        val choreographer = Choreographer.getInstance()
        val callback = Choreographer.FrameCallback {
            if (continuation.isActive) continuation.resume(Unit)
        }
        choreographer.postFrameCallback(callback)
        continuation.invokeOnCancellation {
            Handler(Looper.getMainLooper()).post { choreographer.removeFrameCallback(callback) }
        }
    }

    private fun invalidate(status: String, notifyLoss: Boolean = false) {
        if (foreground && replayClip == null && notifyLoss) snapshot?.let { previous ->
            recorder.offer(previous.copy(image = null, cells = emptyList(), liveUsable = false,
                cameraUsable = false, tofUsable = false,
                decision = com.linnan.blindassist.risk.TofCorridorDemo.evaluate(8, 8, emptyList(), false),
                status = status), SystemClock.elapsedRealtime())
        }
        snapshot = null
        connectionStatus = status
        if (obstacleSpeech || !notifyLoss) {
            tts?.stop()
            vibrator?.cancel()
            obstacleSpeech = false
        }
        if (notifyLoss && foreground) {
            feedbackPolicy.update(SystemClock.elapsedRealtime(), true, false, false, null)?.let { emitFeedback(it) }
        }
    }

    override fun onStop() {
        foreground = false
        feedbackPolicy.reset()
        previousCameraUsable = null
        replayJob?.cancel()
        replayGeneration++
        replayPlaying = false
        recorder.clear()
        vibrator?.cancel()
        connectionGeneration++
        polling?.cancel()
        wifiClient?.close()
        wifiClient = null
        watchdog?.cancel()
        polling = null
        watchdog = null
        invalidate("展示已暂停 · UNKNOWN")
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        super.onStop()
    }

    override fun onDestroy() {
        recorder.close()
        ttsReady = false
        tts?.stop()
        tts?.shutdown()
        tts = null
        super.onDestroy()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString("hardware_export_id", pendingExport)
        outState.putString("hardware_replay_id", pendingReplayId ?: replayId)
        outState.putLong("hardware_replay_position", replayPosition)
        super.onSaveInstanceState(outState)
    }

    private fun refreshRecordings() {
        lifecycleScope.launch { recordings = withContext(Dispatchers.IO) { evidenceStore.list() } }
    }

    private fun saveEvidence() {
        if (evidenceBusy || replayClip != null || snapshot?.mode == "replay") return
        val clip = try { recorder.freeze() } catch (error: IllegalStateException) {
            evidenceStatus = error.message ?: "暂无记录"; return
        }
        evidenceBusy = true
        lifecycleScope.launch {
            try {
                val saved = withContext(Dispatchers.IO) { evidenceStore.save(clip) }
                evidenceStatus = "已保存 ${"%.1f".format(clip.durationMs / 1000.0)} 秒 · ${saved.bytes / 1024} KB"
                refreshRecordings()
            } catch (error: Exception) { evidenceStatus = error.message ?: "保存失败" }
            finally { evidenceBusy = false }
        }
    }

    private fun stopLive() {
        connectionGeneration++
        polling?.cancel(); polling = null
        wifiClient?.close(); wifiClient = null
        feedbackPolicy.reset(); previousCameraUsable = null
        tts?.stop(); vibrator?.cancel(); obstacleSpeech = false
        snapshot = null
    }

    private fun openReplay(id: String, initialPosition: Long = 0) {
        if (evidenceBusy) return
        evidenceBusy = true
        replayLoading = true
        pendingReplayId = id
        val loadGeneration = ++replayLoadGeneration
        stopLive()
        replayJob?.cancel(); replayGeneration++
        lifecycleScope.launch {
            try {
                val clip = withContext(Dispatchers.IO) { evidenceStore.load(id) }
                if (loadGeneration != replayLoadGeneration) return@launch
                stopLive()
                replayLoading = false
                replayClip = clip; replayId = id
                evidenceStatus = "回放保存时的测距与判决 · 不发语音或震动"
                replayPosition = initialPosition.coerceIn(0, clip.durationMs)
                if (foreground) showReplay(replayPosition, false)
            } catch (error: Exception) {
                if (loadGeneration != replayLoadGeneration) return@launch
                evidenceStatus = error.message ?: "读取失败"
                replayLoading = false
                replayClip = null; replayId = null
                if (foreground) beginConnection(false)
            } finally {
                if (loadGeneration == replayLoadGeneration) pendingReplayId = null
                evidenceBusy = false
            }
        }
    }

    private fun showReplay(positionMs: Long, play: Boolean) {
        val clip = replayClip ?: return
        replayJob?.cancel()
        val generation = ++replayGeneration
        replayPlaying = play
        replayPosition = positionMs.coerceIn(0, clip.durationMs)
        val startPosition = replayPosition
        replayJob = lifecycleScope.launch {
            val started = SystemClock.elapsedRealtime()
            var lastIndex = -1
            do {
                val position = if (play) (startPosition + SystemClock.elapsedRealtime() - started).coerceAtMost(clip.durationMs)
                    else startPosition
                val index = clip.frameIndex(position)
                if (index != lastIndex) {
                    val frame = withContext(Dispatchers.IO) { clip.snapshotAt(position) }
                    if (!foreground || generation != replayGeneration) return@launch
                    snapshot = frame; connectionStatus = frame.status
                    lastIndex = index
                }
                replayPosition = position
                if (!play || position >= clip.durationMs) break
                delay(50)
            } while (isActive)
            replayPlaying = false
        }
    }

    private fun returnToLive() {
        replayLoadGeneration++
        replayLoading = false
        pendingReplayId = null
        replayJob?.cancel(); replayGeneration++
        replayPlaying = false; replayClip = null; replayId = null; replayPosition = 0
        recorder.clear()
        evidenceStatus = "最近 20 秒循环暂存 · ${if (recordImages) "含 2 Hz 画面抽帧" else "不记录画面"}"
        beginConnection(discover = wireless)
    }

    private fun deleteEvidence(id: String) {
        if (evidenceBusy) return
        evidenceBusy = true
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) { evidenceStore.delete(id) }
                if (replayId == id) returnToLive()
                evidenceStatus = "记录已删除"
                refreshRecordings()
            } catch (error: Exception) { evidenceStatus = error.message ?: "删除失败" }
            finally { evidenceBusy = false }
        }
    }
}

private val DemoBackground = Color(0xFF101112)
private val DemoPanel = Color(0xFF1A1C1D)
private val DemoInk = Color(0xFFF2F1ED)
private val DemoMuted = Color(0xFF9B9E9E)
private val DemoTeal = Color(0xFFB6C9BE)
private val DemoAmber = Color(0xFFE9BD87)

@Composable
private fun HardwareDemoScreen(
    snapshot: HardwareDemoSnapshot?, connectionStatus: String,
    localEnabled: Boolean, onLocalChanged: (Boolean) -> Unit,
    speechEnabled: Boolean, speechStatus: String, vibrationEnabled: Boolean, wireless: Boolean,
    cameraEndpoint: String, tofEndpoint: String,
    evidenceContent: @Composable () -> Unit,
    onCameraChanged: (String) -> Unit, onTofChanged: (String) -> Unit,
    onConnect: () -> Unit, onDiscover: () -> Unit, onModeChanged: (Boolean) -> Unit,
    onHotspotSettings: () -> Unit, onSpeechChanged: (Boolean) -> Unit,
    onVibrationChanged: (Boolean) -> Unit, onMore: () -> Unit
) {
    val replay = snapshot?.mode == "replay"
    val usable = snapshot?.tofUsable == true
    val cameraUsable = snapshot?.cameraUsable == true
    val alert = usable && snapshot?.decision?.alert == true
    val accent = if (alert) DemoAmber else DemoTeal
    var detailsExpanded by remember { mutableStateOf(false) }
    var connectionExpanded by remember { mutableStateOf(false) }
    var mountExpanded by remember { mutableStateOf(false) }
    Surface(Modifier.fillMaxSize(), color = DemoBackground, contentColor = DemoInk) {
        Column(Modifier.safeDrawingPadding().verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp, vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(22.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("BLINDASSIST", color = DemoMuted, fontSize = 10.sp, letterSpacing = 2.sp)
                    Text("前视感知", fontSize = 27.sp, lineHeight = 38.sp, fontWeight = FontWeight.Medium)
                }
                TextButton(onClick = onMore, modifier = Modifier.testTag("hardware_demo_more")) {
                    Text("更多功能", color = DemoMuted, fontSize = 12.sp)
                }
            }
            Box(Modifier.fillMaxWidth().aspectRatio(0.94f).clip(RoundedCornerShape(5.dp))
                .background(DemoPanel)) {
                val bitmap = snapshot?.image?.takeIf { cameraUsable }
                if (bitmap != null) {
                    // Fit preserves the camera's full field of view; no decorative crop or fake detections.
                    Image(bitmap.asImageBitmap(), "实物相机视野", Modifier.fillMaxWidth()
                        .aspectRatio(4f / 3f).align(Alignment.Center), contentScale = ContentScale.Fit)
                } else {
                    Column(Modifier.align(Alignment.Center).testTag("hardware_demo_empty_image"),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(if (replay) "此时未记录画面" else if (usable) "仅测距模式" else "等待实时画面", fontSize = 18.sp, color = DemoInk)
                        Text(if (replay) "历史测距可在下方查看" else if (usable) "相机不可用 · ToF 仍在工作" else "请连接相机与 ToF", fontSize = 12.sp, color = DemoMuted)
                    }
                }
                Box(Modifier.fillMaxSize().background(Brush.verticalGradient(listOf(
                    Color.Black.copy(alpha = 0.45f), Color.Transparent, Color.Black.copy(alpha = 0.8f)))))
                Row(Modifier.align(Alignment.TopStart).fillMaxWidth().padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(5.dp).background(if (usable) accent else DemoMuted, RoundedCornerShape(3.dp)))
                    Text(when { replay -> "  REPLAY / 静音"; usable && cameraUsable -> "  LIVE VIEW";
                        usable -> "  TOF ONLY / 仅测距"; cameraUsable -> "  CAMERA ONLY / 仅画面"; else -> "  OFFLINE" },
                        color = DemoInk, fontSize = 10.sp, letterSpacing = 1.sp,
                        modifier = Modifier.weight(1f).testTag("hardware_demo_mode"))
                    Text(if (wireless) "Wi-Fi" else "USB", color = DemoInk, fontSize = 10.sp)
                }
                Row(Modifier.align(Alignment.BottomStart).fillMaxWidth().padding(18.dp),
                    verticalAlignment = Alignment.Bottom) {
                    Column(Modifier.weight(1f)) {
                        Text("最近触发测距", fontSize = 10.sp, color = DemoMuted, letterSpacing = 1.sp)
                        Row(verticalAlignment = Alignment.Bottom) {
                            Text(if (usable) snapshot?.decision?.nearestMm?.let {
                                String.format(Locale.US, "%.2f", it / 1000.0) } ?: "—" else "—",
                                fontSize = 46.sp, lineHeight = 54.sp, fontWeight = FontWeight.Light)
                            Text(" m", fontSize = 18.sp, color = DemoMuted, modifier = Modifier.padding(bottom = 7.dp))
                        }
                    }
                    Column(horizontalAlignment = Alignment.End, modifier = Modifier.padding(bottom = 5.dp)) {
                        Text(if (usable) "${snapshot!!.decision.validZones} / 64" else "— / 64",
                            fontSize = 17.sp, fontWeight = FontWeight.Medium)
                        Text("有效测距区域", fontSize = 10.sp, color = DemoMuted)
                    }
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(3.dp, 44.dp).background(if (alert) DemoAmber else DemoMuted))
                Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                    Text(when {
                        !usable -> "等待有效感知"
                        alert && replay -> "回放 · 前方可能有障碍"
                        alert -> "前方可能有障碍"
                        else -> "当前未触发提示"
                    }, fontSize = 20.sp, lineHeight = 27.sp, fontWeight = FontWeight.Medium,
                        modifier = Modifier.testTag("hardware_demo_decision"))
                    Text(if (!usable || !cameraUsable) connectionStatus else "未提示不代表可通行 · 请以实际环境为准",
                        color = DemoMuted, fontSize = 11.sp, lineHeight = 16.sp)
                }
            }
            Box(Modifier.fillMaxWidth().height(1.dp).background(Color(0xFF303233)))
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("A+LOCAL 实验模式", fontSize = 14.sp)
                    Text(if (replay) "回放使用保存的判决" else if (localEnabled) "未完成实物标定 · 模拟效果不代表实机效果" else "当前使用基础 ToF",
                        color = DemoMuted, fontSize = 10.sp)
                }
                Switch(localEnabled, onLocalChanged, enabled = !replay,
                    modifier = Modifier.testTag("hardware_demo_local").semantics { contentDescription = "A+LOCAL 实验模式" },
                    colors = SwitchDefaults.colors(checkedThumbColor = DemoBackground,
                        checkedTrackColor = DemoTeal, uncheckedTrackColor = DemoPanel))
            }
            if (localEnabled || replay) Text(snapshot?.status ?: connectionStatus,
                color = DemoMuted, fontSize = 10.sp,
                modifier = Modifier.testTag("hardware_demo_algorithm_status"))
            if ((!usable || !cameraUsable) && !replay) {
                TextButton(onClick = onDiscover, modifier = Modifier.testTag("hardware_demo_reconnect")) {
                    Text(if (wireless) "重新发现并连接" else "重新连接 USB 中转", color = DemoTeal)
                }
            }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("语音提示", fontSize = 14.sp)
                    Text(if (replay) "回放静音" else if (!speechEnabled) "已关闭" else speechStatus,
                        color = DemoMuted, fontSize = 10.sp)
                }
                Switch(speechEnabled && !replay, onSpeechChanged, enabled = !replay,
                    modifier = Modifier.testTag("hardware_demo_speech").semantics { contentDescription = "语音提示" },
                    colors = SwitchDefaults.colors(checkedThumbColor = DemoBackground,
                        checkedTrackColor = DemoTeal, uncheckedTrackColor = DemoPanel))
            }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("震动提示", fontSize = 14.sp)
                    Text(if (replay) "回放不震动" else "障碍事件与连接中断提醒", color = DemoMuted, fontSize = 10.sp)
                }
                Switch(vibrationEnabled && !replay, onVibrationChanged, enabled = !replay,
                    modifier = Modifier.testTag("hardware_demo_vibration").semantics { contentDescription = "震动提示" },
                    colors = SwitchDefaults.colors(checkedThumbColor = DemoBackground,
                        checkedTrackColor = DemoTeal, uncheckedTrackColor = DemoPanel))
            }
            evidenceContent()
            TextButton(onClick = { mountExpanded = !mountExpanded }, modifier = Modifier.testTag("hardware_mount_toggle")) {
                Text(if (mountExpanded) "收起安装检查 −" else "安装朝向检查 ＋", color = DemoMuted)
            }
            if (mountExpanded) HardwareMountCheckPanel(snapshot)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                TextButton(onClick = { detailsExpanded = !detailsExpanded },
                    modifier = Modifier.testTag("hardware_demo_detail_toggle")) {
                    Text(if (detailsExpanded) "收起数据 −" else "感知数据 ＋", color = DemoMuted, fontSize = 12.sp)
                }
                TextButton(onClick = { connectionExpanded = !connectionExpanded }) {
                    Text(if (connectionExpanded) "收起连接 −" else "设备连接 ＋", color = DemoMuted, fontSize = 12.sp)
                }
            }
            if (connectionExpanded) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Wi-Fi 直连", modifier = Modifier.weight(1f))
                        Switch(wireless, onModeChanged)
                    }
                    if (wireless) {
                        Text("开启 2.4 GHz 手机热点，给两块硬件供电。", fontSize = 12.sp, color = DemoMuted)
                        Row {
                            TextButton(onClick = onDiscover) { Text("自动发现", color = DemoTeal) }
                            TextButton(onClick = onHotspotSettings) { Text("网络设置", color = DemoTeal) }
                        }
                        OutlinedTextField(cameraEndpoint, onCameraChanged, label = { Text("相机 IP") },
                            singleLine = true, modifier = Modifier.fillMaxWidth())
                        OutlinedTextField(tofEndpoint, onTofChanged, label = { Text("ToF IP") },
                            singleLine = true, modifier = Modifier.fillMaxWidth())
                        TextButton(onClick = onConnect) { Text("连接指定设备", color = DemoTeal) }
                    }
                }
            }
            if (detailsExpanded) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("ToF / 8 × 8", fontSize = 14.sp, letterSpacing = 1.sp)
                    if (usable && snapshot != null) TofGrid(snapshot)
                    else Text("UNKNOWN · 等待有效测距", color = DemoMuted,
                        modifier = Modifier.testTag("hardware_demo_empty_heatmap"))
                    Text("原始距离 · mm  /  橙色为触发区域；其他颜色不代表安全。", color = DemoMuted, fontSize = 10.sp)
                    Text(connectionStatus, color = DemoTeal, fontSize = 11.sp, lineHeight = 17.sp)
                    if (usable && snapshot != null) Text("相机帧 ${snapshot.cameraSequence} · ToF 帧 ${snapshot.tofSequence}",
                        color = DemoMuted, fontSize = 11.sp)
                    Text("ToF 完整区域与前方走廊相交判定。相机提供现场画面；两路独立采样，尚未完成空间标定。",
                        color = DemoMuted, fontSize = 11.sp, lineHeight = 17.sp)
                }
            }
            Text("研究演示  /  不可用于独立行走", color = DemoMuted.copy(alpha = 0.8f),
                fontSize = 9.sp, letterSpacing = 1.sp, modifier = Modifier.padding(bottom = 8.dp))
        }
    }
}

@Composable
private fun TofGrid(snapshot: HardwareDemoSnapshot) {
    val rows = snapshot.rows
    val cols = snapshot.cols
    if (rows !in listOf(4, 8) || cols !in listOf(4, 8)) {
        Text("UNKNOWN · 网格尺寸无效", color = DemoMuted)
        return
    }
    val byZone = snapshot.cells.associateBy { it.zone }
    Column(Modifier.testTag("hardware_demo_heatmap"), verticalArrangement = Arrangement.spacedBy(3.dp)) {
        repeat(rows) { row ->
            Row(horizontalArrangement = Arrangement.spacedBy(3.dp)) {
                repeat(cols) { col ->
                    val zone = row * cols + col
                    val cell = byZone[zone]
                    val support = zone in snapshot.decision.triggeringZones
                    val hasRawRange = cell != null && cell.rangeMm.isFinite()
                    // Color encodes raw distance only, never validity, confidence, or safe passage.
                    val intensity = if (hasRawRange)
                        (1.0 - cell!!.rangeMm / 4000.0).coerceIn(0.0, 1.0).toFloat() else 0f
                    val fill = when {
                        support -> DemoAmber
                        hasRawRange -> lerp(Color(0xFF112838), Color(0xFF256E6B), intensity)
                        else -> DemoBackground
                    }
                    Box(modifier = Modifier.weight(1f).height(if (rows == 8) 17.dp else 36.dp)
                        .background(fill, RoundedCornerShape(3.dp)), contentAlignment = Alignment.Center) {
                        // At 8 × 8 the compact overview retains every numeric reading. Zone IDs
                        // remain in the expandable trigger list instead of crowding each cell.
                        Text(if (hasRawRange) cell!!.rangeMm.toInt().toString() else "—",
                            fontSize = if (cols == 8) 9.sp else 12.sp, lineHeight = 14.sp,
                            color = if (support) DemoBackground else DemoTeal.copy(alpha = 0.65f + intensity * 0.35f),
                            fontWeight = FontWeight.Medium)
                    }
                }
            }
        }
    }
}
