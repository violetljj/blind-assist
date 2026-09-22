package com.linnan.blindassist

import android.os.Bundle
import android.content.Intent
import android.provider.Settings
import android.os.SystemClock
import android.os.Handler
import android.os.Looper
import android.view.Choreographer
import android.speech.tts.TextToSpeech
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import com.linnan.blindassist.device.glasses.HardwareDemoClient
import com.linnan.blindassist.device.glasses.HardwareDemoSnapshot
import com.linnan.blindassist.device.glasses.HardwareWifiDemoClient
import com.linnan.blindassist.device.glasses.HardwareWifiDiscovery
import com.linnan.blindassist.device.glasses.HardwareWifiEndpoint
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import java.util.Locale
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
    private var speechStatus by mutableStateOf("语音初始化中")
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var foreground = false
    private var polling: Job? = null
    private var watchdog: Job? = null
    private var lastResultAt = 0L
    private var lastSpokenAt = -3_000L
    private var lastSpokenFrame: Pair<String, Long?>? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val preferences = getSharedPreferences("hardware_demo", MODE_PRIVATE)
        cameraEndpoint = preferences.getString("camera", "") ?: ""
        tofEndpoint = preferences.getString("tof", "") ?: ""
        wireless = preferences.getBoolean("wireless", true)
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.SIMPLIFIED_CHINESE)
                ttsReady = result != null && result >= TextToSpeech.LANG_AVAILABLE
            }
            speechStatus = if (ttsReady) "中文语音可用" else "中文语音不可用，请查看屏幕"
        }
        setContent {
            BlindAssistTheme {
                HardwareDemoScreen(
                    snapshot = snapshot,
                    connectionStatus = connectionStatus,
                    speechEnabled = speechEnabled,
                    speechStatus = speechStatus,
                    wireless = wireless,
                    cameraEndpoint = cameraEndpoint,
                    tofEndpoint = tofEndpoint,
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
                        if (!it) tts?.stop()
                    },
                    onBack = ::finish
                )
            }
        }
    }

    override fun onStart() {
        super.onStart()
        foreground = true
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        beginConnection(discover = wireless)
        // Independent clock: a blocked request must never keep an alert alive.
        watchdog = lifecycleScope.launch {
            while (isActive) {
                delay(100)
                if (snapshot != null && SystemClock.elapsedRealtime() - lastResultAt > 1_500) {
                    invalidate("连接失效 · 超过 1.5 秒未收到结果")
                }
            }
        }
    }

    private fun beginConnection(discover: Boolean) {
        if (!foreground) return
        val generation = ++connectionGeneration
        polling?.cancel()
        wifiClient?.close()
        wifiClient = null
        invalidate(if (wireless) "正在连接无线硬件…" else "等待 USB 电脑中转")
        lastResultAt = SystemClock.elapsedRealtime()
        polling = lifecycleScope.launch {
            val useWifi = wireless
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
                    val result = ownedWifi?.poll() ?: withContext(Dispatchers.IO) { requireNotNull(usbClient).poll() }
                    if (generation != connectionGeneration) break
                    lastResultAt = SystemClock.elapsedRealtime()
                    snapshot = result
                    connectionStatus = result.status
                    if (!result.liveUsable || !result.decision.alert) tts?.stop()
                    maybeSpeak(result)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (_: Exception) {
                    invalidate(if (useWifi) "无线连接失效 · 请检查热点和硬件供电" else "连接失效 · 请检查电脑中转与 USB")
                }
                if (useWifi) awaitDisplayFrame() else delay(200)
            }
            } finally {
                ownedWifi?.close()
            }
        }
    }

    private fun maybeSpeak(result: HardwareDemoSnapshot) {
        val now = SystemClock.elapsedRealtime()
        val key = result.runId to result.tofSequence
        if (foreground && speechEnabled && ttsReady && result.mode == "live" &&
            result.liveUsable && result.decision.alert && result.tofSequence != null &&
            key != lastSpokenFrame && now - lastSpokenAt >= 3_000
        ) {
            if (tts?.speak("前方可能有障碍", TextToSpeech.QUEUE_FLUSH, null, "hardware-demo") == TextToSpeech.SUCCESS) {
                lastSpokenAt = now
                lastSpokenFrame = key
            }
        }
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

    private fun invalidate(status: String) {
        snapshot = null
        connectionStatus = status
        tts?.stop()
    }

    override fun onStop() {
        foreground = false
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
        ttsReady = false
        tts?.stop()
        tts?.shutdown()
        tts = null
        super.onDestroy()
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
    speechEnabled: Boolean, speechStatus: String, wireless: Boolean,
    cameraEndpoint: String, tofEndpoint: String,
    onCameraChanged: (String) -> Unit, onTofChanged: (String) -> Unit,
    onConnect: () -> Unit, onDiscover: () -> Unit, onModeChanged: (Boolean) -> Unit,
    onHotspotSettings: () -> Unit, onSpeechChanged: (Boolean) -> Unit, onBack: () -> Unit
) {
    val replay = snapshot?.mode == "replay"
    val usable = snapshot != null && (snapshot.liveUsable || (replay && snapshot.image != null))
    val alert = usable && snapshot?.decision?.alert == true
    val accent = if (alert) DemoAmber else DemoTeal
    var detailsExpanded by remember { mutableStateOf(false) }
    var connectionExpanded by remember { mutableStateOf(false) }
    Surface(Modifier.fillMaxSize(), color = DemoBackground, contentColor = DemoInk) {
        Column(Modifier.safeDrawingPadding().verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp, vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(22.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("BLINDASSIST", color = DemoMuted, fontSize = 10.sp, letterSpacing = 2.sp)
                    Text("前视感知", fontSize = 27.sp, lineHeight = 38.sp, fontWeight = FontWeight.Medium)
                }
                TextButton(onClick = onBack) { Text("退出", color = DemoMuted, fontSize = 12.sp) }
            }
            Box(Modifier.fillMaxWidth().aspectRatio(0.94f).clip(RoundedCornerShape(5.dp))
                .background(DemoPanel)) {
                val bitmap = snapshot?.image?.takeIf { usable }
                if (bitmap != null) {
                    // Fit preserves the camera's full field of view; no decorative crop or fake detections.
                    Image(bitmap.asImageBitmap(), "实物相机视野", Modifier.fillMaxWidth()
                        .aspectRatio(4f / 3f).align(Alignment.Center), contentScale = ContentScale.Fit)
                } else {
                    Column(Modifier.align(Alignment.Center).testTag("hardware_demo_empty_image"),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("等待实时画面", fontSize = 18.sp, color = DemoInk)
                        Text("请连接相机与 ToF", fontSize = 12.sp, color = DemoMuted)
                    }
                }
                Box(Modifier.fillMaxSize().background(Brush.verticalGradient(listOf(
                    Color.Black.copy(alpha = 0.45f), Color.Transparent, Color.Black.copy(alpha = 0.8f)))))
                Row(Modifier.align(Alignment.TopStart).fillMaxWidth().padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(5.dp).background(if (usable) accent else DemoMuted, RoundedCornerShape(3.dp)))
                    Text(when { replay -> "  REPLAY / 静音"; usable -> "  LIVE VIEW"; else -> "  OFFLINE" },
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
                    Text(if (!usable) connectionStatus else "未提示不代表可通行 · 请以实际环境为准",
                        color = DemoMuted, fontSize = 11.sp, lineHeight = 16.sp)
                }
            }
            Box(Modifier.fillMaxWidth().height(1.dp).background(Color(0xFF303233)))
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("语音提示", fontSize = 14.sp)
                    Text(if (replay) "回放静音" else if (!speechEnabled) "已关闭" else speechStatus,
                        color = DemoMuted, fontSize = 10.sp)
                }
                Switch(speechEnabled && !replay, onSpeechChanged, enabled = !replay,
                    colors = SwitchDefaults.colors(checkedThumbColor = DemoBackground,
                        checkedTrackColor = DemoTeal, uncheckedTrackColor = DemoPanel))
            }
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
