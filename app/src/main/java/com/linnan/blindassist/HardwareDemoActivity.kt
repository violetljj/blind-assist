package com.linnan.blindassist

import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import com.linnan.blindassist.device.glasses.HardwareDemoClient
import com.linnan.blindassist.device.glasses.HardwareDemoSnapshot
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Explicit, foreground-only hardware demonstration; never runs the normal assist session. */
class HardwareDemoActivity : ComponentActivity() {
    private var snapshot by mutableStateOf<HardwareDemoSnapshot?>(null)
    private var connectionStatus by mutableStateOf("等待电脑中转连接")
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
        lastResultAt = SystemClock.elapsedRealtime()
        // Each start owns a new client. Cancelled old IO cannot publish into a new session.
        val client = HardwareDemoClient()
        polling = lifecycleScope.launch {
            while (isActive) {
                try {
                    val result = withContext(Dispatchers.IO) { client.poll() }
                    lastResultAt = SystemClock.elapsedRealtime()
                    snapshot = result
                    connectionStatus = result.status
                    if (!result.liveUsable || !result.decision.alert) tts?.stop()
                    maybeSpeak(result)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (_: Exception) {
                    invalidate("连接失效 · 请检查电脑中转与 USB")
                }
                delay(200)
            }
        }
        // Independent main-dispatcher clock: a slow/blocking network request cannot keep an alert alive.
        watchdog = lifecycleScope.launch {
            while (isActive) {
                delay(100)
                if (SystemClock.elapsedRealtime() - lastResultAt > 1_500) {
                    invalidate("连接失效 · 超过 1.5 秒未收到结果")
                }
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

    private fun invalidate(status: String) {
        snapshot = null
        connectionStatus = status
        tts?.stop()
    }

    override fun onStop() {
        foreground = false
        polling?.cancel()
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

@Composable
private fun HardwareDemoScreen(
    snapshot: HardwareDemoSnapshot?,
    connectionStatus: String,
    speechEnabled: Boolean,
    speechStatus: String,
    onSpeechChanged: (Boolean) -> Unit,
    onBack: () -> Unit
) {
    val replay = snapshot?.mode == "replay"
    val usable = snapshot != null && (snapshot.liveUsable || (replay && snapshot.image != null))
    val alert = usable && snapshot?.decision?.alert == true
    val accent = when {
        !usable -> Color(0xFF8B4513)
        replay -> Color(0xFF5B408F)
        alert -> Color(0xFFA52624)
        else -> Color(0xFF425466)
    }
    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text("硬件避障展示", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
                OutlinedButton(onClick = onBack) { Text("返回") }
            }
            Text("基础 ToF 走廊相交", fontWeight = FontWeight.Bold)
            Surface(color = accent, contentColor = Color.White, shape = RoundedCornerShape(12.dp)) {
                Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        when {
                            !usable && replay -> "回放 · 输入缺失或过期 · 静音"
                            !usable -> "连接失效 / 等待连接"
                            replay -> "回放 · 全程静音"
                            else -> "实物实时 · USB 电脑中转"
                        },
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.testTag("hardware_demo_mode")
                    )
                    Text(
                        when {
                            !usable -> "UNKNOWN · 当前无法判断"
                            alert && replay -> "回放结果：前方可能有障碍"
                            alert -> "前方可能有障碍"
                            else -> "UNKNOWN · 未触发障碍提示"
                        },
                        fontSize = 23.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.testTag("hardware_demo_decision")
                    )
                    Text(if (usable) "未提示不代表道路安全" else "旧画面和旧提示已清除")
                }
            }
            Text(connectionStatus, style = MaterialTheme.typography.bodySmall)
            val bitmap = snapshot?.image?.takeIf { usable }
            Box(
                Modifier.fillMaxWidth().aspectRatio(4f / 3f).background(Color(0xFF18232C), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                if (bitmap != null) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = if (replay) "回放相机画面" else "实物相机画面",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Fit
                    )
                } else {
                    Text("等待有效相机画面", color = Color.White)
                }
            }
            if (usable && snapshot != null) {
                Text("原始 ToF ${snapshot.rows} × ${snapshot.cols} 网格 · 单位 mm", fontWeight = FontWeight.Bold)
                TofGrid(snapshot)
                Text(
                    "有效 zone：${snapshot.decision.validZones}/${snapshot.rows * snapshot.cols} · " +
                        "支持 zone：${snapshot.decision.triggeringZones.joinToString().ifEmpty { "无" }}"
                )
                snapshot.decision.nearestMm?.takeIf { alert }?.let {
                    Text("支持测距最近值：$it mm（传感器测距）")
                }
                Text(
                    "相机帧 ${snapshot.cameraSequence ?: "—"} · ToF 帧 ${snapshot.tofSequence ?: "—"}",
                    style = MaterialTheme.typography.bodySmall
                )
            } else {
                Text("ToF：UNKNOWN · 等待有效测距")
            }
            Text("相机与 ToF 仅粗略同向，尚未完成标定；网格不与相机像素对齐。")
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("实时障碍语音", fontWeight = FontWeight.Bold)
                    Text(if (replay) "回放强制静音" else speechStatus, style = MaterialTheme.typography.bodySmall)
                }
                Switch(checked = speechEnabled, onCheckedChange = onSpeechChanged)
            }
            Text("研究演示 · 不可用于独立行走", style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun TofGrid(snapshot: HardwareDemoSnapshot) {
    val rows = snapshot.rows
    val cols = snapshot.cols
    if (rows !in listOf(4, 8) || cols !in listOf(4, 8)) {
        Text("UNKNOWN · 网格尺寸无效")
        return
    }
    val byZone = snapshot.cells.associateBy { it.zone }
    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        repeat(rows) { row ->
            Row(horizontalArrangement = Arrangement.spacedBy(3.dp)) {
                repeat(cols) { col ->
                    val zone = row * cols + col
                    val cell = byZone[zone]
                    val support = zone in snapshot.decision.triggeringZones
                    val hasRawRange = cell != null && cell.rangeMm.isFinite()
                    Box(
                        modifier = Modifier.weight(1f).aspectRatio(1f).background(
                            if (support) Color(0xFFA52624) else Color(0xFFF1EFEA),
                            RoundedCornerShape(4.dp)
                        ),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text("#$zone", fontSize = 9.sp, color = if (support) Color.White else Color(0xFF51616D))
                            Text(
                                if (hasRawRange) cell!!.rangeMm.toInt().toString() else "—",
                                fontSize = if (cols == 8) 10.sp else 15.sp,
                                color = if (support) Color.White else Color(0xFF18232C),
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }
        }
    }
    Text("红色：算法支持提示的 zone；其余为原始读数，可能无效。有效数量以算法判断为准。", style = MaterialTheme.typography.bodySmall)
}
