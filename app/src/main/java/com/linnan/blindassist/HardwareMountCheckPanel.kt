package com.linnan.blindassist

import android.os.SystemClock
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import com.linnan.blindassist.device.glasses.HardwareDemoSnapshot
import com.linnan.blindassist.risk.HardwareMountCheck
import kotlinx.coroutines.delay
import java.util.Locale

@Composable
fun HardwareMountCheckPanel(snapshot: HardwareDemoSnapshot?) {
    var captures by remember { mutableStateOf(listOf<HardwareMountCheck.Capture>()) }
    var collecting by remember { mutableStateOf(false) }
    var generation by remember { mutableStateOf(0) }
    var message by remember { mutableStateOf("仅本次会话有效；这是粗朝向检查，不是像素标定，也不会修改判决参数。") }
    var result by remember { mutableStateOf<HardwareMountCheck.Result?>(null) }
    var captureRun by remember { mutableStateOf<String?>(null) }
    val current by rememberUpdatedState(snapshot)
    val steps = listOf("移开目标，保持背景不动", "将目标放在画面中央", "将目标移到画面左侧", "将目标移到画面右侧")
    LaunchedEffect(snapshot?.runId, snapshot?.mode) {
        if (captureRun != null && (snapshot?.runId != captureRun || snapshot?.mode != "live")) {
            generation++; collecting = false; captures = emptyList(); result = null; captureRun = null
            message = "连接会话已改变；请重新采集背景与三个位置"
        }
    }
    LaunchedEffect(generation, collecting) {
        if (!collecting) return@LaunchedEffect
        val initial = current
        val run = initial?.runId
        var lastSeq = initial?.tofSequence
        val frames = mutableListOf<List<com.linnan.blindassist.risk.DemoTofCell>>()
        val started = SystemClock.elapsedRealtime()
        var interrupted = false
        while (SystemClock.elapsedRealtime() - started < 2000) {
            delay(40)
            val s = current
            if (s == null || s.runId != run || s.mode != "live" || !s.liveUsable || s.image == null) {
                interrupted = true
                break
            }
            val seq = s.tofSequence
            if (seq != null && (lastSeq == null || seq > lastSeq!!)) {
                lastSeq = seq
                if (s.rows == 8 && s.cols == 8) frames += s.cells.toList()
            }
            message = "采集中：${frames.size} 个不同的新帧 / 至少 5 个；请保持目标和设备静止"
        }
        if (interrupted || frames.size < 5) {
            message = "采集失败：需要连续实时画面与测距，以及 2 秒内至少 5 个新帧。请重试当前步骤。"
        } else {
            captureRun = run
            captures = captures + HardwareMountCheck.Capture(frames)
            message = "已采集 ${captures.size}/4 步，每步 ${frames.size} 个新帧"
            if (captures.size == 4) result = HardwareMountCheck.evaluate(captures[0], captures[1], captures[2], captures[3])
        }
        collecting = false
    }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.testTag("hardware_mount_panel")) {
        Text("安装粗检查 · 本次会话", color = Color.White)
        Text("固定设备、保持背景不动，使用较大的平面目标，目标应比背景近至少 20 厘米。每步保持 2 秒；目标上下位置尽量居中。", color = Color(0xFFBFCAD5))
        Text("只检查声明位置与原始测距变化的对应；不验证安全性、视场精度或像素对齐。", color = Color(0xFFBFCAD5))
        Text(message, color = Color.White, modifier = Modifier.testTag("hardware_mount_status"))
        if (captures.size < 4) Text("第 ${captures.size + 1}/4 步：${steps[captures.size]}", color = Color.White)
        Row {
            if (captures.size < 4) TextButton(onClick = { collecting = true },
                enabled = !collecting && snapshot?.mode == "live" && snapshot?.liveUsable == true && snapshot?.image != null,
                modifier = Modifier.testTag("hardware_mount_capture")) { Text("采集当前步骤") }
            TextButton(onClick = {
                generation++; collecting = false; captures = emptyList(); result = null; captureRun = null
                message = "已重置；移开目标后采集背景"
            }, modifier = Modifier.testTag("hardware_mount_reset")) { Text("重新检查") }
        }
        result?.let { r ->
            Text(r.message, color = if (r.consistent) Color(0xFF7CE8BF) else Color(0xFFFFC37D), modifier = Modifier.testTag("hardware_mount_result"))
            r.positions.forEachIndexed { i, p ->
                Text(String.format(Locale.ROOT, "%s：左右重心 %.2f，上下重心 %.2f；变化 %d / 配对 %d 区域", listOf("中央", "左侧", "右侧")[i], p.row, p.column, p.changedZones, p.pairedZones), color = Color(0xFFBFCAD5))
            }
            Text("重心范围 −1 至 +1：左右值越大越靠相机左侧，上下值越大越靠上。", color = Color(0xFFBFCAD5))
        }
    }
}
