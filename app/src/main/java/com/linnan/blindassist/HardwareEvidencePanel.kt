package com.linnan.blindassist

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.linnan.blindassist.device.glasses.HardwareEvidenceClip
import com.linnan.blindassist.device.glasses.HardwareEvidenceSummary
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
internal fun HardwareEvidencePanel(
    images: Boolean, busy: Boolean, status: String,
    recordings: List<HardwareEvidenceSummary>, selectedId: String?,
    clip: HardwareEvidenceClip?, positionMs: Long, playing: Boolean,
    onImages: (Boolean) -> Unit, onSave: () -> Unit,
    onReplay: (String) -> Unit, onExport: (String) -> Unit, onDelete: (String) -> Unit,
    onSeek: (Long) -> Unit, onPlay: () -> Unit, onLive: () -> Unit,
) {
    val muted = Color(0xFF9B9E9E)
    val ink = Color(0xFFB6C9BE)
    var deleteId by remember { mutableStateOf<String?>(null) }
    var expanded by remember { mutableStateOf(false) }
    val dateFormat = remember { SimpleDateFormat("MM-dd HH:mm:ss", Locale.getDefault()) }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(Modifier.fillMaxWidth()) {
            TextButton(onClick = onSave, enabled = !busy && clip == null,
                modifier = Modifier.weight(1f).testTag("hardware_evidence_save")) {
                Text(if (busy) "正在保存…" else "保存刚才 20 秒", color = ink)
            }
            TextButton(onClick = { expanded = !expanded },
                modifier = Modifier.testTag("hardware_evidence_toggle")) {
                Text(if (expanded) "收起记录 −" else "记录 / 回放 ＋", color = muted)
            }
        }
        Text(status, color = muted, fontSize = 11.sp, modifier = Modifier.testTag("hardware_evidence_status"))
        if (clip != null) {
            Text("历史记录 · 静音回放", color = ink, modifier = Modifier.testTag("hardware_evidence_replay"))
            Text("${"%.1f".format(positionMs / 1000.0)} / ${"%.1f".format(clip.durationMs / 1000.0)} 秒 · 原版本 ${clip.appVersion}",
                color = muted, fontSize = 11.sp)
            Slider(value = positionMs.toFloat(), onValueChange = { onSeek(it.toLong()) },
                valueRange = 0f..clip.durationMs.coerceAtLeast(1).toFloat(),
                modifier = Modifier.testTag("hardware_evidence_seek").semantics { contentDescription = "记录回放进度" })
            Row {
                TextButton(onClick = onPlay, modifier = Modifier.testTag("hardware_evidence_play")) {
                    Text(if (playing) "暂停" else "播放", color = ink)
                }
                TextButton(onClick = onLive, modifier = Modifier.testTag("hardware_evidence_live")) { Text("返回实时", color = ink) }
            }
            val frame = clip.frames[clip.frameIndex(positionMs)]
            Text("历史判决：${frame.decision.state}\n触发区域：${frame.decision.triggeringZones.joinToString().ifEmpty { "无" }}\n" +
                "相机时龄 ≤${frame.cameraAgeMs ?: "—"} ms · ToF 时龄 ≤${frame.tofAgeMs ?: "—"} ms",
                color = muted, fontSize = 11.sp)
            val nearby = clip.events.filter { it.atMs <= frame.atMs || positionMs >= clip.durationMs }.lastOrNull()
            Text(nearby?.let { "最近提示请求：${it.text}\n语音已入队：${it.speechQueued} · 震动已请求：${it.vibrationRequested}" }
                ?: "此时尚无提示请求记录", color = muted, fontSize = 11.sp)
        }
        if (expanded) {
            Row(Modifier.fillMaxWidth()) {
                Column(Modifier.weight(1f)) {
                    Text("附带画面抽帧", fontSize = 14.sp)
                    Text("默认关闭 · 最多 2 帧/秒 · 关闭清除未保存画面", color = muted, fontSize = 10.sp)
                }
                Switch(images, onImages, enabled = clip == null,
                    modifier = Modifier.testTag("hardware_evidence_images").semantics { contentDescription = "附带画面抽帧" })
            }
            Text("只在本机保留，最多 50 条 / 50 MB；导出时才写入你选择的位置。回放保留原判决，不重新计算。",
                color = muted, fontSize = 11.sp)
            if (recordings.isEmpty()) Text("尚无已保存记录", color = muted)
            recordings.forEachIndexed { index, item ->
                Column {
                    Text("${if (selectedId == item.id) "正在回放 · " else ""}${dateFormat.format(Date(item.createdAtMs))} · ${item.bytes / 1024} KB",
                        color = ink, fontSize = 12.sp)
                    Row {
                        TextButton(onClick = { onReplay(item.id) }, enabled = !busy,
                            modifier = Modifier.testTag("hardware_evidence_open_$index")) { Text("回放", color = ink) }
                        TextButton(onClick = { onExport(item.id) }, enabled = !busy) { Text("导出", color = muted) }
                        TextButton(onClick = { deleteId = item.id }, enabled = !busy) { Text("删除", color = muted) }
                    }
                }
            }
        }
    }
    deleteId?.let { id ->
        AlertDialog(onDismissRequest = { deleteId = null }, title = { Text("删除这条记录？") },
            text = { Text("删除后无法在 App 中恢复，已导出的副本不受影响。") },
            confirmButton = { TextButton(onClick = { deleteId = null; onDelete(id) }) { Text("删除") } },
            dismissButton = { TextButton(onClick = { deleteId = null }) { Text("保留") } })
    }
}
