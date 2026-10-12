package com.linnan.blindassist

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import com.linnan.blindassist.capture.CaptureConnectionDiagnostics
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
internal fun CaptureExportTaskCard(task: CaptureExportTask, onCancel: () -> Unit,
    onRetry: () -> Unit, retryEnabled: Boolean) {
    OutlinedCard(shape = RoundedCornerShape(24.dp), modifier = Modifier.fillMaxWidth().testTag("capture_export_task")) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("导出任务", style = MaterialTheme.typography.titleMedium)
            Text(task.message, style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.testTag("capture_export_task_status"))
            if (task.isActive) {
                if (task.totalBytes > 0) {
                    LinearProgressIndicator(progress = { (task.bytesWritten.toFloat() / task.totalBytes).coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth().testTag("capture_export_progress"))
                    Text(String.format(Locale.getDefault(), "%.1f / %.1f MB", task.bytesWritten / 1048576.0,
                        task.totalBytes / 1048576.0), style = MaterialTheme.typography.bodySmall)
                } else LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                Text("可以离开此页面，返回后查看结果。允许通知后也可从通知栏查看进度。",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                OutlinedButton(onClick = onCancel, enabled = task.status != CaptureExportStatus.CANCELLING,
                    modifier = Modifier.heightIn(min = 48.dp).testTag("capture_export_cancel")) {
                    Text(if (task.status == CaptureExportStatus.CANCELLING) "正在取消" else "取消导出")
                }
            } else if (task.status != CaptureExportStatus.COMPLETED) {
                OutlinedButton(onClick = onRetry, enabled = retryEnabled,
                    modifier = Modifier.heightIn(min = 48.dp).testTag("capture_export_retry")) { Text("重新选择位置导出") }
            }
        }
    }
}

@Composable
internal fun CaptureConnectionCard(diagnostics: CaptureConnectionDiagnostics) {
    if (diagnostics.history.isEmpty() && diagnostics.persistenceError == null) return
    var expanded by rememberSaveable { mutableStateOf(false) }
    var showAll by rememberSaveable { mutableStateOf(false) }
    val timeFormat = remember { SimpleDateFormat("MM-dd HH:mm:ss", Locale.getDefault()) }
    OutlinedCard(shape = RoundedCornerShape(24.dp), modifier = Modifier.fillMaxWidth().testTag("capture_connection_history")) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("连接记录", style = MaterialTheme.typography.titleMedium)
            diagnostics.latest.forEach { event ->
                Text("${streamLabel(event.stream)} · ${event.message}", style = MaterialTheme.typography.bodySmall)
            }
            diagnostics.persistenceError?.let {
                Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }
            TextButton(onClick = { expanded = !expanded }, modifier = Modifier.heightIn(min = 48.dp)
                .testTag("capture_connection_expand")) { Text(if (expanded) "收起记录" else "查看最近记录") }
            if (expanded) diagnostics.history.takeLast(if (showAll) 64 else 12).asReversed().forEach { event ->
                HorizontalDivider()
                Text("${timeFormat.format(Date(event.wallClockMs))} · ${streamLabel(event.stream)}",
                    style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(event.message, style = MaterialTheme.typography.bodySmall)
                if (event.attempts > 0 || event.recoveryElapsedMs != null) Text(buildString {
                    append("连接尝试 ${event.attempts} 次")
                    event.recoveryElapsedMs?.let { append(String.format(Locale.getDefault(), " · 恢复用时 %.1f 秒", it / 1000.0)) }
                }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (expanded && !showAll && diagnostics.history.size > 12) TextButton(onClick = { showAll = true },
                modifier = Modifier.heightIn(min = 48.dp).testTag("capture_connection_show_all")) {
                Text("显示全部 ${diagnostics.history.size} 条记录")
            }
        }
    }
}

private fun streamLabel(stream: String): String = when (stream) {
    "camera" -> "RGB"
    "tof" -> "ToF"
    "imu" -> "IMU"
    else -> stream
}
