package com.linnan.blindassist

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.linnan.blindassist.capture.CnhCaptureFrame
import com.linnan.blindassist.capture.PhoneCaptureState
import com.linnan.blindassist.capture.PhoneCaptureSummary
import java.util.Locale

@Composable
internal fun PhoneCaptureScreen(
    state: PhoneCaptureState, preview: () -> AtomPreviewFrame?, onPreviewDraw: (Long) -> Unit,
    host: String, atomHost: String, note: String,
    sessions: List<PhoneCaptureSummary>, busy: Boolean, exportMessage: String,
    onHostChange: (String) -> Unit, onAtomHostChange: (String) -> Unit,
    onImuEnabledChange: (Boolean) -> Unit, onNoteChange: (String) -> Unit,
    onConnect: () -> Unit, onDiscover: () -> Unit, onDisconnect: () -> Unit,
    onStart: () -> Unit, onStop: () -> Unit, onExport: (PhoneCaptureSummary) -> Unit, onBack: () -> Unit,
    exportTask: CaptureExportTask? = null, onCancelExport: () -> Unit = {}, onRetryExport: () -> Unit = {},
    onFusionExperiment: () -> Unit = {},
) {
    var showSettings by rememberSaveable { mutableStateOf(false) }
    var showSessions by rememberSaveable { mutableStateOf(false) }
    val colors = MaterialTheme.colorScheme
    Scaffold(containerColor = colors.background) { insets ->
        Column(Modifier.fillMaxSize().padding(insets).verticalScroll(rememberScrollState()).padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically) {
                Column {
                    Text("数据采集", style = MaterialTheme.typography.headlineSmall,
                        modifier = Modifier.semantics { heading() })
                    Text("画面 · 距离 · 姿态", style = MaterialTheme.typography.bodyMedium,
                        color = colors.onSurfaceVariant)
                }
                TextButton(onClick = onBack) { Text("返回首页") }
            }
            Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("实时预览 · " + if (state.recording) "正在记录 ${state.elapsedMs / 1000} 秒" else "准备采集",
                    style = MaterialTheme.typography.titleMedium, color = colors.primary,
                    modifier = Modifier.semantics { heading() })
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Text("RGB ${if (state.cameraReady) "已就绪" else "等待画面"}  ·  ToF ${if (state.tofReady) "已就绪" else "等待数据"}",
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.weight(1f).testTag("capture_readiness"))
                    if (state.imuEnabled) Text("IMU ${if (state.imuReady) "已就绪" else "等待数据"}",
                        style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant,
                        modifier = Modifier.testTag("capture_imu_readiness"))
                }
                CapturePreviews(state, preview, onPreviewDraw)
                if (state.recording) {
                    Button(onClick = onStop, enabled = !busy,
                        modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp).testTag("capture_stop")) { Text("停止并保存") }
                } else {
                    Button(onClick = onStart, enabled = state.cameraReady && state.tofReady && (!state.imuEnabled || state.imuReady) && !busy,
                        modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp).testTag("capture_start")) { Text("开始采集") }
                }
                Text("RGB ${state.rgbCount} · ToF ${state.tofCount}" + (if (state.imuEnabled) " · IMU ${state.imuCount}" else ""),
                    style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant,
                    modifier = Modifier.testTag("capture_counts"))
                Text(state.message, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant,
                    modifier = Modifier.testTag("capture_status"))
                if (state.imuEnabled) Text(state.imuMessage, style = MaterialTheme.typography.bodySmall,
                    color = colors.onSurfaceVariant,
                    modifier = Modifier.testTag("capture_imu_status"))
                if (!state.recording) Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = onDiscover, enabled = !busy,
                        modifier = Modifier.weight(1f).heightIn(min = 48.dp).testTag("capture_discover")) { Text("自动寻找设备") }
                    OutlinedButton(onClick = onConnect, enabled = !busy,
                        modifier = Modifier.heightIn(min = 48.dp).testTag("capture_connect")) { Text("连接 / 重连") }
                }
            }
            exportTask?.let { CaptureExportTaskCard(it, onCancelExport, onRetryExport, retryEnabled = !state.recording && !busy) }
            CaptureConnectionCard(state.connectionDiagnostics)
            if (exportMessage.isNotBlank()) Surface(color = colors.primaryContainer, shape = RoundedCornerShape(18.dp)) {
                Text(exportMessage, modifier = Modifier.fillMaxWidth().padding(14.dp),
                    style = MaterialTheme.typography.bodyMedium, color = colors.onPrimaryContainer)
            }
            if (!state.recording) {
                OutlinedButton(onClick = { showSettings = !showSettings },
                    modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).testTag("capture_settings_toggle")) {
                    Text(if (showSettings) "收起连接与备注" else "连接与备注 ▾")
                }
                if (showSettings) CapturePanel {
                    CaptureImuToggle(state.imuEnabled, !busy, onImuEnabledChange)
                    OutlinedTextField(value = atomHost, onValueChange = onAtomHostChange,
                        label = { Text("Atom 地址（RGB 与 IMU）") }, singleLine = true,
                        modifier = Modifier.fillMaxWidth().testTag("capture_atom_host"))
                    OutlinedTextField(value = host, onValueChange = onHostChange, label = { Text("ToF 地址") }, singleLine = true,
                        modifier = Modifier.fillMaxWidth().testTag("capture_host"))
                    OutlinedTextField(value = note, onValueChange = onNoteChange, label = { Text("场景备注（可选）") },
                        modifier = Modifier.fillMaxWidth().testTag("capture_note"))
                    if (state.connected || state.cameraConnected) TextButton(onClick = onDisconnect) { Text("断开设备") }
                }
            }
            Text("采集时保持页面打开。切到后台或断连会结束本段，并保留已有数据。",
                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
            if (!state.recording) TextButton(onClick = onFusionExperiment, enabled = !busy,
                modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).testTag("capture_fusion_experiment")) {
                Text("打开融合实验")
            }
            HorizontalDivider(color = colors.outlineVariant)
            TextButton(onClick = { showSessions = !showSessions },
                modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).testTag("capture_sessions_toggle")) {
                Text("已保存 ${sessions.size} 段 ${if (showSessions) "▴" else "▾"}")
            }
            if (showSessions) sessions.forEach { session ->
                key(session.id) {
                    CapturePanel {
                        Text(session.sceneNote.ifBlank { session.id }, style = MaterialTheme.typography.titleSmall)
                        Text("RGB ${session.rgbCount} / ToF ${session.tofCount} / IMU ${session.imuCount} 帧 · ${session.bytes / 1024 / 1024} MB",
                            style = MaterialTheme.typography.bodySmall)
                        Text(if (session.status == "COMPLETED") "已保存" else "部分记录已保留", style = MaterialTheme.typography.bodySmall)
                        OutlinedButton(onClick = { onExport(session) }, enabled = !state.recording && !busy && exportTask?.isActive != true,
                            modifier = Modifier.testTag("capture_export_${session.id}")) { Text("导出 ZIP") }
                    }
                }
            }
        }
    }
}

@Composable
internal fun CaptureImuToggle(checked: Boolean, enabled: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().heightIn(min = 48.dp)
        .testTag("capture_imu_enabled")
        .toggleable(value = checked, enabled = enabled, role = Role.Checkbox, onValueChange = onCheckedChange),
        verticalAlignment = Alignment.CenterVertically) {
        Checkbox(checked = checked, onCheckedChange = null, enabled = enabled)
        Text("同时记录 IMU", modifier = Modifier.padding(start = 12.dp))
    }
}

@Composable
internal fun CapturePreviews(state: PhoneCaptureState, preview: () -> AtomPreviewFrame?, onPreviewDraw: (Long) -> Unit) {
    val fontScale = LocalDensity.current.fontScale
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        // Fixed compact cards are only suitable when their text fits beside the grid.
        // Stacked cards measure text naturally; incoming camera frames still only invalidate Canvas.
        val stacked = fontScale > 1.3f || maxWidth < 340.dp
        if (stacked) {
            Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                CaptureRgbPreview(state, preview, onPreviewDraw, Modifier.fillMaxWidth(), stacked = true)
                CaptureTofPreview(state, Modifier.fillMaxWidth(), stacked = true)
                CaptureImuPreview(state, Modifier.fillMaxWidth(), stacked = true)
            }
        } else {
            Row(Modifier.fillMaxWidth().height(384.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CaptureRgbPreview(state, preview, onPreviewDraw, Modifier.weight(1f).fillMaxHeight(), stacked = false)
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    CaptureTofPreview(state, Modifier.fillMaxWidth().height(228.dp), stacked = false)
                    CaptureImuPreview(state, Modifier.fillMaxWidth().weight(1f), stacked = false)
                }
            }
        }
    }
}

@Composable
private fun PreviewCard(modifier: Modifier, content: @Composable ColumnScope.() -> Unit) {
    Card(modifier, shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant), content = content)
}

@Composable
private fun CaptureRgbPreview(state: PhoneCaptureState, preview: () -> AtomPreviewFrame?, onPreviewDraw: (Long) -> Unit,
    modifier: Modifier, stacked: Boolean) {
    PreviewCard(modifier.testTag("capture_preview")) {
        Column((if (stacked) Modifier.fillMaxWidth() else Modifier.fillMaxSize()).padding(10.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Atom RGB", style = MaterialTheme.typography.titleSmall)
            Text(freshness(state.cameraAgeMs, state.cameraReady), style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.testTag("capture_rgb_freshness"))
            Box(Modifier.fillMaxWidth().then(if (stacked) Modifier.aspectRatio(4f / 3f) else Modifier.weight(1f))
                .clip(RoundedCornerShape(14.dp)).background(Color(0xff111827)), contentAlignment = Alignment.Center) {
                AtomImagePreview(preview, onPreviewDraw, state.displayRotationDegrees, state.displayMirrorHorizontal)
            }
            Text(state.cameraMessage, style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.testTag("capture_camera_status"))
        }
    }
}

@Composable
private fun CaptureTofPreview(state: PhoneCaptureState, modifier: Modifier, stacked: Boolean) {
    PreviewCard(modifier.testTag("capture_tof_preview")) {
        Column((if (stacked) Modifier.fillMaxWidth() else Modifier.fillMaxSize()).padding(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(3.dp)) {
            if (stacked) {
                Text("ToF · m", style = MaterialTheme.typography.titleSmall, modifier = Modifier.testTag("capture_tof_title"))
                Text(freshness(state.tofAgeMs, state.tofReady), style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.testTag("capture_tof_freshness"))
            } else {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("ToF · m", style = MaterialTheme.typography.titleSmall, modifier = Modifier.testTag("capture_tof_title"))
                    Text(freshness(state.tofAgeMs, state.tofReady), fontSize = 10.sp,
                        modifier = Modifier.testTag("capture_tof_freshness"))
                }
            }
            TofDistanceGrid(state.tofFrame,
                if (stacked) Modifier.widthIn(max = 320.dp).fillMaxWidth() else Modifier.size(160.dp))
            Text("无：无读数", style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.testTag("capture_tof_no_reading_legend"))
        }
    }
}

@Composable
private fun CaptureImuPreview(state: PhoneCaptureState, modifier: Modifier, stacked: Boolean) {
    PreviewCard(modifier.testTag("capture_imu_preview")) {
        Column(Modifier.padding(8.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            if (stacked) {
                Text("IMU", style = MaterialTheme.typography.titleSmall)
                Text(if (state.imuEnabled) freshness(state.imuAgeMs, state.imuReady) else "已关闭",
                    style = MaterialTheme.typography.labelSmall, modifier = Modifier.testTag("capture_imu_freshness"))
            } else {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("IMU", style = MaterialTheme.typography.titleSmall)
                    Text(if (state.imuEnabled) freshness(state.imuAgeMs, state.imuReady) else "已关闭",
                        fontSize = 10.sp, modifier = Modifier.testTag("capture_imu_freshness"))
                }
            }
            val imu = state.imuFrame
            Text("俯仰 ${angle(imu?.pitchDeg)}  横滚 ${angle(imu?.rollDeg)}", fontSize = 11.sp, lineHeight = 14.sp,
                modifier = Modifier.testTag("capture_imu_angles"))
            Text("a(g) ${vector(imu?.accelG)}", fontSize = 10.sp, lineHeight = 13.sp,
                modifier = Modifier.testTag("capture_imu_accel"))
            Text("ω(°/s) ${vector(imu?.gyroDps)}", fontSize = 10.sp, lineHeight = 13.sp,
                modifier = Modifier.testTag("capture_imu_gyro"))
            Text("BMI270 原生轴", fontSize = 9.sp, lineHeight = 12.sp,
                modifier = Modifier.testTag("capture_imu_native_axes"))
        }
    }
}

@Composable
private fun CapturePanel(content: @Composable ColumnScope.() -> Unit) {
    Surface(modifier = Modifier.fillMaxWidth(), color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(24.dp), border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp), content = content)
    }
}

@Composable
private fun AtomImagePreview(
    preview: () -> AtomPreviewFrame?, onDraw: (Long) -> Unit, rotation: Int, mirror: Boolean
) {
    // Availability changes composition; each new image only invalidates drawing.
    // Reading the frame in composition remeasures this preview at camera rate.
    val available by remember(preview) { derivedStateOf { preview() != null } }
    if (!available) {
        Text("等待 Atom 画面", color = Color.White, style = MaterialTheme.typography.bodySmall)
        return
    }
    Canvas(Modifier.fillMaxSize().testTag("capture_atom_image").semantics { contentDescription = "Atom 实时 RGB 画面" }) {
        val frame = preview() ?: return@Canvas
        val image = frame.image
        val rotated = rotation % 180 != 0
        val width = if (rotated) image.height else image.width
        val height = if (rotated) image.width else image.height
        val fit = minOf(size.width / width, size.height / height)
        withTransform({
            translate(size.width / 2, size.height / 2)
            if (mirror) scale(-1f, 1f, pivot = Offset.Zero)
            rotate(rotation.toFloat(), pivot = Offset.Zero)
            scale(fit, fit, pivot = Offset.Zero)
        }) {
            drawImage(image, topLeft = Offset(-image.width / 2f, -image.height / 2f))
        }
        onDraw(frame.receiptNs)
    }
}

@Immutable
private data class TofPreviewCells(val distanceMm: List<Int?>)

@Composable
private fun TofDistanceGrid(frame: CnhCaptureFrame?, modifier: Modifier = Modifier) {
    val cells = remember(frame) {
        TofPreviewCells((0 until 64).map { display ->
            val raw = 63 - display
            frame?.distanceMm?.getOrNull(raw)?.takeIf {
                it > 0 && frame.targetStatus.getOrNull(raw) == 5 && (frame.targetCount.getOrNull(raw) ?: 0) > 0
            }
        })
    }
    TofCellsGrid(cells, modifier)
}

@Composable
private fun TofCellsGrid(cells: TofPreviewCells, modifier: Modifier) {
    Column(modifier.aspectRatio(1f).testTag("capture_tof_grid")) {
        repeat(8) { displayRow ->
            Row(Modifier.fillMaxWidth().weight(1f)) {
                repeat(8) { displayCol ->
                    val meters = cells.distanceMm[displayRow * 8 + displayCol]?.div(1000f)
                    val background = if (meters == null) Color(0xff263243) else Color.hsv((meters.coerceIn(0f, 3f) / 3f) * 210f, 0.7f, 0.65f)
                    Box(Modifier.weight(1f).fillMaxSize().padding(0.5.dp).background(background)
                        .semantics { contentDescription = if (meters == null) "无读数" else "${format(meters.toDouble(), 1)} 米" },
                        contentAlignment = Alignment.Center) {
                        Text(if (meters == null) "无" else format(meters.toDouble(), 1), color = Color.White,
                            fontSize = 8.sp, textAlign = TextAlign.Center)
                    }
                }
            }
        }
    }
}

private fun freshness(ageMs: Long?, ready: Boolean): String = when {
    ageMs == null -> "等待数据"
    !ready -> "已过期"
    else -> "${ageMs}ms前"
}
private fun angle(value: Double?): String = value?.let { "${format(it, 1)}°" } ?: "—"
private fun vector(values: List<Double>?): String = values?.joinToString(" / ") { format(it, 2) } ?: "— / — / —"
private fun format(value: Double, digits: Int): String = String.format(Locale.US, "%.${digits}f", value)
