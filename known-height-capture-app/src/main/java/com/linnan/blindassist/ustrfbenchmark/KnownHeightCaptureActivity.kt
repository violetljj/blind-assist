package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.content.pm.PackageManager
import android.database.Cursor
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class KnownHeightCaptureActivity : ComponentActivity() {
    private var captureEngine: KnownHeightCaptureEngine? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.light(android.graphics.Color.rgb(156, 244, 201), android.graphics.Color.rgb(156, 244, 201)),
            navigationBarStyle = SystemBarStyle.light(android.graphics.Color.rgb(247, 251, 247), android.graphics.Color.rgb(247, 251, 247)),
        )
        setContent {
            MaterialTheme(colorScheme = captureColorScheme()) {
                KnownHeightCaptureRoute()
            }
        }
    }

    @Composable
    private fun KnownHeightCaptureRoute() {
        val previewView = remember {
            PreviewView(this).apply {
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                scaleType = PreviewView.ScaleType.FIT_CENTER
            }
        }
        var runState by remember { mutableStateOf<CaptureRunState>(CaptureRunState.Idle) }
        var sessionId by rememberSaveable { mutableStateOf(defaultSessionId(CapturePhase.P0)) }
        var phaseName by rememberSaveable { mutableStateOf(CapturePhase.P0.name) }
        var height by rememberSaveable { mutableStateOf("") }
        var uncertainty by rememberSaveable { mutableStateOf("0.01") }
        var mountId by rememberSaveable { mutableStateOf("") }
        var referenceUri by rememberSaveable { mutableStateOf<Uri?>(null) }
        var referenceName by rememberSaveable { mutableStateOf<String?>(null) }
        val phase = CapturePhase.valueOf(phaseName)
        val form = CaptureFormState(sessionId, phase, height, uncertainty, mountId, referenceName)

        val referencePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) {
                runCatching { contentResolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION) }
                referenceUri = uri
                referenceName = displayName(uri)
            }
        }
        val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
            val complete = runState as? CaptureRunState.Complete
            if (uri != null && complete != null) {
                val exported = runCatching { exportSession(File(complete.sessionDirectory), uri) }.isSuccess
                Toast.makeText(this, if (exported) "采集包已导出" else "导出失败，请重试", Toast.LENGTH_LONG).show()
            }
        }
        fun beginCapture() {
            val uri = referenceUri ?: return
            captureEngine?.stop()
            captureEngine = KnownHeightCaptureEngine(this, this, previewView) { runState = it }
            captureEngine?.start(KnownHeightCaptureRequest(form, uri))
        }
        val cameraPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) beginCapture() else runState = CaptureRunState.Hold("相机权限被拒绝，无法采集")
        }
        DisposableEffect(Unit) { onDispose { captureEngine?.stop() } }

        KnownHeightCaptureScreen(
            form = form,
            runState = runState,
            previewView = previewView,
            onSessionIdChange = { sessionId = it },
            onPhaseChange = {
                phaseName = it.name
                if (runState is CaptureRunState.Idle) sessionId = defaultSessionId(it)
            },
            onHeightChange = { height = numericInput(it) },
            onUncertaintyChange = { uncertainty = numericInput(it) },
            onMountIdChange = { mountId = it },
            onPickReference = { referencePicker.launch(arrayOf("application/json", "text/*")) },
            onStart = {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) beginCapture()
                else cameraPermission.launch(Manifest.permission.CAMERA)
            },
            onCancel = { captureEngine?.cancel() },
            onReset = {
                captureEngine?.stop()
                runState = CaptureRunState.Idle
                sessionId = defaultSessionId(phase)
            },
            onExport = { exportLauncher.launch("${form.sessionId}.zip") },
        )
    }

    override fun onDestroy() {
        captureEngine?.stop()
        super.onDestroy()
    }

    private fun displayName(uri: Uri): String = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { cursor: Cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
        ?: uri.lastPathSegment
        ?: "reference.json"

    private fun defaultSessionId(phase: CapturePhase): String = "${phase.name}-${SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())}"
    private fun numericInput(value: String): String = value.filterIndexed { index, char -> char.isDigit() || (char == '.' && index > 0) }.take(6)

    private fun exportSession(directory: File, destination: Uri) {
        require(directory.isDirectory)
        contentResolver.openOutputStream(destination).use { output ->
            requireNotNull(output)
            ZipOutputStream(output).use { zip ->
                directory.walkTopDown().filter(File::isFile).forEach { file ->
                    zip.putNextEntry(ZipEntry(file.relativeTo(directory).invariantSeparatorsPath))
                    file.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun KnownHeightCaptureScreen(
    form: CaptureFormState,
    runState: CaptureRunState,
    previewView: PreviewView,
    onSessionIdChange: (String) -> Unit,
    onPhaseChange: (CapturePhase) -> Unit,
    onHeightChange: (String) -> Unit,
    onUncertaintyChange: (String) -> Unit,
    onMountIdChange: (String) -> Unit,
    onPickReference: () -> Unit,
    onStart: () -> Unit,
    onCancel: () -> Unit,
    onReset: () -> Unit,
    onExport: () -> Unit,
) {
    val running = runState is CaptureRunState.Preparing || runState is CaptureRunState.Capturing
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Column { Text("高度标定采集", fontWeight = FontWeight.Bold); Text("隔离 Shadow 工具 · 不产生导航提示", style = MaterialTheme.typography.labelSmall) } },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
            )
        },
    ) { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding).padding(horizontal = 16.dp).testTag("capture_form_list"),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item { Spacer(Modifier.height(2.dp)) }
            item { RunStateCard(runState, form.phase.frameTarget) }
            item {
                Card {
                    val previewRatio = if (running) 4f / 3f else 16f / 9f
                    Box(Modifier.fillMaxWidth().aspectRatio(previewRatio), contentAlignment = Alignment.Center) {
                        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                        if (runState is CaptureRunState.Idle || runState is CaptureRunState.Hold) {
                            Text("填写并检查信息后启动相机", color = Color.White, modifier = Modifier.padding(12.dp))
                        }
                    }
                }
            }
            item {
                SectionCard("1 · 本次采集") {
                    Text("阶段", style = MaterialTheme.typography.labelLarge)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        CapturePhase.entries.forEach { phase ->
                            FilterChip(selected = form.phase == phase, onClick = { onPhaseChange(phase) }, enabled = !running, label = { Text(phase.label) })
                        }
                    }
                    OutlinedTextField(value = form.sessionId, onValueChange = onSessionIdChange, enabled = !running, label = { Text("Session ID") }, supportingText = { Text("每次采集必须唯一") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                    OutlinedTextField(value = form.mountProfileId, onValueChange = onMountIdChange, enabled = !running, label = { Text("固定支架编号") }, placeholder = { Text("例如 tripod-A-145cm") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                }
            }
            item {
                SectionCard("2 · 现场量高") {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        OutlinedTextField(value = form.cameraHeightM, onValueChange = onHeightChange, enabled = !running, label = { Text("光心高度 (m)") }, placeholder = { Text("1.45") }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal), singleLine = true, modifier = Modifier.weight(1f))
                        OutlinedTextField(value = form.cameraHeightUncertaintyM, onValueChange = onUncertaintyChange, enabled = !running, label = { Text("误差 (m)") }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal), singleLine = true, modifier = Modifier.weight(1f))
                    }
                    Text("必须现场测量相机镜头光心；误差上限 0.02 m，不能用估计值。", style = MaterialTheme.typography.bodySmall)
                }
            }
            item {
                SectionCard("3 · 独立参考") {
                    Text(form.referenceDisplayName ?: "尚未选择卷尺/激光参考清单", style = MaterialTheme.typography.bodyMedium)
                    OutlinedButton(onClick = onPickReference, enabled = !running, modifier = Modifier.fillMaxWidth()) { Text(if (form.referenceDisplayName == null) "选择参考清单" else "更换参考清单") }
                    Text("App 会自动复制并计算 SHA-256；参考数据只供离线评价，不进入模型推理。", style = MaterialTheme.typography.bodySmall)
                }
            }
            item {
                val problems = form.validationProblems()
                SectionCard(if (problems.isEmpty()) "✓ 已满足启动条件" else "启动前还需处理 ${problems.size} 项") {
                    if (problems.isEmpty()) Text("固定支架后即可采集 ${form.phase.frameTarget} 帧。", color = MaterialTheme.colorScheme.primary)
                    else problems.forEach { Text("• $it", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
                }
            }
            item {
                if (runState is CaptureRunState.Complete || runState is CaptureRunState.Hold) {
                    if (runState is CaptureRunState.Complete) {
                        Button(onClick = onExport, modifier = Modifier.fillMaxWidth().height(52.dp)) { Text("导出 ZIP 采集包") }
                        Spacer(Modifier.height(8.dp))
                    }
                    OutlinedButton(onClick = onReset, modifier = Modifier.fillMaxWidth().height(52.dp)) { Text("新建一次采集") }
                } else {
                    Button(onClick = onStart, enabled = form.canStart && !running, modifier = Modifier.fillMaxWidth().height(56.dp).testTag("start_capture")) {
                        Text(if (running) "采集中，请保持支架不动" else "开始采集 ${form.phase.frameTarget} 帧")
                    }
                    if (running) {
                        Spacer(Modifier.height(8.dp))
                        OutlinedButton(onClick = onCancel, modifier = Modifier.fillMaxWidth().height(48.dp)) { Text("停止并记为 HOLD") }
                    }
                }
                Spacer(Modifier.height(20.dp))
            }
        }
    }
}

@Composable
private fun RunStateCard(state: CaptureRunState, target: Int) {
    val (title, detail, progress) = when (state) {
        CaptureRunState.Idle -> Triple("等待准备", "按 1 → 2 → 3 完成现场信息", null)
        is CaptureRunState.Preparing -> Triple("正在准备", state.message, null)
        is CaptureRunState.Capturing -> Triple("正在采集 ${state.captured} / ${state.target}", "保持支架固定，不要遮挡镜头", state.captured.toFloat() / state.target)
        is CaptureRunState.Complete -> Triple("采集完成", "已保存 ${state.captured} 帧\n${state.sessionDirectory}", 1f)
        is CaptureRunState.Hold -> Triple("已停止 · HOLD", state.reason, null)
    }
    Card(colors = CardDefaults.cardColors(containerColor = when (state) { is CaptureRunState.Hold -> MaterialTheme.colorScheme.errorContainer; is CaptureRunState.Complete -> MaterialTheme.colorScheme.primaryContainer; else -> MaterialTheme.colorScheme.surfaceVariant })) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
            Text(detail, style = MaterialTheme.typography.bodyMedium)
            if (progress != null) LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth(), gapSize = 0.dp)
            if (state is CaptureRunState.Idle) Text("目标：$target 帧", style = MaterialTheme.typography.labelMedium)
        }
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            content()
        }
    }
}

@Composable
private fun captureColorScheme() = androidx.compose.material3.lightColorScheme(
    primary = Color(0xFF006C4C),
    primaryContainer = Color(0xFF9CF4C9),
    secondary = Color(0xFF4D6358),
    background = Color(0xFFF7FBF7),
    surface = Color(0xFFF7FBF7),
)
