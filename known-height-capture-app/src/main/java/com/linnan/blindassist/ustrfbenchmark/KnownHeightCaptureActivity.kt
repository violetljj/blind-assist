package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
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
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
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
        setContent { MaterialTheme(colorScheme = captureColorScheme()) { CaptureRoute() } }
    }

    @Composable
    private fun CaptureRoute() {
        val mountPreferences = remember { getSharedPreferences("fixed_mount", MODE_PRIVATE) }
        val previewView = remember {
            PreviewView(this).apply {
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                scaleType = PreviewView.ScaleType.FIT_CENTER
            }
        }
        var runState by remember { mutableStateOf<CaptureRunState>(CaptureRunState.Idle) }
        var sessionId by rememberSaveable { mutableStateOf(defaultSessionId(CapturePhase.DEV)) }
        var phaseName by rememberSaveable { mutableStateOf(CapturePhase.DEV.name) }
        var mountId by rememberSaveable { mutableStateOf(mountPreferences.getString("mount_id", "固定支架") ?: "固定支架") }
        var methodName by rememberSaveable { mutableStateOf(MeasurementMethod.SAMSUNG_QUICK_MEASURE.name) }
        var instrumentErrorCm by rememberSaveable { mutableStateOf(MeasurementMethod.SAMSUNG_QUICK_MEASURE.suggestedInstrumentErrorCm) }
        var height1 by rememberSaveable { mutableStateOf(mountPreferences.getString("camera_height_cm", "") ?: "") }
        var height2 by rememberSaveable { mutableStateOf("") }
        var height3 by rememberSaveable { mutableStateOf("") }
        var nearDistance by rememberSaveable { mutableStateOf("") }
        var middleDistance by rememberSaveable { mutableStateOf("") }
        var farDistance by rememberSaveable { mutableStateOf("") }
        var developmentDistanceCm by rememberSaveable { mutableStateOf("") }
        val phase = CapturePhase.valueOf(phaseName)
        val method = MeasurementMethod.valueOf(methodName)
        val form = CaptureFormState(
            sessionId = sessionId,
            phase = phase,
            mountProfileId = mountId,
            measurementMethod = method,
            instrumentErrorCm = instrumentErrorCm,
            heightReading1Cm = height1,
            heightReading2Cm = height2,
            heightReading3Cm = height3,
            nearDistanceM = nearDistance,
            middleDistanceM = middleDistance,
            farDistanceM = farDistance,
            developmentDistanceCm = developmentDistanceCm,
        )
        val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
            val complete = runState as? CaptureRunState.Complete
            if (uri != null && complete != null) {
                val ok = runCatching { exportSession(File(complete.sessionDirectory), uri) }.isSuccess
                Toast.makeText(this, if (ok) "采集包已导出" else "导出失败，请重试", Toast.LENGTH_LONG).show()
            }
        }
        fun beginCapture() {
            captureEngine?.stop()
            captureEngine = KnownHeightCaptureEngine(this, this, previewView) { runState = it }
            captureEngine?.start(KnownHeightCaptureRequest(form))
        }
        val cameraPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) beginCapture() else runState = CaptureRunState.Hold("相机权限被拒绝，无法采集")
        }
        DisposableEffect(Unit) { onDispose { captureEngine?.stop() } }

        CaptureScreen(
            form = form,
            runState = runState,
            previewView = previewView,
            onMountIdChange = { value -> mountId = value; mountPreferences.edit().putString("mount_id", value).apply() },
            onHeight1Change = { value -> height1 = numericInput(value); mountPreferences.edit().putString("camera_height_cm", height1).apply() },
            onDevelopmentDistanceChange = { developmentDistanceCm = numericInput(it) },
            onOpenQuickMeasure = {
                runCatching { startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("ruler://com.samsung.android.ruler"))) }
                    .onFailure { Toast.makeText(this, "未找到三星快速测量", Toast.LENGTH_LONG).show() }
            },
            onStart = {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) beginCapture()
                else cameraPermission.launch(Manifest.permission.CAMERA)
            },
            onCancel = { captureEngine?.cancel() },
            onReset = {
                captureEngine?.stop()
                runState = CaptureRunState.Idle
                sessionId = defaultSessionId(CapturePhase.DEV)
                developmentDistanceCm = ""
            },
            onExport = { exportLauncher.launch("${form.sessionId}.zip") },
        )
    }

    override fun onDestroy() {
        captureEngine?.stop()
        super.onDestroy()
    }

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
private fun CaptureScreen(
    form: CaptureFormState,
    runState: CaptureRunState,
    previewView: PreviewView,
    onMountIdChange: (String) -> Unit,
    onHeight1Change: (String) -> Unit,
    onDevelopmentDistanceChange: (String) -> Unit,
    onOpenQuickMeasure: () -> Unit,
    onStart: () -> Unit,
    onCancel: () -> Unit,
    onReset: () -> Unit,
    onExport: () -> Unit,
) {
    val running = runState is CaptureRunState.Preparing || runState is CaptureRunState.Capturing
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Column { Text("快速采集", fontWeight = FontWeight.Bold); Text("测距 → 采集 → 下一目标", style = MaterialTheme.typography.labelSmall) } },
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
                SectionCard("固定支架 · 只填一次") {
                    OutlinedTextField(
                        value = form.mountProfileId,
                        onValueChange = onMountIdChange,
                        enabled = !running,
                        label = { Text("支架名称") },
                        placeholder = { Text("例如：三脚架A") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth().testTag("mount_id"),
                    )
                    MeasurementField("地面到后摄镜头中心（cm）", form.heightReading1Cm, onHeight1Change, !running, Modifier.fillMaxWidth(), "例如 143", "height_1")
                    Text(
                        "必须实际架高到 80–220 cm；15 cm 低支架不可用。",
                        color = MaterialTheme.colorScheme.error,
                        fontWeight = FontWeight.SemiBold,
                        style = MaterialTheme.typography.bodySmall,
                    )
                    form.heightReading1Cm.toDoubleOrNull()?.let { heightCm ->
                        Text(
                            "当前填写：%.0f cm（%.2f m），请和镜头实际位置核对。".format(heightCm, heightCm / 100.0),
                            color = MaterialTheme.colorScheme.primary,
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    Text("核对后会自动保存，下次不需要重复填写。", style = MaterialTheme.typography.bodySmall)
                }
            }
            item {
                SectionCard("当前目标") {
                    Button(onClick = onOpenQuickMeasure, enabled = !running, modifier = Modifier.fillMaxWidth().height(54.dp)) {
                        Text("打开三星快速测量")
                    }
                    MeasurementField(
                        "快速测量读数（厘米）",
                        form.developmentDistanceCm,
                        onDevelopmentDistanceChange,
                        !running,
                        Modifier.fillMaxWidth(),
                        "例如 29",
                        "development_distance_cm",
                    )
                    form.developmentDistanceCm.toDoubleOrNull()?.let { distanceCm ->
                        Text("将记录为 %.2f 米".format(distanceCm / 100.0), color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.SemiBold)
                    }
                    Text("三星 AR 测距仅作开发参考；采集包自动标记 DEVELOPMENT_ONLY。", style = MaterialTheme.typography.bodySmall)
                }
            }
            item {
                val problems = form.validationProblems()
                SectionCard(if (problems.isEmpty()) "✓ 可以采集" else "还差 ${problems.size} 项") {
                    if (problems.isEmpty()) Text("距离已记录，直接开始。", color = MaterialTheme.colorScheme.primary)
                    else problems.forEach { Text("• $it", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
                }
            }
            if (running) {
                item {
                    Card {
                        Box(Modifier.fillMaxWidth().aspectRatio(4f / 3f), contentAlignment = Alignment.Center) {
                            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                        }
                    }
                }
            }
            item {
                when (runState) {
                    is CaptureRunState.Complete -> {
                        Button(onClick = onExport, modifier = Modifier.fillMaxWidth().height(54.dp)) { Text("导出 ZIP 采集包") }
                        Spacer(Modifier.height(8.dp))
                        OutlinedButton(onClick = onReset, modifier = Modifier.fillMaxWidth().height(50.dp)) { Text("采下一个目标") }
                    }
                    is CaptureRunState.Hold -> OutlinedButton(onClick = onReset, modifier = Modifier.fillMaxWidth().height(52.dp)) { Text("重新开始") }
                    else -> {
                        Button(onClick = onStart, enabled = form.canStart && !running, modifier = Modifier.fillMaxWidth().height(58.dp).testTag("start_capture")) {
                            Text(if (running) "正在采集 ${form.phase.frameTarget} 帧…" else "采集当前目标")
                        }
                        if (running) {
                            Spacer(Modifier.height(8.dp))
                            OutlinedButton(onClick = onCancel, modifier = Modifier.fillMaxWidth().height(48.dp)) { Text("停止并作废本组") }
                        }
                    }
                }
                Spacer(Modifier.height(20.dp))
            }
        }
    }
}

@Composable
private fun MeasurementField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    enabled: Boolean,
    modifier: Modifier,
    placeholder: String,
    tag: String,
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        enabled = enabled,
        label = { Text(label) },
        placeholder = { Text(placeholder) },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        singleLine = true,
        modifier = modifier.testTag(tag),
    )
}

@Composable
private fun RunStateCard(state: CaptureRunState, target: Int) {
    val (title, detail, progress) = when (state) {
        CaptureRunState.Idle -> Triple("准备采集", "填写当前快速测量距离，然后一键采集", null)
        is CaptureRunState.Preparing -> Triple("正在准备", state.message, null)
        is CaptureRunState.Capturing -> Triple("正在采集 ${state.captured} / ${state.target}", "保持支架不动，不要遮挡镜头", state.captured.toFloat() / state.target)
        is CaptureRunState.Complete -> Triple("采集完成", "${state.captured} 帧和测量记录均已保存", 1f)
        is CaptureRunState.Hold -> Triple("本组已作废", state.reason, null)
    }
    Card(colors = CardDefaults.cardColors(containerColor = when (state) { is CaptureRunState.Hold -> MaterialTheme.colorScheme.errorContainer; is CaptureRunState.Complete -> MaterialTheme.colorScheme.primaryContainer; else -> MaterialTheme.colorScheme.surfaceVariant })) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
            Text(detail, style = MaterialTheme.typography.bodyMedium)
            if (progress != null) LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth(), gapSize = 0.dp)
            if (state is CaptureRunState.Idle) Text("本组目标：$target 帧", style = MaterialTheme.typography.labelMedium)
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
