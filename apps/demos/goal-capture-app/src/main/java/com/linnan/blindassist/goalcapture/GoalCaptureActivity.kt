package com.linnan.blindassist.goalcapture

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class GoalCaptureActivity : ComponentActivity() {
    private var engine: GoalCaptureEngine? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent { MaterialTheme { GoalCaptureRoute() } }
    }

    @Composable
    private fun GoalCaptureRoute() {
        val preview = remember {
            PreviewView(this).apply {
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                scaleType = PreviewView.ScaleType.FIT_CENTER
            }
        }
        var planResult by remember { mutableStateOf(runCatching { loadPlan() }) }
        var state by remember {
            mutableStateOf<RecorderState>(
                planResult.fold(
                    onSuccess = { RecorderState.Preparing("capture plan 已验证；等待相机权限") },
                    onFailure = { RecorderState.Hold("capture plan 不可用：${it.message ?: it.javaClass.simpleName}") },
                ),
            )
        }
        fun startEngine() {
            val plan = planResult.getOrElse { return }
            if (engine != null) return
            val root = requireNotNull(getExternalFilesDir(EXTERNAL_ROOT))
                .resolve("sessions")
                .resolve(plan.bodySha256.take(16))
            engine = GoalCaptureEngine(this, this, preview, plan, root) { state = it }.also { it.start() }
        }
        val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startEngine() else state = RecorderState.Hold("相机权限被拒绝；没有生成任何可评价 receipt")
        }
        val exporter = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
            val complete = state as? RecorderState.Complete
            if (uri != null && complete != null) {
                val ok = runCatching { exportSession(complete.sessionDirectory, uri) }.isSuccess
                Toast.makeText(this, if (ok) "采集包已导出" else "导出失败", Toast.LENGTH_LONG).show()
            }
        }
        val planImporter = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) {
                planResult = runCatching { importPlan(uri) }
                state = planResult.fold(
                    onSuccess = { RecorderState.Preparing("capture plan 已验证；等待相机权限") },
                    onFailure = { RecorderState.Hold("capture plan 导入失败：${it.message ?: it.javaClass.simpleName}") },
                )
            }
        }
        DisposableEffect(Unit) { onDispose { engine?.shutdown() } }

        GoalCaptureScreen(
            state = state,
            preview = preview,
            onEnableCamera = {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    startEngine()
                } else {
                    permission.launch(Manifest.permission.CAMERA)
                }
            },
            onImportPlan = { planImporter.launch(arrayOf("application/json", "text/plain")) },
            onStartEpisode = { engine?.startEpisode() },
            onStopEpisode = { engine?.stopEpisode() },
            onCancel = { engine?.cancelSession() },
            onExport = {
                val complete = state as? RecorderState.Complete
                if (complete != null) exporter.launch("goal-capture-${complete.sessionDirectory.name}.zip")
            },
            onOpenAppSettings = {
                startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName")))
            },
        )
    }

    override fun onDestroy() {
        engine?.shutdown()
        super.onDestroy()
    }

    private fun loadPlan(): CapturePlan {
        val inbox = requireNotNull(getExternalFilesDir(EXTERNAL_ROOT)).resolve("inbox").apply { mkdirs() }
        val file = inbox.resolve("capture_plan.json")
        require(file.isFile) { "请先放入 ${file.absolutePath}" }
        return CapturePlanParser.parse(file.readText())
    }

    private fun importPlan(source: Uri): CapturePlan {
        val text = contentResolver.openInputStream(source).use { input ->
            requireNotNull(input) { "无法读取所选文件" }
            input.bufferedReader().readText()
        }
        val plan = CapturePlanParser.parse(text)
        val externalRoot = requireNotNull(getExternalFilesDir(EXTERNAL_ROOT))
        val session = externalRoot.resolve("sessions").resolve(plan.bodySha256.take(16))
        require(!session.exists()) { "该 plan 已有 session；必须重新 arm，不能覆盖或续写" }
        val inbox = externalRoot.resolve("inbox").apply { mkdirs() }
        val destination = inbox.resolve("capture_plan.json")
        val temporary = inbox.resolve("capture_plan.json.tmp")
        temporary.writeText(text)
        Files.move(
            temporary.toPath(),
            destination.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
        return CapturePlanParser.parse(destination.readText())
    }

    private fun exportSession(root: File, destination: Uri) {
        require(root.isDirectory)
        contentResolver.openOutputStream(destination).use { raw ->
            requireNotNull(raw)
            ZipOutputStream(raw).use { zip ->
                root.walkTopDown().filter(File::isFile).forEach { file ->
                    zip.putNextEntry(ZipEntry(file.relativeTo(root).invariantSeparatorsPath))
                    file.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
            }
        }
    }

    private companion object {
        const val EXTERNAL_ROOT = "prospective-goal-capture"
    }
}

@Composable
private fun GoalCaptureScreen(
    state: RecorderState,
    preview: PreviewView,
    modifier: Modifier = Modifier,
    onEnableCamera: () -> Unit,
    onImportPlan: () -> Unit,
    onStartEpisode: () -> Unit,
    onStopEpisode: () -> Unit,
    onCancel: () -> Unit,
    onExport: () -> Unit,
    onOpenAppSettings: () -> Unit,
) {
    val active = state is RecorderState.Ready || state is RecorderState.RecordingEpisode || state is RecorderState.Finalizing
    Column(
        modifier = modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("目标入口 · Prospective 研究采集", style = MaterialTheme.typography.headlineSmall)
        Text("完整 roster 按冻结顺序采集；每段走近入口，在入口仍在画面中时停止。禁止回放、补录和人工选帧。")
        StateCard(state)
        if (active) {
            Card {
                Box(Modifier.fillMaxWidth().aspectRatio(4f / 3f), contentAlignment = Alignment.Center) {
                    AndroidView(factory = { preview }, modifier = Modifier.fillMaxSize().testTag("goal_capture_preview"))
                }
            }
        }
        when (state) {
            is RecorderState.Preparing -> Button(onClick = onEnableCamera, modifier = Modifier.fillMaxWidth().height(56.dp)) {
                Text("验证权限并启动后置相机")
            }
            is RecorderState.Ready -> Button(onClick = onStartEpisode, modifier = Modifier.fillMaxWidth().height(58.dp)) {
                Text("开始本段连续录像")
            }
            is RecorderState.RecordingEpisode -> Button(onClick = onStopEpisode, modifier = Modifier.fillMaxWidth().height(58.dp)) {
                Text("入口在画面中 · 停止本段")
            }
            is RecorderState.Complete -> Button(onClick = onExport, modifier = Modifier.fillMaxWidth().height(58.dp)) {
                Text("导出完整采集包")
            }
            is RecorderState.Hold -> {
                Button(onClick = onImportPlan, modifier = Modifier.fillMaxWidth().height(56.dp)) {
                    Text("选择并验证 capture_plan.json")
                }
                OutlinedButton(onClick = onOpenAppSettings, modifier = Modifier.fillMaxWidth().height(52.dp)) {
                    Text("打开应用设置")
                }
            }
            else -> Unit
        }
        if (active) {
            OutlinedButton(onClick = onCancel, modifier = Modifier.fillMaxWidth().height(50.dp)) {
                Text("取消并封存 HOLD（不生成 receipt）")
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun StateCard(state: RecorderState) {
    val episode = when (state) {
        is RecorderState.Ready -> state.episode
        is RecorderState.RecordingEpisode -> state.episode
        else -> null
    }
    val position = when (state) {
        is RecorderState.Ready -> state.episodeIndex to state.episodeCount
        is RecorderState.RecordingEpisode -> state.episodeIndex to state.episodeCount
        is RecorderState.Finalizing -> state.episodeIndex to state.episodeCount
        else -> null
    }
    val title = when (state) {
        RecorderState.LoadingPlan -> "读取 capture plan"
        is RecorderState.Preparing -> "准备中"
        is RecorderState.Ready -> "第 ${state.episodeIndex + 1}/${state.episodeCount} 段已就绪"
        is RecorderState.RecordingEpisode -> "正在录制第 ${state.episodeIndex + 1}/${state.episodeCount} 段"
        is RecorderState.Finalizing -> "正在校验第 ${state.episodeIndex + 1}/${state.episodeCount} 段"
        is RecorderState.Complete -> "完整 roster 已完成"
        is RecorderState.Hold -> "HOLD · 不可评价"
    }
    val detail = when (state) {
        is RecorderState.Preparing -> state.message
        is RecorderState.Complete -> "device receipt 已生成；导出后仍需 host hash/denominator 验证。"
        is RecorderState.Hold -> state.reason
        else -> episode?.goalTextOriginal ?: "请保持设备前向、连续录像至少 3 秒，最长 45 秒。"
    }
    Card(
        colors = CardDefaults.cardColors(
            containerColor = when (state) {
                is RecorderState.Hold -> MaterialTheme.colorScheme.errorContainer
                is RecorderState.Complete -> MaterialTheme.colorScheme.primaryContainer
                else -> MaterialTheme.colorScheme.surfaceVariant
            },
        ),
        modifier = Modifier.fillMaxWidth().semantics(mergeDescendants = true) {},
    ) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(detail)
            if (position != null) {
                LinearProgressIndicator(
                    progress = { position.first.toFloat() / position.second.coerceAtLeast(1) },
                    modifier = Modifier.fillMaxWidth(),
                    gapSize = 0.dp,
                )
            }
        }
    }
}
