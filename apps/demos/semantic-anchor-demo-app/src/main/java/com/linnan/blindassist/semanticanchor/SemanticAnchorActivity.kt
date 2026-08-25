package com.linnan.blindassist.semanticanchor

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
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
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class SemanticAnchorActivity : ComponentActivity() {
    private var engine: SemanticAnchorCameraEngine? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent { MaterialTheme { SemanticAnchorRoute() } }
    }

    override fun onDestroy() {
        engine?.shutdown()
        super.onDestroy()
    }

    @Composable
    private fun SemanticAnchorRoute() {
        val preview = remember {
            PreviewView(this).apply {
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
        }
        val session = remember { SemanticAnchorSession(DEFAULT_MARKER_TARGET) }
        var state by remember { mutableStateOf(session.state) }
        var draftMode by remember { mutableStateOf(AnchorMode.MARKER) }
        var draftValue by remember { mutableStateOf(DEFAULT_MARKER_TARGET.value) }
        var status by remember { mutableStateOf("RESEARCH-ONLY · 默认 App 未改变") }
        var cameraVisible by remember { mutableStateOf(false) }
        var replayRunning by remember { mutableStateOf(false) }
        val scope = rememberCoroutineScope()

        fun applyTarget(mode: AnchorMode = draftMode, value: String = draftValue): Boolean {
            val candidate = runCatching { AnchorTarget(mode, value) }.getOrElse {
                status = "目标不能为空"
                return false
            }
            state = session.reset(candidate)
            engine?.updateTarget(candidate)
            status = "目标已更新；旧 lock 已清空"
            return true
        }

        fun startCamera() {
            if (engine == null) {
                engine = SemanticAnchorCameraEngine(
                    context = this,
                    lifecycleOwner = this,
                    previewView = preview,
                    initialTarget = state.target,
                    onObservation = { state = session.observe(it) },
                    onStatus = { status = it },
                ).also { it.start() }
            }
            engine?.setReplayPaused(false)
            state = session.reset(state.target)
            cameraVisible = true
            status = "LIVE · 新会话从 SEARCH 开始"
        }

        val cameraPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else status = "相机权限被拒绝；仍可运行无相机 replay canary"
        }
        DisposableEffect(Unit) { onDispose { engine?.shutdown(); engine = null } }

        SemanticAnchorScreen(
            state = state,
            status = status,
            preview = preview,
            cameraVisible = cameraVisible,
            draftMode = draftMode,
            draftValue = draftValue,
            replayRunning = replayRunning,
            guidanceArm = state.guidanceArm,
            onMode = { mode ->
                draftMode = mode
                draftValue = if (mode == AnchorMode.MARKER) DEFAULT_MARKER_TARGET.value else DEFAULT_OCR_TARGET.value
            },
            onValue = { draftValue = it },
            onGuidanceArm = { state = session.setGuidanceArm(it) },
            onApplyTarget = { applyTarget() },
            onStartCamera = {
                if (!applyTarget()) return@SemanticAnchorScreen
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    startCamera()
                } else {
                    cameraPermission.launch(Manifest.permission.CAMERA)
                }
            },
            onReplay = {
                if (replayRunning || !applyTarget()) return@SemanticAnchorScreen
                engine?.setReplayPaused(true)
                cameraVisible = false
                replayRunning = true
                status = "REPLAY CANARY · 非真实相机证据"
                scope.launch {
                    val targetValue = session.state.target.value
                    val script = buildList {
                        addAll(List(2) { listOf(targetValue) })
                        addAll(List(5) { emptyList() })
                        addAll(List(2) { listOf(targetValue) })
                    }
                    script.forEach { candidates ->
                        state = session.observe(AnchorObservation(candidates, "REPLAY CANARY"))
                        delay(220)
                    }
                    replayRunning = false
                    status = "REPLAY COMPLETE · 期待 REACQUIRED 1/1"
                }
            },
        )
    }

    private companion object {
        val DEFAULT_MARKER_TARGET = AnchorTarget(AnchorMode.MARKER, "BLINDASSIST:ANCHOR:17")
        val DEFAULT_OCR_TARGET = AnchorTarget(AnchorMode.OCR, "ROOM 302")
    }
}

@Composable
private fun SemanticAnchorScreen(
    state: AnchorUiState,
    status: String,
    preview: PreviewView,
    cameraVisible: Boolean,
    draftMode: AnchorMode,
    draftValue: String,
    replayRunning: Boolean,
    guidanceArm: GuidanceArm,
    onMode: (AnchorMode) -> Unit,
    onValue: (String) -> Unit,
    onGuidanceArm: (GuidanceArm) -> Unit,
    onApplyTarget: () -> Unit,
    onStartCamera: () -> Unit,
    onReplay: () -> Unit,
) {
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Semantic Anchor Lab", style = MaterialTheme.typography.headlineSmall)
        Text("Appearance 只提供相似性；QR/OCR 语义证据才拥有物理 referent 的 lock authority。")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(
                selected = draftMode == AnchorMode.MARKER,
                onClick = { onMode(AnchorMode.MARKER) },
                label = { Text("QR Marker") },
            )
            FilterChip(
                selected = draftMode == AnchorMode.OCR,
                onClick = { onMode(AnchorMode.OCR) },
                label = { Text("Natural Text") },
            )
        }
        OutlinedTextField(
            value = draftValue,
            onValueChange = onValue,
            label = { Text(if (draftMode == AnchorMode.MARKER) "Exact QR payload" else "OCR substring") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth().testTag("semantic_target_input"),
        )
        Text("Control arm · QR 物理边长固定 0.16 m · target-front standoff 0.65 m")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(
                selected = guidanceArm == GuidanceArm.CENTER_BASELINE,
                onClick = { onGuidanceArm(GuidanceArm.CENTER_BASELINE) },
                label = { Text("Center baseline") },
            )
            FilterChip(
                selected = guidanceArm == GuidanceArm.PNP_POSE,
                onClick = { onGuidanceArm(GuidanceArm.PNP_POSE) },
                label = { Text("PnP pose") },
            )
        }
        OutlinedButton(onClick = onApplyTarget, modifier = Modifier.fillMaxWidth()) {
            Text("应用目标并清空旧 lock")
        }
        PhaseCard(state)
        GuidanceCard(state.guidance)
        if (cameraVisible) {
            Card {
                AndroidView(
                    factory = { preview },
                    modifier = Modifier.fillMaxWidth().aspectRatio(4f / 3f).testTag("semantic_camera_preview"),
                )
            }
        }
        Text(status, style = MaterialTheme.typography.bodySmall, modifier = Modifier.testTag("semantic_runtime_status"))
        Button(onClick = onStartCamera, enabled = !replayRunning, modifier = Modifier.fillMaxWidth().height(56.dp)) {
            Text("启动实时相机 · fresh session")
        }
        OutlinedButton(onClick = onReplay, enabled = !replayRunning, modifier = Modifier.fillMaxWidth().height(52.dp)) {
            Text(if (replayRunning) "正在回放状态机…" else "运行自动 Replay Canary")
        }
        Text("Replay 只验证状态闭环与演示 UI，不计作真实相机 marker/OCR 结果。", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun GuidanceCard(guidance: MarkerGuidance) {
    val color = when (guidance.phase) {
        GuidancePhase.ARRIVE -> Color(0xFFC8F4D2)
        GuidancePhase.LOST -> Color(0xFFFFD8D8)
        GuidancePhase.SEARCH -> Color(0xFFE8EAF6)
        GuidancePhase.ALIGN -> Color(0xFFFFEDC2)
        GuidancePhase.ADVANCE -> Color(0xFFD6F3FF)
    }
    Card(colors = CardDefaults.cardColors(containerColor = color), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(guidance.command, style = MaterialTheme.typography.headlineMedium, modifier = Modifier.testTag("marker_guidance_command"))
            Text(guidance.detail, modifier = Modifier.testTag("marker_guidance_detail"))
        }
    }
}

@Composable
private fun PhaseCard(state: AnchorUiState) {
    val color = when (state.phase) {
        AnchorPhase.SEARCH -> Color(0xFFE8EAF6)
        AnchorPhase.LOCKED -> Color(0xFFD7F5DE)
        AnchorPhase.LOST -> Color(0xFFFFE2E2)
        AnchorPhase.REACQUIRED -> Color(0xFFD6F3FF)
    }
    Card(
        colors = CardDefaults.cardColors(containerColor = color),
        modifier = Modifier.fillMaxWidth().semantics(mergeDescendants = true) {},
    ) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(state.phase.name, style = MaterialTheme.typography.headlineMedium, modifier = Modifier.testTag("semantic_phase"))
            Text("${state.target.mode} · ${state.target.value}")
            Text(state.evidence)
            Text(
                "source=${state.source} · frames=${state.frameCount} · lock=${state.lockCount} · reacquire=${state.reacquisitionCount}",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.testTag("semantic_counters"),
            )
        }
    }
}
