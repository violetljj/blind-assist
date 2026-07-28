package com.linnan.blindassist.ui.compose

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.BatteryAlert
import androidx.compose.material.icons.rounded.Bluetooth
import androidx.compose.material.icons.rounded.LinkOff
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.AssistInputSource
import com.linnan.blindassist.model.ReplayScenario
import kotlinx.coroutines.delay

@Composable
fun GlassesSimulatorScreen(
    state: GlassesSimulatorUiState,
    language: AppLanguage,
    onBack: () -> Unit,
    onConnect: () -> Unit,
    onConnectionCompleted: () -> Unit,
    onLowBattery: () -> Unit,
    onDisconnect: () -> Unit,
    onReset: () -> Unit,
    onReplayScenarioSelected: (ReplayScenario) -> Unit,
    onStartReplay: (ReplayScenario) -> Unit,
    modifier: Modifier = Modifier
) {
    val english = language == AppLanguage.EN
    BackHandler(onBack = onBack)
    LaunchedEffect(state.connectionState) {
        if (state.connectionState == GlassesConnectionState.CONNECTING) {
            delay(CONNECTION_DELAY_MS)
            onConnectionCompleted()
        }
    }

    ScreenColumn(modifier = modifier.testTag("glasses_simulator_screen")) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(
                onClick = onBack,
                modifier = Modifier.semantics {
                    contentDescription = if (english) {
                        "Back from simulated glasses center"
                    } else {
                        "返回功能页，离开眼镜模拟中心"
                    }
                }
            ) {
                Icon(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = null, tint = BaText)
            }
            Text(
                text = if (english) "Simulated glasses center" else "眼镜设备模拟中心",
                style = MaterialTheme.typography.headlineSmall,
                color = BaText,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.semantics { heading() }
            )
        }
        Spacer(Modifier.height(10.dp))
        Text(
            text = if (english) {
                "Simulation only: no Bluetooth scan, no network access, and no real glasses connection."
            } else {
                "仅用于模拟：不会扫描蓝牙、不会联网，也未连接真实眼镜。"
            },
            color = BaAmber,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.testTag("glasses_simulation_boundary")
        )
        Spacer(Modifier.height(16.dp))

        SimulatorStatusCard(state = state, language = language)
        Spacer(Modifier.height(16.dp))

        when (state.connectionState) {
            GlassesConnectionState.DISCONNECTED,
            GlassesConnectionState.CONNECTION_LOST -> {
                Button(
                    onClick = onConnect,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 52.dp)
                        .testTag("simulate_glasses_connect")
                        .semantics { contentDescription = if (english) "Start simulated glasses connection" else "开始模拟连接眼镜设备" },
                    colors = ButtonDefaults.buttonColors(containerColor = BaMint, contentColor = BaInk)
                ) {
                    Icon(Icons.Rounded.Bluetooth, contentDescription = null)
                    Text(if (english) " Simulate connection" else " 模拟连接")
                }
            }
            GlassesConnectionState.CONNECTING -> {
                Button(
                    onClick = {},
                    enabled = false,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 52.dp)
                        .testTag("simulate_glasses_connecting")
                        .semantics { contentDescription = if (english) "Simulated glasses connecting" else "模拟眼镜连接中" }
                ) {
                    Text(if (english) "Simulated connection in progress…" else "模拟连接中…")
                }
            }
            GlassesConnectionState.CONNECTED -> {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedButton(
                        onClick = onLowBattery,
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(min = 52.dp)
                            .testTag("simulate_low_battery")
                            .semantics { contentDescription = if (english) "Simulate glasses low battery" else "模拟眼镜低电量" }
                    ) {
                        Icon(Icons.Rounded.BatteryAlert, contentDescription = null)
                        Text(if (english) " Simulate low battery" else " 模拟低电量")
                    }
                    OutlinedButton(
                        onClick = onDisconnect,
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(min = 52.dp)
                            .testTag("simulate_disconnect")
                            .semantics { contentDescription = if (english) "Simulate glasses disconnection" else "模拟眼镜断连" }
                    ) {
                        Icon(Icons.Rounded.LinkOff, contentDescription = null)
                        Text(if (english) " Simulate disconnect" else " 模拟断连")
                    }
                }
            }
        }

        Spacer(Modifier.height(10.dp))
        OutlinedButton(
            onClick = onReset,
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 48.dp)
                .testTag("reset_glasses_simulation")
                .semantics { contentDescription = if (english) "Reset glasses simulation" else "重置眼镜设备模拟" }
        ) {
            Icon(Icons.Rounded.Refresh, contentDescription = null)
            Text(if (english) " Reset simulation" else " 重置模拟")
        }

        if (state.debugReplayAvailable && state.connectionState == GlassesConnectionState.CONNECTED) {
            Spacer(Modifier.height(20.dp))
            ReplaySimulationControls(
                selectedScenario = state.selectedReplayScenario,
                language = language,
                onScenarioSelected = onReplayScenarioSelected,
                onStartReplay = { onStartReplay(state.selectedReplayScenario) }
            )
        }
    }
}

@Composable
private fun SimulatorStatusCard(state: GlassesSimulatorUiState, language: AppLanguage) {
    val english = language == AppLanguage.EN
    val connection = state.connectionState.label(language)
    val battery = state.batteryPercent?.let { "$it%" } ?: if (english) "Not simulated" else "未模拟"
    val input = when (state.selectedInput) {
        AssistInputSource.PHONE_CAMERA -> if (english) "Simulated phone camera source" else "模拟手机摄像头来源"
        AssistInputSource.OFFLINE_REPLAY -> if (english) "Simulated offline replay source" else "模拟离线回放来源"
    }
    val feedback = if (state.connectionState == GlassesConnectionState.CONNECTED) {
        if (english) "Simulated speech and vibration link available" else "模拟语音与震动反馈链路可用"
    } else {
        if (english) "Simulated feedback link unavailable" else "模拟反馈链路不可用"
    }
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("glasses_simulator_status")
            .semantics(mergeDescendants = true) {
                contentDescription = "$connection, $battery, $input, $feedback"
                stateDescription = connection
            },
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(if (english) "Simulated device status" else "模拟设备状态", color = BaText, fontWeight = FontWeight.Bold)
            Text(connection, color = BaMint)
            Text(if (english) "Simulated battery: $battery" else "模拟电量：$battery", color = BaTextMuted)
            Text(input, color = BaTextMuted)
            Text(feedback, color = BaTextMuted)
        }
    }
}

@Composable
private fun ReplaySimulationControls(
    selectedScenario: ReplayScenario,
    language: AppLanguage,
    onScenarioSelected: (ReplayScenario) -> Unit,
    onStartReplay: () -> Unit
) {
    val english = language == AppLanguage.EN
    Text(
        if (english) "Simulated offline replay (debug)" else "模拟离线回放（调试）",
        color = BaText,
        fontWeight = FontWeight.Bold,
        modifier = Modifier.semantics { heading() }
    )
    Text(
        if (english) "Local test images enter the real detection and reminder pipeline." else "本地测试素材将进入真实检测与提醒链路。",
        color = BaTextMuted,
        style = MaterialTheme.typography.bodySmall
    )
    Spacer(Modifier.height(10.dp))
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        ReplayScenario.entries.forEach { scenario ->
            val label = scenario.label(language)
            FilterChip(
                selected = selectedScenario == scenario,
                onClick = { onScenarioSelected(scenario) },
                label = { Text(label) },
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("replay_scenario_${scenario.name.lowercase()}")
                    .semantics {
                        contentDescription = if (english) "Select simulated replay scenario: $label" else "选择模拟回放场景：$label"
                        stateDescription = if (selectedScenario == scenario) {
                            if (english) "Selected simulated scenario" else "已选模拟场景"
                        } else {
                            if (english) "Unselected simulated scenario" else "未选模拟场景"
                        }
                    }
            )
        }
    }
    Spacer(Modifier.height(12.dp))
    Button(
        onClick = onStartReplay,
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 52.dp)
            .testTag("start_offline_replay")
            .semantics { contentDescription = if (english) "Start simulated offline replay" else "开始模拟离线回放" },
        colors = ButtonDefaults.buttonColors(containerColor = BaSky, contentColor = BaInk)
    ) {
        Icon(Icons.Rounded.PlayArrow, contentDescription = null)
        Text(if (english) " Start simulated replay" else " 开始模拟离线回放")
    }
}

private fun GlassesConnectionState.label(language: AppLanguage): String {
    val english = language == AppLanguage.EN
    return when (this) {
        GlassesConnectionState.DISCONNECTED -> if (english) "Simulated status: disconnected" else "模拟状态：未连接"
        GlassesConnectionState.CONNECTING -> if (english) "Simulated status: connecting" else "模拟状态：连接中"
        GlassesConnectionState.CONNECTED -> if (english) "Simulated status: connected" else "模拟状态：已连接"
        GlassesConnectionState.CONNECTION_LOST -> if (english) "Simulated status: connection lost" else "模拟状态：连接已断开"
    }
}

private fun ReplayScenario.label(language: AppLanguage): String {
    val english = language == AppLanguage.EN
    return when (this) {
        ReplayScenario.HIGH_CENTER -> if (english) "Simulated high risk, center" else "模拟高风险·中央"
        ReplayScenario.MEDIUM_RIGHT -> if (english) "Simulated medium risk, right" else "模拟中风险·右侧"
        ReplayScenario.LOW_CENTER -> if (english) "Simulated low risk, center" else "模拟低风险·中央"
        ReplayScenario.NONE -> if (english) "Simulated no alert-level evidence" else "模拟未达提醒等级"
    }
}

private const val CONNECTION_DELAY_MS = 800L
