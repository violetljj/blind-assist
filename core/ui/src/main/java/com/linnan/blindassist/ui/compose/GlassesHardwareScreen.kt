package com.linnan.blindassist.ui.compose

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Link
import androidx.compose.material.icons.rounded.LinkOff
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
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
import com.linnan.blindassist.model.ReplayScenario

@Composable
fun GlassesHardwareScreen(
    state: GlassesSimulatorUiState,
    language: AppLanguage,
    onBack: () -> Unit,
    onConnect: () -> Unit,
    onDisconnect: () -> Unit,
    onStartLiveAssist: (String) -> Unit,
    @Suppress("UNUSED_PARAMETER") onReplayScenarioSelected: (ReplayScenario) -> Unit,
    @Suppress("UNUSED_PARAMETER") onStartReplay: (ReplayScenario) -> Unit,
    modifier: Modifier = Modifier
) {
    val english = language == AppLanguage.EN
    BackHandler(onBack = onBack)
    ScreenColumn(
        modifier = modifier
            .statusBarsPadding()
            .testTag("glasses_device_screen")
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(
                onClick = onBack,
                modifier = Modifier.semantics {
                    contentDescription = if (english) "Back from glasses device center" else "返回功能页，离开眼镜设备中心"
                }
            ) {
                Icon(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = null, tint = BaHomeInk)
            }
            Text(
                text = if (english) "Glasses device center" else "眼镜外设连接中心",
                style = MaterialTheme.typography.headlineSmall,
                color = BaHomeInk,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.semantics { heading() }
            )
        }
        Spacer(Modifier.height(10.dp))
        Text(
            text = if (english) {
                "Real external-hardware connection. Current adapter: AtomS3R-M12 + ToF4M over local Wi-Fi."
            } else {
                "真实外界硬件连接入口。当前适配器：AtomS3R-M12 + ToF4M 局域网连接。"
            },
            color = BaHomeAmber,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.testTag("glasses_hardware_boundary")
        )
        Spacer(Modifier.height(16.dp))
        HardwareStatusCard(state = state, language = language)
        Spacer(Modifier.height(16.dp))
        when (state.connectionState) {
            GlassesConnectionState.DISCONNECTED,
            GlassesConnectionState.CONNECTION_LOST -> Button(
                onClick = onConnect,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 52.dp)
                    .testTag("connect_glasses_device")
                    .semantics { contentDescription = if (english) "Connect external glasses hardware" else "连接眼镜外界硬件" },
                colors = ButtonDefaults.buttonColors(
                    containerColor = BaHomeActionEnd,
                    contentColor = BaHomeOnAction
                ),
                shape = BaShapeControl
            ) {
                Icon(Icons.Rounded.Link, contentDescription = null)
                Text(if (english) " Connect device" else " 连接设备")
            }
            GlassesConnectionState.CONNECTING -> Button(
                onClick = {},
                enabled = false,
                modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp).testTag("connecting_glasses_device")
            ) {
                Text(if (english) "Connecting to AtomS3R…" else "正在连接 AtomS3R…")
            }
            GlassesConnectionState.CONNECTED -> OutlinedButton(
                onClick = onDisconnect,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 52.dp)
                    .testTag("disconnect_glasses_device")
            ) {
                Icon(Icons.Rounded.LinkOff, contentDescription = null)
                Text(if (english) " Disconnect" else " 断开连接")
            }
        }
        if (state.connectionState == GlassesConnectionState.CONNECTED && state.streamReachable) {
            Spacer(Modifier.height(10.dp))
            Button(
                onClick = { onStartLiveAssist(state.endpoint) },
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 52.dp)
                    .testTag("start_glasses_live_assist"),
                colors = ButtonDefaults.buttonColors(
                    containerColor = BaHomeAmber,
                    contentColor = BaHomeOnAction
                )
            ) {
                Text(if (english) "Use live glasses camera" else "使用眼镜实时画面")
            }
        }
        state.errorMessage?.let { error ->
            Spacer(Modifier.height(10.dp))
            Text(
                text = if (english) "Connection failed: $error" else "连接失败：$error",
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.testTag("glasses_connection_error")
            )
        }
        Spacer(Modifier.height(18.dp))
        InfoStrip(
            icon = Icons.Rounded.Link,
            title = if (english) "Current integration boundary" else "当前接入边界",
            body = if (english) {
                "Live MJPEG can enter the existing detection and reminder pipeline. ToF is frame-bound metadata only; calibrated fusion remains paused."
            } else {
                "实时 MJPEG 可进入现有检测与提醒链路。ToF 当前仅作逐帧绑定元数据；标定融合仍暂缓。"
            }
        )
    }
}

@Composable
private fun HardwareStatusCard(state: GlassesSimulatorUiState, language: AppLanguage) {
    val english = language == AppLanguage.EN
    val connection = state.connectionState.hardwareLabel(language)
    val distance = state.tofRangeMm?.let { "$it mm" } ?: if (english) "Unavailable" else "不可用"
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("glasses_device_status")
            .semantics(mergeDescendants = true) {
                contentDescription = "$connection, ${state.endpoint}, $distance"
                stateDescription = connection
            },
        shape = BaShapeCard,
        colors = CardDefaults.cardColors(containerColor = BaHomeSurface),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
        border = BorderStroke(1.dp, BaHomeHairline.copy(alpha = 0.82f))
    ) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(if (english) "External device status" else "外界硬件状态", color = BaHomeInk, fontWeight = FontWeight.Bold)
            Text(connection, color = if (state.connectionState == GlassesConnectionState.CONNECTED) BaHomeGreen else BaHomeTextMuted)
            Text(if (english) "Endpoint: ${state.endpoint}" else "设备地址：${state.endpoint}", color = BaHomeTextMuted)
            state.firmwareVersion?.let { Text(if (english) "Firmware: $it" else "固件：$it", color = BaHomeTextMuted) }
            state.wifiRssiDbm?.let { Text(if (english) "Wi-Fi RSSI: $it dBm" else "Wi-Fi 信号：$it dBm", color = BaHomeTextMuted) }
            Text(if (english) "ToF: $distance" else "ToF 距离：$distance", color = BaHomeTextMuted)
            Text(
                if (state.streamReachable) {
                    if (english) "MJPEG endpoint reachable" else "MJPEG 视频端点可达"
                } else {
                    if (english) "MJPEG endpoint not verified" else "MJPEG 视频端点未验证"
                },
                color = BaHomeTextMuted
            )
        }
    }
}

private fun GlassesConnectionState.hardwareLabel(language: AppLanguage): String {
    val english = language == AppLanguage.EN
    return when (this) {
        GlassesConnectionState.DISCONNECTED -> if (english) "Disconnected" else "未连接"
        GlassesConnectionState.CONNECTING -> if (english) "Connecting" else "连接中"
        GlassesConnectionState.CONNECTED -> if (english) "Connected" else "已连接"
        GlassesConnectionState.CONNECTION_LOST -> if (english) "Connection failed or lost" else "连接失败或已断开"
    }
}
