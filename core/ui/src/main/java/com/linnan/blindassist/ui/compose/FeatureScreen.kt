package com.linnan.blindassist.ui.compose

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.LocalIndication
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.outlined.Bolt
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.Eco
import androidx.compose.material.icons.outlined.LightMode
import androidx.compose.material.icons.outlined.Shield
import androidx.compose.material.icons.outlined.Visibility
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.preferences.DailyUsageMode

@Suppress("UNUSED_PARAMETER")
@Composable
fun FeatureScreen(
    controls: AssistControlsUiState,
    modelStatus: String,
    appVersion: String,
    onOpenCamera: () -> Unit,
    onShowGlassesCenter: () -> Unit,
    onDailyUsageModeChange: (DailyUsageMode) -> Unit,
    modifier: Modifier = Modifier,
    glassesConnectionState: GlassesConnectionState = GlassesConnectionState.DISCONNECTED,
    onQuietShortcut: () -> Unit = {},
    onSensitiveShortcut: () -> Unit = {}
) {
    val language = controls.appLanguage
    val selectedMode = when (controls.alertProfile) {
        AlertProfile.QUIET -> HomeAssistMode.QUIET
        AlertProfile.SENSITIVE -> HomeAssistMode.SENSITIVE
        else -> HomeAssistMode.DAILY
    }
    val feedbackSummary = feedbackSummary(controls, language)

    Box(
        modifier = modifier
            .fillMaxSize()
            .appAtmosphere()
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 22.dp)
                .padding(top = 84.dp, bottom = 30.dp)
        ) {
            HomeBrandHeader(language = language)
            Spacer(Modifier.height(88.dp))

            Text(
                text = if (language == AppLanguage.EN) {
                    "Today, move with confidence"
                } else {
                    "今天，安心出发"
                },
                color = BaHomeInk,
                fontSize = 38.sp,
                lineHeight = 46.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = (-0.6).sp,
                modifier = Modifier.semantics { heading() }
            )
            Spacer(Modifier.height(28.dp))

            PrimaryAssistAction(
                subtitle = feedbackSummary,
                language = language,
                onClick = onOpenCamera
            )
            Spacer(Modifier.height(38.dp))

            Text(
                text = if (language == AppLanguage.EN) "Choose assist mode" else "选择辅助模式",
                color = BaHomeInk,
                fontSize = 18.sp,
                lineHeight = 24.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.semantics { heading() }
            )
            Spacer(Modifier.height(16.dp))

            HomeModeSelector(
                selectedMode = selectedMode,
                language = language,
                onDailyClick = { onDailyUsageModeChange(DailyUsageMode.GENERAL_DAILY) },
                onQuietClick = onQuietShortcut,
                onSensitiveClick = onSensitiveShortcut
            )
            Spacer(Modifier.height(28.dp))

            HomeInfoRow(
                icon = Icons.Outlined.Visibility,
                iconTint = BaHomeCobalt,
                title = glassesLabel(glassesConnectionState, language),
                onClick = onShowGlassesCenter,
                trailingIcon = Icons.Rounded.ChevronRight,
                modifier = Modifier.testTag("home_glasses_entry")
            )
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(1.dp)
                    .background(BaHomeHairline)
            )
            HomeInfoRow(
                icon = Icons.Outlined.Shield,
                iconTint = BaHomeGreen,
                title = if (language == AppLanguage.EN) {
                    "Processed on device · Images are not uploaded"
                } else {
                    "本地处理，不上传画面"
                }
            )
        }
    }
}

@Composable
private fun HomeBrandHeader(
    language: AppLanguage,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "BlindAssist",
            color = BaHomeInk,
            fontSize = 27.sp,
            lineHeight = 34.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = (-0.8).sp
        )
        Spacer(Modifier.width(20.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(9.dp)
                    .clip(CircleShape)
                    .background(BaHomeGreen)
            )
            Spacer(Modifier.width(8.dp))
            Text(
                text = if (language == AppLanguage.EN) "Ready" else "已就绪",
                color = BaHomeGreen,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@Composable
private fun PrimaryAssistAction(
    subtitle: String,
    language: AppLanguage,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val interactionSource = remember { MutableInteractionSource() }
    val pressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (pressed) 0.985f else 1f,
        animationSpec = tween(durationMillis = 120),
        label = "primary-assist-press"
    )
    val shape = RoundedCornerShape(24.dp)

    Surface(
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 118.dp)
            .graphicsLayer {
                scaleX = scale
                scaleY = scale
            },
        shape = shape,
        color = BaHomeActionEnd,
        shadowElevation = 8.dp
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(shape)
                .background(
                    Brush.horizontalGradient(
                        colors = listOf(BaHomeActionStart, BaHomeActionEnd)
                    )
                )
                .clickable(
                    interactionSource = interactionSource,
                    indication = LocalIndication.current,
                    role = Role.Button,
                    onClick = onClick
                )
                .testTag("home_primary_assist")
                .semantics {
                    contentDescription = if (language == AppLanguage.EN) {
                        "Start assist. $subtitle"
                    } else {
                        "开始辅助。$subtitle"
                    }
                }
                .padding(horizontal = 24.dp, vertical = 22.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Outlined.CameraAlt,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(42.dp)
            )
            Spacer(Modifier.width(22.dp))
            Box(
                Modifier
                    .width(1.dp)
                    .height(64.dp)
                    .background(Color.White.copy(alpha = 0.22f))
            )
            Spacer(Modifier.width(22.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    text = if (language == AppLanguage.EN) "Start assist" else "开始辅助",
                    color = Color.White,
                    fontSize = 27.sp,
                    lineHeight = 36.sp,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Spacer(Modifier.height(5.dp))
                Text(
                    text = subtitle,
                    color = BaHomeActionTextMuted,
                    style = MaterialTheme.typography.bodyMedium.copy(
                        fontSize = 14.sp,
                        lineHeight = 20.sp
                    ),
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
private fun HomeModeSelector(
    selectedMode: HomeAssistMode,
    language: AppLanguage,
    onDailyClick: () -> Unit,
    onQuietClick: () -> Unit,
    onSensitiveClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .height(72.dp)
            .testTag("daily_usage_mode_selector"),
        shape = RoundedCornerShape(30.dp),
        color = BaHomeControlRail,
        shadowElevation = 4.dp
    ) {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(6.dp)
                .selectableGroup(),
            horizontalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            HomeModeItem(
                mode = HomeAssistMode.DAILY,
                selected = selectedMode == HomeAssistMode.DAILY,
                language = language,
                icon = Icons.Outlined.LightMode,
                onClick = onDailyClick,
                modifier = Modifier.weight(1f)
            )
            HomeModeItem(
                mode = HomeAssistMode.QUIET,
                selected = selectedMode == HomeAssistMode.QUIET,
                language = language,
                icon = Icons.Outlined.Eco,
                onClick = onQuietClick,
                modifier = Modifier.weight(1f)
            )
            HomeModeItem(
                mode = HomeAssistMode.SENSITIVE,
                selected = selectedMode == HomeAssistMode.SENSITIVE,
                language = language,
                icon = Icons.Outlined.Bolt,
                onClick = onSensitiveClick,
                modifier = Modifier.weight(1f)
            )
        }
    }
}

@Composable
private fun HomeModeItem(
    mode: HomeAssistMode,
    selected: Boolean,
    language: AppLanguage,
    icon: ImageVector,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val background by animateColorAsState(
        targetValue = if (selected) BaHomeSurface else Color.Transparent,
        animationSpec = tween(durationMillis = 180),
        label = "home-mode-background"
    )
    val foreground by animateColorAsState(
        targetValue = if (selected) BaHomeGreen else BaHomeInk,
        animationSpec = tween(durationMillis = 180),
        label = "home-mode-foreground"
    )
    val elevation by animateDpAsState(
        targetValue = if (selected) 2.dp else 0.dp,
        animationSpec = tween(durationMillis = 180),
        label = "home-mode-elevation"
    )
    val shape = RoundedCornerShape(25.dp)
    val label = mode.label(language)

    Surface(
        modifier = modifier
            .fillMaxHeight()
            .clip(shape)
            .selectable(
                selected = selected,
                role = Role.RadioButton,
                onClick = onClick
            )
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Choose $label assist mode"
                } else {
                    "选择${label}辅助模式"
                }
            },
        shape = shape,
        color = background,
        shadowElevation = elevation
    ) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = foreground,
                modifier = Modifier.size(24.dp)
            )
            Spacer(Modifier.width(7.dp))
            Text(
                text = label,
                color = foreground,
                fontSize = 16.sp,
                lineHeight = 22.sp,
                fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                maxLines = 1
            )
        }
    }
}

@Composable
private fun HomeInfoRow(
    icon: ImageVector,
    iconTint: Color,
    title: String,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null,
    trailingIcon: ImageVector? = null
) {
    val interactionModifier = if (onClick == null) {
        Modifier.semantics(mergeDescendants = true) {}
    } else {
        Modifier
            .clickable(role = Role.Button, onClick = onClick)
            .semantics(mergeDescendants = true) {
                contentDescription = title
            }
    }

    Row(
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 72.dp)
            .then(interactionModifier)
            .padding(horizontal = 6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = iconTint,
            modifier = Modifier.size(30.dp)
        )
        Spacer(Modifier.width(20.dp))
        Text(
            text = title,
            color = BaHomeInk,
            fontSize = 16.sp,
            lineHeight = 24.sp,
            modifier = Modifier.weight(1f)
        )
        trailingIcon?.let {
            Icon(
                imageVector = it,
                contentDescription = null,
                tint = BaHomeInk,
                modifier = Modifier.size(26.dp)
            )
        }
    }
}

private fun feedbackSummary(
    controls: AssistControlsUiState,
    language: AppLanguage
): String {
    val mode = when (controls.alertProfile) {
        AlertProfile.QUIET -> if (language == AppLanguage.EN) "Quiet" else "安静提醒"
        AlertProfile.SENSITIVE -> if (language == AppLanguage.EN) "Sensitive" else "灵敏提醒"
        else -> if (language == AppLanguage.EN) "General daily" else "通用日常"
    }
    val channels = if (language == AppLanguage.EN) {
        when {
            controls.speechEnabled && controls.vibrationEnabled -> "Voice and vibration on"
            controls.speechEnabled -> "Voice on"
            controls.vibrationEnabled -> "Vibration on"
            else -> "Voice and vibration off"
        }
    } else {
        when {
            controls.speechEnabled && controls.vibrationEnabled -> "语音与震动已开启"
            controls.speechEnabled -> "语音已开启"
            controls.vibrationEnabled -> "震动已开启"
            else -> "语音与震动已关闭"
        }
    }
    return "$mode · $channels"
}

private fun glassesLabel(
    state: GlassesConnectionState,
    language: AppLanguage
): String {
    val status = if (language == AppLanguage.EN) {
        when (state) {
            GlassesConnectionState.DISCONNECTED -> "Disconnected"
            GlassesConnectionState.CONNECTING -> "Connecting"
            GlassesConnectionState.CONNECTED -> "Connected"
            GlassesConnectionState.CONNECTION_LOST -> "Connection lost"
        }
    } else {
        when (state) {
            GlassesConnectionState.DISCONNECTED -> "未连接"
            GlassesConnectionState.CONNECTING -> "连接中"
            GlassesConnectionState.CONNECTED -> "已连接"
            GlassesConnectionState.CONNECTION_LOST -> "连接已断开"
        }
    }
    return if (language == AppLanguage.EN) "Glasses · $status" else "眼镜设备 · $status"
}

private enum class HomeAssistMode {
    DAILY,
    QUIET,
    SENSITIVE;

    fun label(language: AppLanguage): String = when (this) {
        DAILY -> if (language == AppLanguage.EN) "Daily" else "日常"
        QUIET -> if (language == AppLanguage.EN) "Quiet" else "安静"
        SENSITIVE -> if (language == AppLanguage.EN) "Sensitive" else "灵敏"
    }
}

@Composable
internal fun InfoStrip(
    icon: ImageVector,
    title: String,
    body: String
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {},
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(containerColor = BaHomeSurface),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier.padding(18.dp),
            verticalAlignment = Alignment.Top
        ) {
            IconTile(icon = icon, accent = BaHomeAmber)
            Spacer(Modifier.width(14.dp))
            Column {
                Text(text = title, color = BaHomeInk, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(4.dp))
                Text(text = body, color = BaHomeTextMuted, style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}
