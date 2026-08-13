package com.linnan.blindassist.ui.compose

import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.activity.compose.BackHandler
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Bluetooth
import androidx.compose.material.icons.rounded.BugReport
import androidx.compose.material.icons.rounded.CameraAlt
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.Home
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material.icons.rounded.PhoneAndroid
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Shield
import androidx.compose.material.icons.rounded.Tune
import androidx.compose.material.icons.rounded.Vibration
import androidx.compose.material.icons.rounded.Visibility
import androidx.compose.material.icons.rounded.VolumeUp
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.ui.DetectionOverlayView
import kotlinx.coroutines.delay


@Composable
fun ProfileScreen(
    controls: AssistControlsUiState,
    appVersion: String,
    modifier: Modifier = Modifier
) {
    val language = controls.appLanguage
    ScreenColumn(modifier = modifier) {
        ScreenIntro(
            eyebrow = if (language == AppLanguage.EN) "LOCAL PROFILE" else "本地档案",
            title = if (language == AppLanguage.EN) "Your assist profile" else "你的辅助档案",
            body = if (language == AppLanguage.EN) {
                "A quick view of this device, reminder preferences, and current walking setup."
            } else {
                "集中查看本机能力、提醒偏好与当前行走设置。"
            }
        )
        Spacer(Modifier.height(22.dp))
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Box(
                modifier = Modifier
                    .size(58.dp)
                    .clip(RoundedCornerShape(18.dp))
                    .background(BaMintWash),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Rounded.Person, contentDescription = null, tint = BaMint, modifier = Modifier.size(26.dp))
            }
            Spacer(Modifier.width(16.dp))
            Column {
                Text(
                    text = if (language == AppLanguage.EN) "BlindAssist user" else "BlindAssist 用户",
                    style = MaterialTheme.typography.titleMedium,
                    color = BaText,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.semantics { heading() }
                )
                Text(
                    text = if (language == AppLanguage.EN) {
                        "Local prototype mode - no account system"
                    } else {
                        "本地原型模式 · 未接入账号系统"
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    color = BaTextMuted
                )
            }
        }
        Spacer(Modifier.height(24.dp))
        Text(
            text = if (language == AppLanguage.EN) "STATUS AT A GLANCE" else "状态概览",
            color = BaMint,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold
        )
        Spacer(Modifier.height(10.dp))
        StatusGrid(
            leftTitle = if (language == AppLanguage.EN) "Device" else "设备",
            leftBody = if (language == AppLanguage.EN) "Phone camera available" else "手机摄像头可用",
            rightTitle = if (language == AppLanguage.EN) "Glasses" else "眼镜",
            rightBody = if (language == AppLanguage.EN) "Reserved for future extension" else "等待未来扩展"
        )
        Spacer(Modifier.height(12.dp))
        StatusGrid(
            leftTitle = if (language == AppLanguage.EN) "Reminder profile" else "提醒档位",
            leftBody = controls.alertProfile.displayName(language),
            rightTitle = if (language == AppLanguage.EN) "Usage scenario" else "使用场景",
            rightBody = controls.assistScenario.displayName(language)
        )
        Spacer(Modifier.height(12.dp))
        StatusGrid(
            leftTitle = if (language == AppLanguage.EN) "Current version" else "当前版本",
            leftBody = "v$appVersion",
            rightTitle = if (language == AppLanguage.EN) "Reminder explanations" else "提醒解释",
            rightBody = if (language == AppLanguage.EN) "On" else "已开启"
        )
        Spacer(Modifier.height(12.dp))
        InfoStrip(
            icon = Icons.Rounded.Favorite,
            title = if (language == AppLanguage.EN) "Assist preferences" else "辅助偏好",
            body = if (language == AppLanguage.EN) {
                "Speech ${enabledText(controls.speechEnabled, language)} (${controls.speechStyle.displayName(language)}), vibration ${enabledText(controls.vibrationEnabled, language)} (${controls.vibrationStrength.displayName(language)}), Care Mode ${enabledText(controls.careModeEnabled, language)}."
            } else {
                "语音${enabledText(controls.speechEnabled, language)}（${controls.speechStyle.displayName(language)}），震动${enabledText(controls.vibrationEnabled, language)}（${controls.vibrationStrength.displayName(language)}），关怀模式${enabledText(controls.careModeEnabled, language)}。"
            }
        )
    }
}

