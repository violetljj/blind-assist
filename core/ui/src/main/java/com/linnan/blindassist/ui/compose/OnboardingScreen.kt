package com.linnan.blindassist.ui.compose

import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.activity.compose.BackHandler
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.background
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
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


@Composable
fun OnboardingScreen(
    onFinished: () -> Unit,
    modifier: Modifier = Modifier
) {
    var pageIndex by rememberSaveable { mutableStateOf(0) }
    val pages = onboardingPages()
    val page = pages[pageIndex]
    val isLastPage = pageIndex == pages.lastIndex

    ScreenColumn(modifier = modifier) {
        ScreenIntro(
            eyebrow = "BLINDASSIST ${pageIndex + 1} / ${pages.size}",
            title = "开始使用 BlindAssist",
            body = "先了解三件事，再进入本地视觉辅助体验。"
        )
        Spacer(Modifier.height(26.dp))

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 330.dp),
            shape = RoundedCornerShape(28.dp),
            colors = CardDefaults.cardColors(containerColor = BaPanel)
        ) {
            Column(
                modifier = Modifier.padding(22.dp),
                horizontalAlignment = Alignment.Start
            ) {
                IconTile(icon = page.icon, accent = page.accent, emphasized = true)
                Spacer(Modifier.height(24.dp))
                Text(
                    text = page.title,
                    style = MaterialTheme.typography.headlineSmall,
                    color = BaText,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.semantics { heading() }
                )
                Spacer(Modifier.height(12.dp))
                Text(
                    text = page.body,
                    style = MaterialTheme.typography.bodyLarge,
                    color = BaText,
                    lineHeight = MaterialTheme.typography.bodyLarge.lineHeight
                )
                Spacer(Modifier.height(14.dp))
                Text(
                    text = page.detail,
                    style = MaterialTheme.typography.bodyMedium,
                    color = BaTextMuted
                )
            }
        }

        Spacer(Modifier.height(20.dp))
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            pages.forEachIndexed { index, _ ->
                Box(
                    modifier = Modifier
                        .height(5.dp)
                        .weight(1f)
                        .clip(RoundedCornerShape(50))
                        .background(if (index == pageIndex) BaMint else BaPanelSoft)
                )
            }
        }
        Spacer(Modifier.height(22.dp))
        Button(
            onClick = {
                if (isLastPage) {
                    onFinished()
                } else {
                    pageIndex += 1
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 56.dp),
            shape = RoundedCornerShape(18.dp),
            colors = ButtonDefaults.buttonColors(containerColor = BaMint, contentColor = BaInk)
        ) {
            Text(if (isLastPage) "开始使用" else "下一步")
        }
        if (!isLastPage) {
            TextButton(
                onClick = onFinished,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
            ) {
                Text("跳过引导")
            }
        }
    }
}


private data class OnboardingPage(
    val title: String,
    val body: String,
    val detail: String,
    val icon: ImageVector,
    val accent: Color
)

private fun onboardingPages(): List<OnboardingPage> = listOf(
    OnboardingPage(
        title = "使用手机摄像头进行本地识别",
        body = "打开手机摄像头后，App 会在本机运行 YOLO11n 模型，识别画面中的常见目标和相对方向。",
        detail = "画面不上传、不联网、不保存视频，当前版本优先支持手机摄像头。",
        icon = Icons.Rounded.CameraAlt,
        accent = BaMint
    ),
    OnboardingPage(
        title = "通过语音和震动给出辅助提醒",
        body = "当规则层判断近处或迫近风险时，系统会用短句语音和震动帮助你注意前方变化。",
        detail = "你可以在设置页调整语音、震动、关怀模式和提醒档位。",
        icon = Icons.Rounded.Vibration,
        accent = BaSky
    ),
    OnboardingPage(
        title = "不能替代盲杖、导盲犬或人工判断",
        body = "BlindAssist 是助盲避障原型，提醒可能受光照、遮挡、设备性能和模型识别结果影响。",
        detail = "行走时请继续保留人工判断和专业辅助方式，把 App 提醒作为额外参考。",
        icon = Icons.Rounded.Shield,
        accent = BaAmber
    )
)
