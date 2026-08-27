package com.linnan.blindassist.ui.compose

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.Shield
import androidx.compose.material.icons.outlined.Vibration
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp


@Composable
fun OnboardingScreen(
    onFinished: () -> Unit,
    modifier: Modifier = Modifier
) {
    var pageIndex by rememberSaveable { mutableStateOf(0) }
    val pages = onboardingPages()
    val page = pages[pageIndex]
    val isLastPage = pageIndex == pages.lastIndex

    ScreenColumn(modifier = modifier.statusBarsPadding()) {
        ScreenIntro(
            eyebrow = "BLINDASSIST ${pageIndex + 1} / ${pages.size}",
            title = "开始使用 BlindAssist",
            body = "先了解三件事，再进入本地视觉辅助体验。"
        )
        Spacer(Modifier.height(38.dp))

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 304.dp)
                .padding(horizontal = 4.dp),
            horizontalAlignment = Alignment.Start
        ) {
            Box(
                modifier = Modifier
                    .size(50.dp)
                    .clip(CircleShape)
                    .background(page.accent.copy(alpha = 0.10f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = page.icon,
                    contentDescription = null,
                    tint = page.accent,
                    modifier = Modifier.size(25.dp)
                )
            }
            Spacer(Modifier.height(28.dp))
            Text(
                text = page.title,
                style = MaterialTheme.typography.headlineSmall,
                color = BaHomeInk,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.semantics { heading() }
            )
            Spacer(Modifier.height(14.dp))
            Text(
                text = page.body,
                style = MaterialTheme.typography.bodyLarge,
                color = BaHomeInk,
                lineHeight = MaterialTheme.typography.bodyLarge.lineHeight
            )
            Spacer(Modifier.height(22.dp))
            HorizontalDivider(
                modifier = Modifier.fillMaxWidth(0.18f),
                thickness = 2.dp,
                color = page.accent.copy(alpha = 0.62f)
            )
            Spacer(Modifier.height(12.dp))
            Text(
                text = page.detail,
                style = MaterialTheme.typography.bodyMedium,
                color = BaHomeTextMuted
            )
        }

        Spacer(Modifier.height(24.dp))
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            pages.forEachIndexed { index, _ ->
                Box(
                    modifier = Modifier
                        .height(3.dp)
                        .weight(1f)
                        .clip(RoundedCornerShape(50))
                        .background(if (index == pageIndex) BaHomeGreen else BaHomeHairline)
                )
            }
        }
        Spacer(Modifier.height(24.dp))
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
                .heightIn(min = 60.dp),
            shape = RoundedCornerShape(22.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = BaHomeActionEnd,
                contentColor = BaHomeOnAction
            )
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
        icon = Icons.Outlined.CameraAlt,
        accent = BaHomeGreen
    ),
    OnboardingPage(
        title = "通过语音和震动给出辅助提醒",
        body = "当规则层判断近处或迫近风险时，系统会用短句语音和震动帮助你注意前方变化。",
        detail = "你可以在设置页调整语音、震动、关怀模式和提醒档位。",
        icon = Icons.Outlined.Vibration,
        accent = BaHomeCobalt
    ),
    OnboardingPage(
        title = "不能替代盲杖、导盲犬或人工判断",
        body = "BlindAssist 是助盲避障原型，提醒可能受光照、遮挡、设备性能和模型识别结果影响。",
        detail = "行走时请继续保留人工判断和专业辅助方式，把 App 提醒作为额外参考。",
        icon = Icons.Outlined.Shield,
        accent = BaHomeAmber
    )
)
