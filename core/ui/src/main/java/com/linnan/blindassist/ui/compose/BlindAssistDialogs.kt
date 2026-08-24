package com.linnan.blindassist.ui.compose

import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.Shield
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.linnan.blindassist.localization.AppLanguage

@Composable
fun CameraPermissionExplanationDialog(
    language: AppLanguage = AppLanguage.ZH,
    onContinue: () -> Unit,
    onDismiss: () -> Unit
) {
    val english = language == AppLanguage.EN
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            Button(onClick = onContinue, modifier = Modifier.heightIn(min = 48.dp)) {
                Text(if (english) "Continue and allow" else "继续并授权")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) {
                Text(if (english) "Not now" else "暂不打开")
            }
        },
        icon = { Icon(Icons.Outlined.CameraAlt, contentDescription = null, tint = BaHomeGreen) },
        title = { Text(if (english) "Camera permission needed" else "需要相机权限") },
        text = {
            Text(
                if (english) {
                    "The camera is only used for live on-device recognition. BlindAssist does not upload images, does not connect to the network, and does not save video. Speech and vibration reminders are assistive references only and cannot replace a cane, guide dog, or human judgment."
                } else {
                    "相机仅用于手机端实时识别。BlindAssist 不上传画面、不联网、不保存视频；语音和震动提醒只作为辅助参考，不能替代盲杖、导盲犬或人工判断。"
                }
            )
        },
        shape = RoundedCornerShape(28.dp),
        containerColor = BaHomeSurface,
        iconContentColor = BaHomeGreen,
        titleContentColor = BaHomeInk,
        textContentColor = BaHomeTextMuted
    )
}

@Composable
fun CameraPermissionDeniedDialog(
    language: AppLanguage = AppLanguage.ZH,
    onDismiss: () -> Unit
) {
    val english = language == AppLanguage.EN
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) {
                Text(if (english) "Got it" else "知道了")
            }
        },
        icon = { Icon(Icons.Outlined.Shield, contentDescription = null, tint = BaHomeAmber) },
        title = { Text(if (english) "Camera permission is off" else "相机权限未开启") },
        text = {
            Text(
                if (english) {
                    "Phone camera assistance cannot start without camera permission. You can stay on the main screen, review settings, and tap Start assistance again later to allow permission."
                } else {
                    "未获得相机权限时，手机摄像头辅助无法启动。你仍可留在主界面查看设置，稍后再次点击“开始辅助”重新授权。"
                }
            )
        },
        shape = RoundedCornerShape(28.dp),
        containerColor = BaHomeSurface,
        iconContentColor = BaHomeAmber,
        titleContentColor = BaHomeInk,
        textContentColor = BaHomeTextMuted
    )
}
