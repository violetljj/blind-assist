package com.linnan.blindassist.ui.compose

import androidx.compose.foundation.layout.heightIn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Bluetooth
import androidx.compose.material.icons.rounded.CameraAlt
import androidx.compose.material.icons.rounded.Shield
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun GlassesPlaceholderDialog(
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("知道了")
            }
        },
        icon = { Icon(Icons.Rounded.Bluetooth, contentDescription = null, tint = BaSky) },
        title = { Text("眼镜设备连接") },
        text = {
            Text("该入口为未来蓝牙眼镜或外接视觉设备预留。当前版本不会扫描蓝牙、不会联网，也不会申请额外权限。")
        }
    )
}

@Composable
fun CameraPermissionExplanationDialog(
    onContinue: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            Button(onClick = onContinue, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("继续并授权")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("暂不打开")
            }
        },
        icon = { Icon(Icons.Rounded.CameraAlt, contentDescription = null, tint = BaMint) },
        title = { Text("需要相机权限") },
        text = {
            Text("相机仅用于手机端实时识别。BlindAssist 不上传画面、不联网、不保存视频；语音和震动提醒只作为辅助参考，不能替代盲杖、导盲犬或人工判断。")
        }
    )
}

@Composable
fun CameraPermissionDeniedDialog(
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("知道了")
            }
        },
        icon = { Icon(Icons.Rounded.Shield, contentDescription = null, tint = BaAmber) },
        title = { Text("相机权限未开启") },
        text = {
            Text("未获得相机权限时，手机摄像头辅助无法启动。你仍可留在主界面查看设置，稍后再次点击“使用手机摄像头”重新授权。")
        }
    )
}
