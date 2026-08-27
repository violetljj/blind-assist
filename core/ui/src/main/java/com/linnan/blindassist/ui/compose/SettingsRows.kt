package com.linnan.blindassist.ui.compose

import androidx.compose.foundation.clickable
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText

@Composable
internal fun SettingsGroup(
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(22.dp),
        color = BaHomeSurface.copy(alpha = 0.86f),
        shadowElevation = 1.dp
    ) {
        Column(content = content)
    }
}

@Composable
internal fun SettingsDivider(startIndent: Int = 74) {
    HorizontalDivider(
        modifier = Modifier.padding(start = startIndent.dp, end = 18.dp),
        color = BaHomeHairline.copy(alpha = 0.72f)
    )
}

@Composable
internal fun SettingSwitchRow(
    icon: ImageVector,
    title: String,
    body: String,
    checked: Boolean,
    language: AppLanguage = AppLanguage.ZH,
    modifier: Modifier = Modifier,
    onCheckedChange: (Boolean) -> Unit
) {
    val stateText = LocalizedText.enabled(checked, language)
    Row(
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 84.dp)
            .toggleable(
                value = checked,
                role = Role.Switch,
                onValueChange = onCheckedChange
            )
            .semantics(mergeDescendants = true) {
                stateDescription = stateText
                contentDescription = if (language == AppLanguage.EN) {
                    "$title, $body, currently $stateText"
                } else {
                    "$title，$body，当前$stateText"
                }
            }
            .padding(horizontal = 18.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        IconTile(
            icon = icon,
            accent = if (checked) BaHomeGreen else BaHomeTextMuted
        )
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(title, color = BaHomeInk, fontWeight = FontWeight.Bold)
            Text(body, color = BaHomeTextMuted, style = MaterialTheme.typography.bodySmall)
        }
        Switch(
            checked = checked,
            onCheckedChange = null,
            modifier = Modifier.clearAndSetSemantics { }
        )
    }
}

@Composable
internal fun SettingsActionRow(
    icon: ImageVector,
    title: String,
    body: String,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 82.dp)
            .clickable(role = Role.Button, onClick = onClick)
            .semantics(mergeDescendants = true) {}
            .padding(horizontal = 18.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        IconTile(icon = icon, accent = BaHomeGreen)
        Spacer(Modifier.width(14.dp))
        Column(Modifier.weight(1f)) {
            Text(title, color = BaHomeInk, fontWeight = FontWeight.Bold)
            Text(body, color = BaHomeTextMuted, style = MaterialTheme.typography.bodySmall)
        }
        Icon(Icons.Rounded.ChevronRight, contentDescription = null, tint = BaHomeGreen.copy(alpha = 0.82f))
    }
}
