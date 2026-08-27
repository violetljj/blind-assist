package com.linnan.blindassist.ui.compose

import androidx.compose.foundation.BorderStroke
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
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage

@Composable
private fun SelectorCard(
    title: String,
    description: String,
    accessibilityDescription: String,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics { contentDescription = accessibilityDescription },
        shape = BaShapeCard,
        colors = CardDefaults.cardColors(containerColor = BaHomeSurface),
        border = BorderStroke(1.dp, BaHomeHairline),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(Modifier.padding(horizontal = 16.dp, vertical = 15.dp)) {
            Text(
                text = title,
                color = BaHomeInk,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.semantics { heading() }
            )
            Text(
                text = description,
                color = BaHomeTextMuted,
                style = MaterialTheme.typography.bodySmall
            )
            Spacer(Modifier.height(12.dp))
            content()
        }
    }
}

@Composable
private fun <T> SegmentedSelector(
    options: List<T>,
    selected: T,
    optionLabel: (T) -> String,
    selectedStateDescription: String,
    unselectedStateDescription: String,
    optionDescription: (T) -> String,
    onSelected: (T) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(BaShapeControl)
            .background(BaHomeControlRail)
            .padding(3.dp)
            .selectableGroup(),
        horizontalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        options.forEach { option ->
            val isSelected = selected == option
            Box(
                modifier = Modifier
                    .weight(1f)
                    .heightIn(min = 48.dp)
                    .clip(BaShapeCompact)
                    .background(if (isSelected) BaHomeNavIndicator else Color.Transparent)
                    .selectable(
                        selected = isSelected,
                        role = Role.RadioButton,
                        onClick = { onSelected(option) }
                    )
                    .semantics {
                        role = Role.RadioButton
                        stateDescription = if (isSelected) {
                            selectedStateDescription
                        } else {
                            unselectedStateDescription
                        }
                        contentDescription = optionDescription(option)
                    }
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = optionLabel(option),
                    color = if (isSelected) BaHomeGreen else BaHomeInk,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

@Composable
private fun ScenarioRows(
    selected: AssistScenario,
    language: AppLanguage,
    onScenarioChange: (AssistScenario) -> Unit
) {
    Column(
        modifier = Modifier.selectableGroup(),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        AssistScenario.values().forEach { scenario ->
            val isSelected = selected == scenario
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 60.dp)
                    .clip(BaShapeCompact)
                    .background(if (isSelected) BaHomeNavIndicator else BaHomeControlRail)
                    .selectable(
                        selected = isSelected,
                        role = Role.RadioButton,
                        onClick = { onScenarioChange(scenario) }
                    )
                    .semantics {
                        role = Role.RadioButton
                        stateDescription = if (isSelected) {
                            if (language == AppLanguage.EN) "Current scenario" else "当前场景"
                        } else {
                            if (language == AppLanguage.EN) "Not selected" else "未选择"
                        }
                        contentDescription = if (language == AppLanguage.EN) {
                            "Choose ${scenario.displayName(language)} usage scenario, ${scenario.description(language)}"
                        } else {
                            "选择${scenario.displayName(language)}使用场景，${scenario.description(language)}"
                        }
                    }
                    .padding(horizontal = 14.dp, vertical = 10.dp)
            ) {
                Text(
                    text = scenario.displayName(language),
                    color = if (isSelected) BaHomeGreen else BaHomeInk,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = scenario.description(language),
                    color = BaHomeTextMuted,
                    style = MaterialTheme.typography.bodySmall,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
internal fun ProfileSelector(
    selected: AlertProfile,
    language: AppLanguage,
    onProfileChange: (AlertProfile) -> Unit
) {
    val english = language == AppLanguage.EN
    SelectorCard(
        title = if (english) "Reminder profile" else "提醒档位",
        description = if (english) {
            "Quiet reduces interruption, Sensitive confirms medium risk earlier."
        } else {
            "安静减少打扰，敏感更早确认中风险。"
        },
        accessibilityDescription = if (english) {
            "Reminder profile, current ${selected.displayName(language)}. Quiet reduces interruption, Sensitive confirms medium risk earlier."
        } else {
            "提醒档位，当前${selected.displayName(language)}。安静减少打扰，敏感更早确认中风险。"
        }
    ) {
        SegmentedSelector(
            options = AlertProfile.values().asList(),
            selected = selected,
            optionLabel = { it.displayName(language) },
            selectedStateDescription = if (english) "Current profile" else "当前档位",
            unselectedStateDescription = if (english) "Not selected" else "未选择",
            optionDescription = {
                if (english) {
                    "Choose ${it.displayName(language)} reminder profile"
                } else {
                    "选择${it.displayName(language)}提醒档位"
                }
            },
            onSelected = onProfileChange
        )
    }
}

@Composable
internal fun LanguageSelector(
    selected: AppLanguage,
    onLanguageChange: (AppLanguage) -> Unit
) {
    val english = selected == AppLanguage.EN
    SelectorCard(
        title = if (english) "Interface language" else "界面语言",
        description = if (english) {
            "Choose Chinese or English for core reminders and settings."
        } else {
            "选择核心提醒和设置界面的中文或英文。"
        },
        accessibilityDescription = if (english) {
            "Interface language, current English"
        } else {
            "界面语言，当前中文"
        },
        modifier = Modifier.testTag("language_selector")
    ) {
        SegmentedSelector(
            options = AppLanguage.values().asList(),
            selected = selected,
            optionLabel = { it.displayName(selected) },
            selectedStateDescription = if (english) "Current language" else "当前语言",
            unselectedStateDescription = if (english) "Not selected" else "未选择",
            optionDescription = {
                val name = it.displayName(selected)
                if (english) "Choose $name interface language" else "选择$name 界面语言"
            },
            onSelected = onLanguageChange
        )
    }
}

@Composable
internal fun ScenarioSelector(
    selected: AssistScenario,
    language: AppLanguage,
    onScenarioChange: (AssistScenario) -> Unit
) {
    val english = language == AppLanguage.EN
    SelectorCard(
        title = if (english) "Usage scenario" else "使用场景",
        description = if (english) {
            "Manually choose the walking environment to tune confirmation, cooldown, and vibration."
        } else {
            "手动选择行走环境，调整提醒确认、冷却和震动计划。"
        },
        accessibilityDescription = if (english) {
            "Usage scenario, current ${selected.displayName(language)}. ${selected.description(language)}"
        } else {
            "使用场景，当前${selected.displayName(language)}。${selected.description(language)}"
        },
        modifier = Modifier.testTag("scenario_selector")
    ) {
        ScenarioRows(
            selected = selected,
            language = language,
            onScenarioChange = onScenarioChange
        )
    }
}

@Composable
internal fun SpeechStyleSelector(
    selected: SpeechStyle,
    language: AppLanguage,
    onSpeechStyleChange: (SpeechStyle) -> Unit
) {
    val english = language == AppLanguage.EN
    SelectorCard(
        title = if (english) "Speech style" else "语音风格",
        description = if (english) {
            "Brief reduces interruption, Detailed adds object type."
        } else {
            "简短减少打扰，详细会补充目标类别。"
        },
        accessibilityDescription = if (english) {
            "Speech style, current ${selected.displayName(language)}. ${selected.description(language)}"
        } else {
            "语音风格，当前${selected.displayName(language)}。${selected.description(language)}"
        }
    ) {
        SegmentedSelector(
            options = SpeechStyle.values().asList(),
            selected = selected,
            optionLabel = { it.displayName(language) },
            selectedStateDescription = if (english) "Current style" else "当前风格",
            unselectedStateDescription = if (english) "Not selected" else "未选择",
            optionDescription = {
                if (english) {
                    "Choose ${it.displayName(language)} speech style, ${it.description(language)}"
                } else {
                    "选择${it.displayName(language)}语音风格，${it.description(language)}"
                }
            },
            onSelected = onSpeechStyleChange
        )
    }
}

@Composable
internal fun VibrationStrengthSelector(
    selected: VibrationStrength,
    language: AppLanguage,
    onVibrationStrengthChange: (VibrationStrength) -> Unit
) {
    val english = language == AppLanguage.EN
    SelectorCard(
        title = if (english) "Vibration strength" else "震动强度",
        description = if (english) {
            "Choose soft, standard, or stronger feedback for tactile sensitivity."
        } else {
            "按触觉敏感度选择轻柔、标准或更强提醒。"
        },
        accessibilityDescription = if (english) {
            "Vibration strength, current ${selected.displayName(language)}. ${selected.description(language)}"
        } else {
            "震动强度，当前${selected.displayName(language)}。${selected.description(language)}"
        }
    ) {
        SegmentedSelector(
            options = VibrationStrength.values().asList(),
            selected = selected,
            optionLabel = { it.displayName(language) },
            selectedStateDescription = if (english) "Current strength" else "当前强度",
            unselectedStateDescription = if (english) "Not selected" else "未选择",
            optionDescription = {
                if (english) {
                    "Choose ${it.displayName(language)} vibration strength, ${it.description(language)}"
                } else {
                    "选择${it.displayName(language)}震动强度，${it.description(language)}"
                }
            },
            onSelected = onVibrationStrengthChange
        )
    }
}
