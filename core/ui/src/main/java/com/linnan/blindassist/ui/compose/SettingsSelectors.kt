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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
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
private fun SelectorSection(
    title: String,
    description: String,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
    ) {
        Text(
            text = title,
            color = BaHomeInk,
            style = MaterialTheme.typography.titleMedium,
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
    val railShape = RoundedCornerShape(30.dp)
    val largeFont = LocalDensity.current.fontScale >= 1.3f
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = railShape,
        color = BaHomeControlRail,
        shadowElevation = 4.dp
    ) {
        if (largeFont) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(5.dp)
                    .selectableGroup(),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                options.forEach { option ->
                    SegmentedSelectorOption(
                        option = option,
                        selected = selected == option,
                        optionLabel = optionLabel,
                        selectedStateDescription = selectedStateDescription,
                        unselectedStateDescription = unselectedStateDescription,
                        optionDescription = optionDescription,
                        onSelected = onSelected,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        } else {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(5.dp)
                    .selectableGroup(),
                horizontalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                options.forEach { option ->
                    SegmentedSelectorOption(
                        option = option,
                        selected = selected == option,
                        optionLabel = optionLabel,
                        selectedStateDescription = selectedStateDescription,
                        unselectedStateDescription = unselectedStateDescription,
                        optionDescription = optionDescription,
                        onSelected = onSelected,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}

@Composable
private fun <T> SegmentedSelectorOption(
    option: T,
    selected: Boolean,
    optionLabel: (T) -> String,
    selectedStateDescription: String,
    unselectedStateDescription: String,
    optionDescription: (T) -> String,
    onSelected: (T) -> Unit,
    modifier: Modifier = Modifier
) {
    val optionShape = RoundedCornerShape(25.dp)
    Surface(
        modifier = modifier
            .heightIn(min = 48.dp)
            .clip(optionShape)
            .selectable(
                selected = selected,
                role = Role.RadioButton,
                onClick = { onSelected(option) }
            )
            .semantics {
                role = Role.RadioButton
                stateDescription = if (selected) {
                    selectedStateDescription
                } else {
                    unselectedStateDescription
                }
                contentDescription = optionDescription(option)
            },
        shape = optionShape,
        color = if (selected) BaHomeSurface else Color.Transparent,
        shadowElevation = if (selected) 2.dp else 0.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 48.dp)
                .padding(horizontal = 8.dp, vertical = 8.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = optionLabel(option),
                color = if (selected) BaHomeGreen else BaHomeInk,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                textAlign = TextAlign.Center
            )
        }
    }
}

@Composable
private fun ScenarioRow(
    scenario: AssistScenario,
    selected: Boolean,
    language: AppLanguage,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp)
            .heightIn(min = 64.dp)
            .clip(RoundedCornerShape(18.dp))
            .background(if (selected) BaHomeNavIndicator else Color.Transparent)
            .selectable(
                selected = selected,
                role = Role.RadioButton,
                onClick = onClick
            )
            .semantics {
                role = Role.RadioButton
                stateDescription = if (selected) {
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
            .padding(horizontal = 4.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .width(3.dp)
                .height(34.dp)
                .background(if (selected) BaHomeGreen else Color.Transparent)
        )
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(
                text = scenario.displayName(language),
                color = if (selected) BaHomeGreen else BaHomeInk,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = if (selected) FontWeight.Bold else FontWeight.SemiBold,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
            Spacer(Modifier.height(2.dp))
            Text(
                text = scenario.description(language),
                color = BaHomeTextMuted,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun ScenarioDivider() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(1.dp)
            .background(BaHomeHairline.copy(alpha = 0.72f))
    )
}

@Composable
private fun ScenarioRows(
    selected: AssistScenario,
    language: AppLanguage,
    onScenarioChange: (AssistScenario) -> Unit
) {
    val scenarios = AssistScenario.values()
    Column(modifier = Modifier.selectableGroup()) {
        scenarios.forEachIndexed { index, scenario ->
            ScenarioRow(
                scenario = scenario,
                selected = selected == scenario,
                language = language,
                onClick = { onScenarioChange(scenario) }
            )
            if (index != scenarios.lastIndex) {
                ScenarioDivider()
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
    SelectorSection(
        title = if (english) "Reminder profile" else "提醒档位",
        description = if (english) {
            "Quiet reduces interruption, Sensitive confirms medium risk earlier."
        } else {
            "安静减少打扰，敏感更早确认中风险。"
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
    SelectorSection(
        title = if (english) "Interface language" else "界面语言",
        description = if (english) {
            "Choose Chinese or English for core reminders and settings."
        } else {
            "选择核心提醒和设置界面的中文或英文。"
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
    SelectorSection(
        title = if (english) "Usage scenario" else "使用场景",
        description = if (english) {
            "Manually choose the walking environment to tune confirmation, cooldown, and vibration."
        } else {
            "手动选择行走环境，调整提醒确认、冷却和震动计划。"
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
    SelectorSection(
        title = if (english) "Speech style" else "语音风格",
        description = if (english) {
            "Brief reduces interruption, Detailed adds object type."
        } else {
            "简短减少打扰，详细会补充目标类别。"
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
    SelectorSection(
        title = if (english) "Vibration strength" else "震动强度",
        description = if (english) {
            "Choose soft, standard, or stronger feedback for tactile sensitivity."
        } else {
            "按触觉敏感度选择轻柔、标准或更强提醒。"
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
