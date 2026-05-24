package com.linnan.blindassist.ui.compose

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun ProfileSelector(
    selected: AlertProfile,
    language: AppLanguage,
    onProfileChange: (AlertProfile) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Reminder profile, current ${selected.displayName(language)}. Quiet reduces interruption, Sensitive confirms medium risk earlier."
                } else {
                    "提醒档位，当前${selected.displayName(language)}。安静减少打扰，敏感更早确认中风险。"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(if (language == AppLanguage.EN) "Reminder profile" else "提醒档位", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text(if (language == AppLanguage.EN) "Quiet reduces interruption, Sensitive confirms medium risk earlier." else "安静减少打扰，敏感更早确认中风险。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                AlertProfile.values().forEach { profile ->
                    FilterChip(
                        selected = selected == profile,
                        onClick = { onProfileChange(profile) },
                        label = { Text(profile.displayName(language)) },
                        modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).semantics {
                            role = Role.Button
                            stateDescription = if (selected == profile) {
                                if (language == AppLanguage.EN) "Current profile" else "当前档位"
                            } else {
                                if (language == AppLanguage.EN) "Not selected" else "未选择"
                            }
                            contentDescription = if (language == AppLanguage.EN) {
                                "Choose ${profile.displayName(language)} reminder profile"
                            } else {
                                "选择${profile.displayName(language)}提醒档位"
                            }
                        }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun LanguageSelector(
    selected: AppLanguage,
    onLanguageChange: (AppLanguage) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("language_selector")
            .semantics {
                contentDescription = if (selected == AppLanguage.EN) {
                    "Interface language, current English"
                } else {
                    "界面语言，当前中文"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(if (selected == AppLanguage.EN) "Interface language" else "界面语言", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text(if (selected == AppLanguage.EN) "Choose Chinese or English for core reminders and settings." else "选择核心提醒和设置界面的中文或英文。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                AppLanguage.values().forEach { language ->
                    val isEnglishUi = selected == AppLanguage.EN
                    val name = language.displayName(selected)
                    FilterChip(
                        selected = selected == language,
                        onClick = { onLanguageChange(language) },
                        label = { Text(name) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 48.dp)
                            .semantics {
                                role = Role.Button
                                stateDescription = if (selected == language) {
                                    if (isEnglishUi) "Current language" else "当前语言"
                                } else {
                                    if (isEnglishUi) "Not selected" else "未选择"
                                }
                                contentDescription = if (isEnglishUi) {
                                    "Choose $name interface language"
                                } else {
                                    "选择$name 界面语言"
                                }
                            }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun ScenarioSelector(
    selected: AssistScenario,
    language: AppLanguage,
    onScenarioChange: (AssistScenario) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("scenario_selector")
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Usage scenario, current ${selected.displayName(language)}. ${selected.description(language)}"
                } else {
                    "使用场景，当前${selected.displayName(language)}。${selected.description(language)}"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(if (language == AppLanguage.EN) "Usage scenario" else "使用场景", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text(if (language == AppLanguage.EN) "Manually choose the walking environment to tune confirmation, cooldown, and vibration." else "手动选择行走环境，调整提醒确认、冷却和震动计划。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            AssistScenario.values().forEach { scenario ->
                FilterChip(
                    selected = selected == scenario,
                    onClick = { onScenarioChange(scenario) },
                    label = {
                        Text(
                            "${scenario.displayName(language)} · ${scenario.description(language)}",
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp)
                        .semantics {
                            role = Role.Button
                            stateDescription = if (selected == scenario) {
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
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun SpeechStyleSelector(
    selected: SpeechStyle,
    language: AppLanguage,
    onSpeechStyleChange: (SpeechStyle) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Speech style, current ${selected.displayName(language)}. ${selected.description(language)}"
                } else {
                    "语音风格，当前${selected.displayName(language)}。${selected.description(language)}"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(if (language == AppLanguage.EN) "Speech style" else "语音风格", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text(if (language == AppLanguage.EN) "Brief reduces interruption, Detailed adds object type." else "简短减少打扰，详细会补充目标类别。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SpeechStyle.values().forEach { style ->
                    FilterChip(
                        selected = selected == style,
                        onClick = { onSpeechStyleChange(style) },
                        label = { Text(style.displayName(language), maxLines = 1, overflow = TextOverflow.Ellipsis) },
                        modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).semantics {
                            role = Role.Button
                            stateDescription = if (selected == style) {
                                if (language == AppLanguage.EN) "Current style" else "当前风格"
                            } else {
                                if (language == AppLanguage.EN) "Not selected" else "未选择"
                            }
                            contentDescription = if (language == AppLanguage.EN) {
                                "Choose ${style.displayName(language)} speech style, ${style.description(language)}"
                            } else {
                                "选择${style.displayName(language)}语音风格，${style.description(language)}"
                            }
                        }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun VibrationStrengthSelector(
    selected: VibrationStrength,
    language: AppLanguage,
    onVibrationStrengthChange: (VibrationStrength) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = if (language == AppLanguage.EN) {
                    "Vibration strength, current ${selected.displayName(language)}. ${selected.description(language)}"
                } else {
                    "震动强度，当前${selected.displayName(language)}。${selected.description(language)}"
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BaPanel)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(if (language == AppLanguage.EN) "Vibration strength" else "震动强度", color = BaText, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
            Text(if (language == AppLanguage.EN) "Choose soft, standard, or stronger feedback for tactile sensitivity." else "按触觉敏感度选择轻柔、标准或更强提醒。", color = BaTextMuted, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                VibrationStrength.values().forEach { strength ->
                    FilterChip(
                        selected = selected == strength,
                        onClick = { onVibrationStrengthChange(strength) },
                        label = { Text(strength.displayName(language), maxLines = 1, overflow = TextOverflow.Ellipsis) },
                        modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).semantics {
                            role = Role.Button
                            stateDescription = if (selected == strength) {
                                if (language == AppLanguage.EN) "Current strength" else "当前强度"
                            } else {
                                if (language == AppLanguage.EN) "Not selected" else "未选择"
                            }
                            contentDescription = if (language == AppLanguage.EN) {
                                "Choose ${strength.displayName(language)} vibration strength, ${strength.description(language)}"
                            } else {
                                "选择${strength.displayName(language)}震动强度，${strength.description(language)}"
                            }
                        }
                    )
                }
            }
        }
    }
}
