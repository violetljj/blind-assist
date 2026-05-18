package com.linnan.blindassist.ui.compose

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BlindAssistComposeTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<ComposeTestActivity>()

    @Test
    fun settingsScreenChangesFeedbackDetailControls() {
        var controls by mutableStateOf(defaultControls())

        composeRule.setContent {
            BlindAssistTheme {
                SettingsScreen(
                    controls = controls,
                    fieldTestSummary = FieldTestSummaryUiState.empty(controls.alertProfile.displayName),
                    onSpeechChange = { controls = controls.copy(speechEnabled = it) },
                    onVibrationChange = { controls = controls.copy(vibrationEnabled = it) },
                    onCareModeChange = { controls = controls.copy(careModeEnabled = it) },
                    onDebugVisibleChange = { controls = controls.copy(debugVisible = it) },
                    onProfileChange = { controls = controls.copy(alertProfile = it) },
                    onSpeechStyleChange = { controls = controls.copy(speechStyle = it) },
                    onVibrationStrengthChange = { controls = controls.copy(vibrationStrength = it) },
                    onShowOnboarding = {}
                )
            }
        }

        composeRule.onNodeWithText("语音风格").assertExists()
        composeRule.onNodeWithContentDescription("选择详细语音风格，补充目标类别和避让建议")
            .performScrollTo()
            .performClick()
        composeRule.onNodeWithContentDescription("选择强震动强度，增强近处和迫近提醒")
            .performScrollTo()
            .performClick()
        composeRule.onNodeWithContentDescription("选择敏感提醒档位")
            .performScrollTo()
            .performClick()

        composeRule.runOnIdle {
            assertEquals(SpeechStyle.DETAILED, controls.speechStyle)
            assertEquals(VibrationStrength.STRONG, controls.vibrationStrength)
            assertEquals(AlertProfile.SENSITIVE, controls.alertProfile)
        }
    }

    @Test
    fun cameraControlPanelKeepsCoreTogglesAvailable() {
        var controls by mutableStateOf(defaultControls())

        composeRule.setContent {
            BlindAssistTheme {
                CameraControlPanel(
                    controls = controls,
                    guidance = sampleGuidance(),
                    fieldTestSummary = FieldTestSummaryUiState.empty(controls.alertProfile.displayName),
                    onDetectionChange = { controls = controls.copy(detectionEnabled = it) },
                    onSpeechChange = { controls = controls.copy(speechEnabled = it) },
                    onVibrationChange = { controls = controls.copy(vibrationEnabled = it) },
                    onCareModeChange = { controls = controls.copy(careModeEnabled = it) },
                    onDebugVisibleChange = { controls = controls.copy(debugVisible = it) },
                    onProfileChange = { controls = controls.copy(alertProfile = it) },
                    modifier = Modifier
                )
            }
        }

        composeRule.onNodeWithText("检测 开").performClick()
        composeRule.onNodeWithText("语音 开").assertExists()
        composeRule.onNodeWithText("震动 开").assertExists()

        composeRule.runOnIdle {
            assertFalse(controls.detectionEnabled)
        }
    }

    private fun defaultControls(): AssistControlsUiState {
        return AssistControlsUiState(
            detectionEnabled = true,
            speechEnabled = true,
            vibrationEnabled = true,
            careModeEnabled = false,
            debugVisible = false,
            alertProfile = AlertProfile.STANDARD,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STANDARD
        )
    }

    private fun sampleGuidance(): CameraGuidanceUiState {
        return CameraGuidanceUiState(
            title = "安全观察中",
            detail = "未发现需要提醒的近处风险",
            targetLine = "当前画面稳定",
            careTitle = "正在观察",
            careDetail = "前方暂未发现近处风险",
            careTargetLine = "请保持观察",
            debugText = "debug",
            titleColor = Color.White.toArgb(),
            statusBadge = "观察中",
            badgeColor = Color.White.toArgb(),
            badgeTextColor = Color.Black.toArgb(),
            careAccessibilitySummary = "正在观察",
            accessibilitySummary = "安全观察中",
            accessibilityKey = "test"
        )
    }
}
