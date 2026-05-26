package com.linnan.blindassist.ui.compose

import android.Manifest
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.SemanticsNodeInteraction
import androidx.compose.ui.test.assert
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithContentDescription
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.MainActivity
import com.linnan.blindassist.preferences.DailyUsageMode
import org.junit.After
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.rules.TestRule
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BlindAssistComposeTest {
    private val permissionRule: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)
    private val composeRule = createAndroidComposeRule<MainActivity>()

    @get:Rule
    val ruleChain: TestRule = RuleChain.outerRule(permissionRule).around(composeRule)

    @After
    fun closeCameraIfOpen() {
        if (hasTextOrContentDescription("返回功能页")) {
            composeRule.onNodeWithContentDescription("返回功能页").performClick()
            composeRule.waitUntil(timeoutMillis = 5000) {
                hasText("功能")
            }
        }
    }

    @Test
    fun mainShellBottomNavigationSwitchesTopLevelPages() {
        prepareMainShell()

        composeRule.onNodeWithText("个人主页").performClick()
        composeRule.onNodeWithText("BlindAssist 用户").assertExists()
        composeRule.onNodeWithText("设置").performClick()
        composeRule.onNodeWithText("语音提醒").assertExists()
    }

    @Test
    fun phoneCameraEntryUsesExistingCameraPath() {
        prepareMainShell()
        openFeaturesTab()

        composeRule.onNodeWithTag("daily_usage_mode_selector").assertExists()
        composeRule.onNodeWithText("使用手机摄像头").performScrollTo().performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasTextOrContentDescription("返回功能页")
        }
    }

    @Test
    fun featureDailyGuideAppliesCorridorModeToSettings() {
        prepareMainShell()
        openFeaturesTab()

        composeRule.onNodeWithText("走廊通行").performScrollTo().performClick()
        composeRule.onNodeWithText("设置").performClick()

        composeRule.onNodeWithContentDescription("选择走廊通行使用场景，更早关注正前方持续风险")
            .performScrollTo()
            .assertExists()
        composeRule.onNodeWithContentDescription("选择敏感提醒档位")
            .performScrollTo()
            .assertExists()
        composeRule.onNodeWithContentDescription("选择标准语音风格，使用当前短句提醒")
            .performScrollTo()
            .assertExists()
        composeRule.onNodeWithContentDescription("选择标准震动强度，使用默认触觉反馈")
            .performScrollTo()
            .assertExists()
    }

    @Test
    fun settingsScreenChangesFeedbackDetailControls() {
        prepareMainShell()

        composeRule.onNodeWithText("设置").performClick()
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
        composeRule.onNodeWithTag("scenario_selector").performScrollTo().assertExists()
        composeRule.onNodeWithContentDescription("选择走廊通行使用场景，更早关注正前方持续风险")
            .performScrollTo()
            .performClick()
    }

    @Test
    fun settingsScreenCanSwitchCoreTextToEnglish() {
        prepareMainShell()

        composeRule.onNodeWithText("设置").performClick()
        composeRule.onNodeWithTag("language_selector").performScrollTo().assertExists()
        composeRule.onNodeWithText("English").performScrollTo().performClick()
        composeRule.onNodeWithText("Speech reminders").assertExists()
        composeRule.onNodeWithText("功能").performClick()
        composeRule.onNodeWithTag("daily_usage_mode_selector").performScrollTo().assertExists()
        composeRule.onNodeWithContentDescription("Choose Corridor daily mode, Earlier attention to sustained front risks. Profile Sensitive, scenario Corridor, speech Standard, vibration Standard, Care Mode on.")
            .performScrollTo()
            .assertExists()
        composeRule.onNodeWithText("设置").performClick()
        composeRule.onNodeWithContentDescription("Choose Detailed speech style, Adds object type and avoidance guidance")
            .performScrollTo()
            .assertExists()
        composeRule.onNodeWithContentDescription("Choose Strong vibration strength, Strengthen near and critical reminders")
            .performScrollTo()
            .assertExists()
        composeRule.onNodeWithContentDescription("Choose Sensitive reminder profile")
            .performScrollTo()
            .assertExists()
        composeRule.onNodeWithContentDescription("Choose Corridor usage scenario, Notice sustained front risks earlier")
            .performScrollTo()
            .assertExists()
    }

    @Test
    fun cameraPanelShowsScenarioAndRiskExplanationWhenCameraPathOpens() {
        prepareMainShell()
        openFeaturesTab()

        composeRule.onNodeWithText("通用日常").performScrollTo().performClick()
        composeRule.onNodeWithText("使用手机摄像头").performScrollTo().performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasTextOrContentDescription("返回功能页")
        }
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasAnyText("相机启动中", "检测已开启", "安全观察中", "模型不可用")
        }

        composeRule.onNodeWithTag("camera_scenario_label").assertExists()
        composeRule.onNodeWithTag("camera_daily_mode_label").assertExists()
        composeRule.onNodeWithTag("risk_explanation_headline").assertExists()
        composeRule.onNodeWithContentDescription("应用安静提醒快捷设置，保留当前场景，使用安静档位、简短语音和轻柔震动").assertExists()
        composeRule.onNodeWithContentDescription("应用敏感提醒快捷设置，保留当前场景，使用敏感档位、标准语音和强震动").assertExists()
        composeRule.onNodeWithTag("camera_quiet_shortcut").assertNoStateDescription()
        composeRule.onNodeWithTag("camera_sensitive_shortcut").assertNoStateDescription()
        composeRule.onNodeWithTag("camera_scenario_toggle").assertNoStateDescription()
        composeRule.onNodeWithTag("camera_debug_toggle")
            .assertStateDescription("已收起")
            .assertExists()
        composeRule.onNodeWithTag("camera_debug_toggle").performClick()
        composeRule.onNodeWithTag("camera_debug_toggle").assertStateDescription("已展开")
        composeRule.onNodeWithText("FPS", substring = true).assertExists()
        composeRule.onNodeWithContentDescription("检测，当前已开启，点击关闭").performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasText("检测已暂停")
        }
        composeRule.onNodeWithContentDescription("检测，当前已关闭，点击开启").performClick()
    }

    private fun prepareMainShell() {
        composeRule.waitUntil(timeoutMillis = 8000) {
            hasText("功能") || hasText("开始使用 BlindAssist") || hasText("跳过引导")
        }
        if (hasText("跳过引导")) {
            composeRule.onNodeWithText("跳过引导").performClick()
        } else if (hasText("开始使用")) {
            composeRule.onNodeWithText("开始使用").performClick()
        }
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasText("功能")
        }
        ensureChineseUi()
    }

    private fun openFeaturesTab() {
        if (!hasText("日常使用向导") && !hasText("Daily usage guide")) {
            composeRule.onNodeWithText("功能").performClick()
        }
    }

    private fun ensureChineseUi() {
        composeRule.onNodeWithText("设置").performClick()
        if (hasText("Speech reminders")) {
            composeRule.onNodeWithTag("language_selector").performScrollTo()
            composeRule.onNodeWithText("Chinese").performScrollTo().performClick()
            composeRule.waitUntil(timeoutMillis = 5000) {
                hasText("语音提醒")
            }
        }
        composeRule.onNodeWithText("功能").performClick()
    }

    private fun hasText(text: String): Boolean {
        return composeRule.onAllNodesWithText(text).fetchSemanticsNodes().isNotEmpty()
    }

    private fun hasContentDescription(text: String): Boolean {
        return composeRule.onAllNodesWithContentDescription(text).fetchSemanticsNodes().isNotEmpty()
    }

    private fun hasTextOrContentDescription(text: String): Boolean {
        return hasText(text) || hasContentDescription(text)
    }

    private fun hasAnyText(vararg texts: String): Boolean {
        return texts.any(::hasText)
    }

    private fun SemanticsNodeInteraction.assertNoStateDescription(): SemanticsNodeInteraction {
        return assert(SemanticsMatcher.keyNotDefined(SemanticsProperties.StateDescription))
    }

    private fun SemanticsNodeInteraction.assertStateDescription(value: String): SemanticsNodeInteraction {
        return assert(SemanticsMatcher.expectValue(SemanticsProperties.StateDescription, value))
    }
}

@RunWith(AndroidJUnit4::class)
class CameraControlPanelStandaloneTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun cameraPanelKeepsDebugControlsReachableWithLargeFont() {
        composeRule.setContent {
            BlindAssistTheme {
                val density = LocalDensity.current
                CompositionLocalProvider(
                    LocalDensity provides Density(density.density, fontScale = 1.9f)
                ) {
                    Box(
                        modifier = Modifier
                            .size(width = 360.dp, height = 640.dp)
                            .background(Color.Black),
                        contentAlignment = Alignment.BottomCenter
                    ) {
                        CameraControlPanel(
                            controls = cameraPanelControls(debugVisible = true),
                            guidance = cameraPanelGuidance(),
                            fieldTestSummary = cameraPanelSummary(),
                            onDetectionChange = {},
                            onSpeechChange = {},
                            onVibrationChange = {},
                            onCareModeChange = {},
                            onDebugVisibleChange = {},
                            onProfileChange = {},
                            onScenarioChange = {},
                            onQuietShortcut = {},
                            onSensitiveShortcut = {},
                            modifier = Modifier.padding(12.dp)
                        )
                    }
                }
            }
        }

        composeRule.onNodeWithTag("camera_debug_toggle").performScrollTo().assertIsDisplayed()
        composeRule.onAllNodesWithText("FPS", substring = true).onFirst().performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("camera_quiet_shortcut").performScrollTo().assertIsDisplayed()
    }

    private fun cameraPanelControls(debugVisible: Boolean): AssistControlsUiState {
        return AssistControlsUiState(
            detectionEnabled = true,
            speechEnabled = true,
            vibrationEnabled = true,
            careModeEnabled = false,
            debugVisible = debugVisible,
            alertProfile = AlertProfile.STANDARD,
            assistScenario = AssistScenario.GENERAL,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STANDARD,
            appLanguage = AppLanguage.EN,
            dailyUsageMode = DailyUsageMode.GENERAL_DAILY
        )
    }

    private fun cameraPanelGuidance(): CameraGuidanceUiState {
        return CameraGuidanceUiState(
            title = "Observing",
            detail = "Waiting for a stable camera frame and risk result.",
            targetLine = "Model: ready",
            careTitle = "Clear ahead",
            careDetail = "Move naturally and keep listening for alerts.",
            careTargetLine = "Speech and vibration are both available.",
            debugText = "FPS: 18.0\nModel: ready\nRecent risk: clear",
            scenarioName = "General",
            explanationHeadline = "No immediate risk",
            explanationDetail = "Objects are visible but below alert thresholds.",
            careExplanation = "No immediate risk",
            titleColor = android.graphics.Color.rgb(99, 230, 166),
            statusBadge = "Stable",
            badgeColor = android.graphics.Color.rgb(160, 255, 215),
            badgeTextColor = android.graphics.Color.rgb(6, 24, 18),
            careAccessibilitySummary = "Clear ahead",
            accessibilitySummary = "Observing",
            accessibilityKey = "large-font-test"
        )
    }

    private fun cameraPanelSummary(): FieldTestSummaryUiState {
        return FieldTestSummaryUiState(
            title = "Field test summary",
            statusText = "Camera session active",
            detailText = "Runtime: 1 min 20 sec\nFrames: 90\nAlerts: 3\nAverage FPS: 18.0",
            accessibilityText = "Field test summary, camera session active, average FPS 18."
        )
    }
}
