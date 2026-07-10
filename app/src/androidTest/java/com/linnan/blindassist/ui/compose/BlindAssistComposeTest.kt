package com.linnan.blindassist.ui.compose

import android.Manifest
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
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
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.SemanticsNodeInteraction
import androidx.compose.ui.test.assert
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithContentDescription
import androidx.compose.ui.test.onAllNodesWithTag
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
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.MainActivity
import com.linnan.blindassist.preferences.DailyUsageMode
import org.junit.After
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.rules.TestRule
import org.junit.runner.RunWith
import java.io.FileInputStream

@RunWith(AndroidJUnit4::class)
class BlindAssistComposeTest {
    private val permissionRule: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)
    private val composeRule = createAndroidComposeRule<MainActivity>()

    @get:Rule
    val ruleChain: TestRule = RuleChain.outerRule(permissionRule).around(composeRule)

    @Before
    fun dismissDeviceCompatibilityDialog() {
        dismissAndroidCompatibilityDialogIfPresent()
    }

    @After
    fun closeCameraIfOpen() {
        val backDescription = when {
            hasContentDescription("返回功能页") -> "返回功能页"
            hasContentDescription("Back to features") -> "Back to features"
            else -> null
        }
        if (backDescription != null) {
            composeRule.onNodeWithContentDescription(backDescription).performClick()
            composeRule.waitUntil(timeoutMillis = 5000) {
                hasText("功能") || hasText("Features")
            }
            Thread.sleep(1500)
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

        composeRule.onNodeWithText("更换日常模式").performScrollTo().performClick()
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
        composeRule.onNodeWithText("Features").performClick()
        composeRule.onNodeWithTag("daily_usage_mode_selector").performScrollTo().assertExists()
        composeRule.onNodeWithText("Change daily mode").performScrollTo().performClick()
        composeRule.onNodeWithContentDescription("Choose Corridor daily mode, Earlier attention to sustained front risks. Profile Sensitive, scenario Corridor, speech Standard, vibration Standard, Care Mode on.")
            .performScrollTo()
            .assertExists()
        composeRule.onNodeWithText("Settings").performClick()
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

        composeRule.onNodeWithText("更换日常模式").performScrollTo().performClick()
        composeRule.onNodeWithText("通用日常").performScrollTo().performClick()
        composeRule.onNodeWithText("使用手机摄像头").performScrollTo().performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasTextOrContentDescription("返回功能页")
        }
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasAnyText("相机启动中", "检测已开启", "持续检测中", "模型不可用")
        }

        composeRule.onNodeWithTag("camera_scenario_label").assertExists()
        composeRule.onNodeWithTag("camera_daily_mode_label").assertExists()
        composeRule.onNodeWithTag("risk_explanation_headline").assertExists()
        composeRule.onAllNodesWithTag("camera_quiet_shortcut").assertCountEquals(0)
        composeRule.onAllNodesWithTag("camera_sensitive_shortcut").assertCountEquals(0)
        composeRule.onAllNodesWithTag("camera_scenario_toggle").assertCountEquals(0)
        composeRule.onAllNodesWithTag("camera_debug_toggle").assertCountEquals(0)
        composeRule.onNodeWithContentDescription("检测，当前已开启，点击关闭").performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasText("检测已暂停")
        }
    }

    @Test
    fun cameraDebugAreaAppearsOnlyAfterSettingsOptIn() {
        prepareMainShell()

        composeRule.onNodeWithText("设置").performClick()
        setSettingsSwitch("settings_care_mode_toggle", enabled = false)
        setSettingsSwitch("settings_debug_toggle", enabled = true)
        composeRule.onNodeWithText("功能").performClick()
        composeRule.onNodeWithText("使用手机摄像头").performScrollTo().performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasTextOrContentDescription("返回功能页")
        }

        composeRule.onNodeWithTag("camera_debug_toggle")
            .assertStateDescription("已收起")
            .assertExists()
        composeRule.onAllNodesWithText("FPS", substring = true).assertCountEquals(0)
        composeRule.onNodeWithTag("camera_debug_toggle").performClick()
        composeRule.onNodeWithTag("camera_debug_toggle").assertStateDescription("已展开")
        composeRule.onNodeWithText("FPS", substring = true).assertExists()
    }

    private fun prepareMainShell() {
        dismissSystemCompatibilityDialogIfPresent()
        composeRule.waitUntil(timeoutMillis = 8000) {
            hasAnyText(
                "功能",
                "Features",
                "开始使用 BlindAssist",
                "Start using BlindAssist",
                "跳过引导",
                "Skip guide"
            )
        }
        if (hasText("跳过引导")) {
            composeRule.onNodeWithText("跳过引导").performClick()
        } else if (hasText("Skip guide")) {
            composeRule.onNodeWithText("Skip guide").performClick()
        } else if (hasText("开始使用")) {
            composeRule.onNodeWithText("开始使用").performClick()
        } else if (hasText("Start using")) {
            composeRule.onNodeWithText("Start using").performClick()
        }
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasText("功能") || hasText("Features")
        }
        ensureChineseUi()
    }

    private fun dismissSystemCompatibilityDialogIfPresent() {
        val xml = runShellCommand(
            "uiautomator dump /sdcard/blindassist-window.xml >/dev/null 2>&1; cat /sdcard/blindassist-window.xml"
        )
        val hasCompatibilityDialog = xml.contains("Android 应用兼容性") ||
            xml.contains("Android app compatibility")
        if (!hasCompatibilityDialog) return

        val buttonMatch = Regex(
            """text="(?:不再显示|Don[^"]*show again|确定|OK)"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]""""
        ).findAll(xml).lastOrNull() ?: return
        val left = buttonMatch.groupValues[1].toInt()
        val top = buttonMatch.groupValues[2].toInt()
        val right = buttonMatch.groupValues[3].toInt()
        val bottom = buttonMatch.groupValues[4].toInt()
        runShellCommand("input tap ${(left + right) / 2} ${(top + bottom) / 2}")
        composeRule.waitUntil(timeoutMillis = 3000) {
            !runShellCommand(
                "uiautomator dump /sdcard/blindassist-window.xml >/dev/null 2>&1; cat /sdcard/blindassist-window.xml"
            ).contains("Android 应用兼容性")
        }
    }

    private fun runShellCommand(command: String): String {
        val descriptor = InstrumentationRegistry.getInstrumentation()
            .uiAutomation
            .executeShellCommand(command)
        return FileInputStream(descriptor.fileDescriptor).bufferedReader().use { it.readText() }
            .also { descriptor.close() }
    }

    private fun openFeaturesTab() {
        if (!hasText("日常使用向导") && !hasText("Daily usage guide")) {
            if (hasText("Features")) {
                composeRule.onNodeWithText("Features").performClick()
            } else {
                composeRule.onNodeWithText("功能").performClick()
            }
        }
    }

    private fun ensureChineseUi() {
        if (hasText("Settings")) {
            composeRule.onNodeWithText("Settings").performClick()
        } else {
            composeRule.onNodeWithText("设置").performClick()
        }
        if (hasText("Speech reminders")) {
            composeRule.onNodeWithTag("language_selector").performScrollTo()
            composeRule.onNodeWithText("Chinese").performScrollTo().performClick()
            composeRule.waitUntil(timeoutMillis = 5000) {
                hasText("语音提醒")
            }
        }
        if (hasText("Features")) {
            composeRule.onNodeWithText("Features").performClick()
        } else {
            composeRule.onNodeWithText("功能").performClick()
        }
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

    private fun setSettingsSwitch(tag: String, enabled: Boolean) {
        val enabledStates = setOf("on", "已开启")
        val disabledStates = setOf("off", "已关闭")
        val targetStates = if (enabled) enabledStates else disabledStates
        composeRule.onNodeWithTag(tag).performScrollTo()
        if (!hasTagWithAnyState(tag, targetStates)) {
            composeRule.onNodeWithTag(tag).performClick()
        }
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasTagWithAnyState(tag, targetStates)
        }
    }

    private fun hasTagWithAnyState(tag: String, states: Set<String>): Boolean {
        return composeRule.onAllNodesWithTag(tag).fetchSemanticsNodes().any { node ->
            node.config.getOrNull(SemanticsProperties.StateDescription) in states
        }
    }

    private fun SemanticsNodeInteraction.assertNoStateDescription(): SemanticsNodeInteraction {
        return assert(SemanticsMatcher.keyNotDefined(SemanticsProperties.StateDescription))
    }

    private fun SemanticsNodeInteraction.assertStateDescription(value: String): SemanticsNodeInteraction {
        return assert(SemanticsMatcher.expectValue(SemanticsProperties.StateDescription, value))
    }
}

private fun dismissAndroidCompatibilityDialogIfPresent() {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    repeat(6) {
        instrumentation.waitForIdleSync()
        val root = instrumentation.uiAutomation.rootInActiveWindow
        val button = root?.findFirstClickableText("不再显示")
            ?: root?.findFirstClickableText("确定")
            ?: root?.findFirstClickableText("Don't show again")
            ?: root?.findFirstClickableText("OK")
        if (button != null) {
            button.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            SystemClock.sleep(500)
            return
        }
        SystemClock.sleep(250)
    }
}

private fun AccessibilityNodeInfo.findFirstClickableText(text: String): AccessibilityNodeInfo? {
    return findAccessibilityNodeInfosByText(text)
        .firstOrNull { it.isEnabled && it.isClickable }
        ?: findAccessibilityNodeInfosByText(text)
            .mapNotNull { it.firstClickableParent() }
            .firstOrNull { it.isEnabled }
}

private fun AccessibilityNodeInfo.firstClickableParent(): AccessibilityNodeInfo? {
    var current: AccessibilityNodeInfo? = this
    while (current != null) {
        if (current.isClickable) {
            return current
        }
        current = current.parent
    }
    return null
}

@RunWith(AndroidJUnit4::class)
class CameraControlPanelStandaloneTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Before
    fun dismissDeviceCompatibilityDialog() {
        dismissAndroidCompatibilityDialogIfPresent()
    }

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

        composeRule.onNodeWithTag("camera_debug_toggle")
            .performScrollTo()
            .assertIsDisplayed()
            .assertStateDescription("Collapsed")
        composeRule.onAllNodesWithText("FPS", substring = true).assertCountEquals(0)
        composeRule.onNodeWithTag("camera_debug_toggle").performClick()
        composeRule.onNodeWithTag("camera_debug_toggle").assertStateDescription("Expanded")
        composeRule.onAllNodesWithText("FPS", substring = true).onFirst().performScrollTo().assertIsDisplayed()
        composeRule.onAllNodesWithTag("camera_quiet_shortcut").assertCountEquals(0)
    }

    @Test
    fun cameraExperienceUsesEnglishBackAndToggleSemantics() {
        composeRule.setContent {
            BlindAssistTheme {
                CameraExperienceScreen(
                    controls = cameraPanelControls(debugVisible = true),
                    guidance = cameraPanelGuidance(),
                    fieldTestSummary = cameraPanelSummary(),
                    onBack = {},
                    onDetectionChange = {},
                    onSpeechChange = {},
                    onVibrationChange = {},
                    onCareModeChange = {},
                    onDebugVisibleChange = {},
                    onProfileChange = {},
                    onScenarioChange = {},
                    onQuietShortcut = {},
                    onSensitiveShortcut = {},
                    onCameraViewsReady = { _, _ -> }
                )
            }
        }

        composeRule.onNodeWithContentDescription("Back to features").assertExists()
        composeRule.onNodeWithContentDescription("Detection, currently on, tap to turn off")
            .assertStateDescription("on")
            .assertExists()
        composeRule.onNodeWithTag("camera_debug_toggle")
            .assertStateDescription("Collapsed")
            .assertExists()
    }

    @Test
    fun cameraPermissionDialogUsesEnglishAccessibilityCopy() {
        composeRule.setContent {
            BlindAssistTheme {
                CameraPermissionExplanationDialog(
                    language = AppLanguage.EN,
                    onContinue = {},
                    onDismiss = {}
                )
            }
        }
        composeRule.onNodeWithText("Camera permission needed").assertExists()
        composeRule.onNodeWithText("Continue and allow").assertExists()
        composeRule.onNodeWithText("does not upload images", substring = true).assertExists()
    }

    @Test
    fun glassesPlaceholderDialogUsesEnglishAccessibilityCopy() {
        composeRule.setContent {
            BlindAssistTheme {
                GlassesPlaceholderDialog(
                    language = AppLanguage.EN,
                    onDismiss = {}
                )
            }
        }
        composeRule.onNodeWithText("Glasses device connection").assertExists()
        composeRule.onNodeWithText("does not scan Bluetooth", substring = true).assertExists()
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

    private fun SemanticsNodeInteraction.assertStateDescription(value: String): SemanticsNodeInteraction {
        return assert(SemanticsMatcher.expectValue(SemanticsProperties.StateDescription, value))
    }
}
