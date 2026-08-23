package com.linnan.blindassist.ui.compose

import android.Manifest
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.mutableStateOf
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
import androidx.compose.ui.test.assertHasClickAction
import androidx.compose.ui.test.assertHeightIsAtLeast
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
import com.linnan.blindassist.goal.GoalCompletionReceipt
import com.linnan.blindassist.goal.GoalHandoffState
import com.linnan.blindassist.goal.ConfirmationModality
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.MainActivity
import com.linnan.blindassist.model.AssistInputSource
import com.linnan.blindassist.model.ReplayScenario
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
                hasText("辅助") || hasText("Assist")
            }
            Thread.sleep(1500)
        }
    }

    @Test
    fun mainShellBottomNavigationSwitchesTopLevelPages() {
        prepareMainShell()

        composeRule.onNodeWithText("设置").performClick()
        composeRule.onNodeWithText("语音提醒").assertExists()
        composeRule.onNodeWithText("辅助").performClick()
        composeRule.onNodeWithText("选择辅助模式").assertExists()
    }

    @Test
    fun featureScreenExposesOneBrandNameWithoutGraphicDescription() {
        prepareMainShell()
        openFeaturesTab()

        composeRule.onAllNodesWithText("BlindAssist").assertCountEquals(1)
        composeRule.onAllNodesWithContentDescription("BlindAssist").assertCountEquals(0)
    }

    @Test
    fun phoneCameraEntryUsesExistingCameraPath() {
        prepareMainShell()
        openFeaturesTab()

        composeRule.onNodeWithTag("daily_usage_mode_selector").assertExists()
        composeRule.onNodeWithTag("home_primary_assist").performScrollTo().performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasTextOrContentDescription("返回功能页")
        }
    }

    @Test
    fun homeModeSelectorAppliesSensitiveShortcutToSettings() {
        prepareMainShell()
        openFeaturesTab()

        composeRule.onNodeWithContentDescription("选择灵敏辅助模式")
            .performScrollTo()
            .performClick()
        composeRule.onNodeWithText("设置").performClick()

        composeRule.onNodeWithContentDescription("选择敏感提醒档位")
            .performScrollTo()
            .assertExists()
        composeRule.onNodeWithContentDescription("选择标准语音风格，使用当前短句提醒")
            .performScrollTo()
            .assertExists()
        composeRule.onNodeWithContentDescription("选择强震动强度，增强近处和迫近提醒")
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
        composeRule.onNodeWithText("Assist").performClick()
        composeRule.onNodeWithTag("daily_usage_mode_selector").performScrollTo().assertExists()
        composeRule.onNodeWithContentDescription("Choose Sensitive assist mode")
            .performScrollTo()
            .performClick()
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
    }

    @Test
    fun cameraPanelShowsScenarioAndRiskExplanationWhenCameraPathOpens() {
        prepareMainShell()
        openFeaturesTab()

        composeRule.onNodeWithContentDescription("选择日常辅助模式")
            .performScrollTo()
            .performClick()
        composeRule.onNodeWithTag("home_primary_assist").performScrollTo().performClick()
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
        composeRule.onNodeWithText("辅助").performClick()
        composeRule.onNodeWithTag("home_primary_assist").performScrollTo().performClick()
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
                "辅助",
                "Assist",
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
            hasText("辅助") || hasText("Assist")
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
        if (!hasText("选择辅助模式") && !hasText("Choose assist mode")) {
            if (hasText("Assist")) {
                composeRule.onNodeWithText("Assist").performClick()
            } else {
                composeRule.onNodeWithText("辅助").performClick()
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
        if (hasText("Assist")) {
            composeRule.onNodeWithText("Assist").performClick()
        } else {
            composeRule.onNodeWithText("辅助").performClick()
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
        composeRule.onNodeWithTag("camera_debug_toggle").performClick()
        composeRule.onNodeWithTag("camera_debug_toggle").assertStateDescription("Collapsed")
        composeRule.onAllNodesWithText("FPS", substring = true).assertCountEquals(0)
    }

    @Test
    fun handoffReadyShowsExplicitAccessibleLargeConfirmationAction() {
        var confirmations = 0
        composeRule.setContent {
            BlindAssistTheme {
                GoalHandoffCard(
                    state = GoalHandoffState.HandoffReady(
                        goalId = "goal-1",
                        sessionId = "session-1",
                        handoffTimestamp = 1_000L,
                        handoffReason = "CURRENT_FRAME_HANDOFF_READY"
                    ),
                    language = AppLanguage.ZH,
                    onUserConfirmed = { confirmations += 1 }
                )
            }
        }

        composeRule.onNodeWithTag("goal_handoff_card")
            .assertStateDescription("已到交接点，等待用户明确确认")
            .assertIsDisplayed()
        composeRule.onNodeWithText("已经到目标前，请用手或盲杖确认入口。")
            .assertIsDisplayed()
        composeRule.onNodeWithContentDescription("确认已找到目标")
            .assertHasClickAction()
            .assertHeightIsAtLeast(48.dp)
            .performClick()
        assert(confirmations == 1)
    }

    @Test
    fun foundApproachAndCompletedStatesNeverExposeConfirmationButton() {
        val state = mutableStateOf<GoalHandoffState>(
            GoalHandoffState.Found("goal-1", "session-1")
        )
        composeRule.setContent {
            BlindAssistTheme {
                GoalHandoffCard(
                    state = state.value,
                    language = AppLanguage.ZH,
                    onUserConfirmed = {}
                )
            }
        }

        composeRule.onNodeWithText("找到目标，在你的右前方。").assertIsDisplayed()
        composeRule.onAllNodesWithTag("goal_handoff_confirm_button").assertCountEquals(0)

        state.value = GoalHandoffState.Approach("goal-1", "session-1")
        composeRule.onNodeWithText("稍向右，继续向前。").assertIsDisplayed()
        composeRule.onAllNodesWithTag("goal_handoff_confirm_button").assertCountEquals(0)

        state.value = GoalHandoffState.CompletedByUser(
            GoalCompletionReceipt(
                goalId = "goal-1",
                sessionId = "session-1",
                handoffTimestamp = 1_000L,
                handoffReason = "CURRENT_FRAME_HANDOFF_READY",
                confirmationModality = ConfirmationModality.VOICE,
                confirmationTimestamp = 1_100L
            )
        )
        composeRule.onNodeWithText("你已确认找到了。").assertIsDisplayed()
        composeRule.onNodeWithTag("goal_handoff_card")
            .assertStateDescription("已由用户明确确认完成")
        composeRule.onAllNodesWithTag("goal_handoff_confirm_button").assertCountEquals(0)
    }

    @Test
    fun firstRunShowsOnboardingOnTheFirstComposeFrame() {
        composeRule.mainClock.autoAdvance = false
        composeRule.setContent {
            BlindAssistTheme {
                BlindAssistApp(
                    state = appState(showOnboarding = true),
                    actions = appActions()
                )
            }
        }

        composeRule.onNodeWithText("开始使用 BlindAssist").assertIsDisplayed()
        composeRule.onAllNodesWithText("本地视觉辅助引擎启动中").assertCountEquals(0)
        composeRule.onAllNodesWithText("跳过启动页").assertCountEquals(0)
    }

    @Test
    fun returningUserShowsMainShellOnTheFirstComposeFrame() {
        composeRule.mainClock.autoAdvance = false
        composeRule.setContent {
            BlindAssistTheme {
                BlindAssistApp(
                    state = appState(showOnboarding = false),
                    actions = appActions()
                )
            }
        }

        composeRule.onNodeWithText("Choose assist mode").assertIsDisplayed()
        composeRule.onAllNodesWithText("本地视觉辅助引擎启动中").assertCountEquals(0)
    }

    @Test
    fun experimentalEditionBannerKeepsSafetyBoundaryVisible() {
        composeRule.setContent {
            BlindAssistTheme {
                ExperimentalEditionBanner("USTRF二维路线代理实验版 · 不可用于独立行走")
            }
        }

        composeRule.onNodeWithTag("experimental_edition_banner").assertIsDisplayed()
        composeRule.onNodeWithText("USTRF二维路线代理实验版 · 不可用于独立行走").assertIsDisplayed()
    }

    @Test
    fun cameraExperienceUsesEnglishBackAndToggleSemantics() {
        composeRule.setContent {
            BlindAssistTheme {
                CameraExperienceScreen(
                    controls = cameraPanelControls(debugVisible = true),
                    guidance = cameraPanelGuidance(),
                    fieldTestSummary = cameraPanelSummary(),
                    inputSource = AssistInputSource.PHONE_CAMERA,
                    replayScenario = null,
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
                    onCameraViewsReady = { _, _, _ -> }
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
        val showDialog = mutableStateOf(true)
        composeRule.setContent {
            BlindAssistTheme {
                if (showDialog.value) {
                    CameraPermissionExplanationDialog(
                        language = AppLanguage.EN,
                        onContinue = { showDialog.value = false },
                        onDismiss = { showDialog.value = false }
                    )
                }
            }
        }
        composeRule.onNodeWithText("Camera permission needed").assertExists()
        composeRule.onNodeWithText("Continue and allow").assertExists()
        composeRule.onNodeWithText("does not upload images", substring = true).assertExists()
        composeRule.onNodeWithText("Not now").performClick()
        composeRule.onNodeWithText("Camera permission needed").assertDoesNotExist()
    }

    @Test
    fun glassesSimulatorReleaseStateUsesExplicitSimulationCopyAndHidesReplay() {
        composeRule.setContent {
            BlindAssistTheme {
                GlassesSimulatorScreen(
                    state = GlassesSimulatorUiState(
                        connectionState = GlassesConnectionState.DISCONNECTED,
                        debugReplayAvailable = false
                    ),
                    language = AppLanguage.EN,
                    onBack = {},
                    onConnect = {},
                    onConnectionCompleted = {},
                    onLowBattery = {},
                    onDisconnect = {},
                    onReset = {},
                    onReplayScenarioSelected = {},
                    onStartReplay = {}
                )
            }
        }
        composeRule.onNodeWithText("Simulated glasses center").assertExists()
        composeRule.onNodeWithText("no Bluetooth scan", substring = true).assertExists()
        composeRule.onNodeWithTag("simulate_glasses_connect").assertExists()
        composeRule.onAllNodesWithTag("start_offline_replay").assertCountEquals(0)
    }

    @Test
    fun glassesSimulatorDebugConnectedStateExposesReplayAndCallbacks() {
        var selected: ReplayScenario? = null
        var started: ReplayScenario? = null
        composeRule.setContent {
            BlindAssistTheme {
                GlassesSimulatorScreen(
                    state = GlassesSimulatorUiState(
                        connectionState = GlassesConnectionState.CONNECTED,
                        batteryPercent = 82,
                        selectedInput = AssistInputSource.OFFLINE_REPLAY,
                        selectedReplayScenario = ReplayScenario.HIGH_CENTER,
                        debugReplayAvailable = true
                    ),
                    language = AppLanguage.EN,
                    onBack = {},
                    onConnect = {},
                    onConnectionCompleted = {},
                    onLowBattery = {},
                    onDisconnect = {},
                    onReset = {},
                    onReplayScenarioSelected = { selected = it },
                    onStartReplay = { started = it }
                )
            }
        }

        composeRule.onNodeWithTag("replay_scenario_medium_right").performScrollTo().performClick()
        assert(selected == ReplayScenario.MEDIUM_RIGHT)
        composeRule.onNodeWithTag("start_offline_replay").performScrollTo().performClick()
        assert(started == ReplayScenario.HIGH_CENTER)
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

    private fun appState(showOnboarding: Boolean): BlindAssistAppState {
        return BlindAssistAppState(
            controls = cameraPanelControls(debugVisible = false),
            cameraGuidance = cameraPanelGuidance(),
            fieldTestSummary = cameraPanelSummary(),
            modelStatus = "ready",
            appVersion = "test",
            cameraActive = false,
            activeInputSource = AssistInputSource.PHONE_CAMERA,
            activeReplayScenario = null,
            showOnboarding = showOnboarding,
            showGlassesCenter = false,
            glassesSimulator = GlassesSimulatorUiState()
        )
    }

    private fun appActions(): BlindAssistAppActions {
        return BlindAssistAppActions(
            runtime = AssistRuntimeUiActions(
                onOpenCamera = {},
                onCloseCamera = {},
                onStartOfflineReplay = {},
                onDetectionChange = {},
                onSpeechChange = {},
                onVibrationChange = {},
                onCareModeChange = {},
                onDebugVisibleChange = {},
                onProfileChange = {},
                onScenarioChange = {},
                onSpeechStyleChange = {},
                onVibrationStrengthChange = {},
                onDailyUsageModeChange = {},
                onQuietShortcut = {},
                onSensitiveShortcut = {},
                onLanguageChange = {},
                onCameraViewsReady = { _, _, _ -> }
            ),
            navigation = AssistNavigationActions(
                onCompleteOnboarding = {},
                onShowOnboarding = {},
                onShowGlassesCenter = {},
                onDismissGlassesCenter = {}
            ),
            glasses = GlassesSimulatorActions(
                onConnect = {},
                onDisconnect = {},
                onStartLiveAssist = {},
                onReplayScenarioSelected = {}
            )
        )
    }

    private fun SemanticsNodeInteraction.assertStateDescription(value: String): SemanticsNodeInteraction {
        return assert(SemanticsMatcher.expectValue(SemanticsProperties.StateDescription, value))
    }
}
