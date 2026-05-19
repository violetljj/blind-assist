package com.linnan.blindassist.ui.compose

import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.MainActivity
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BlindAssistComposeTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<MainActivity>()

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

        composeRule.onNodeWithText("使用手机摄像头").performScrollTo().performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasText("需要相机权限") || hasText("返回功能页")
        }
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

        composeRule.onNodeWithText("使用手机摄像头").performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasText("需要相机权限") || hasText("返回功能页")
        }

        if (hasText("返回功能页")) {
            composeRule.onNodeWithTag("camera_scenario_label").assertExists()
            composeRule.onNodeWithTag("risk_explanation_headline").assertExists()
        }
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
    }

    private fun openFeaturesTab() {
        if (!hasText("选择一种辅助方式开始。当前版本优先提供手机摄像头本地识别，眼镜连接作为后续扩展入口保留。")) {
            composeRule.onNodeWithText("功能").performClick()
        }
    }

    private fun hasText(text: String): Boolean {
        return composeRule.onAllNodesWithText(text).fetchSemanticsNodes().isNotEmpty()
    }
}
