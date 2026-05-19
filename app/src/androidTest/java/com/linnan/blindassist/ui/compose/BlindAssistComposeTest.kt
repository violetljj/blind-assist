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

        composeRule.onNodeWithTag("daily_usage_mode_selector").assertExists()
        composeRule.onNodeWithText("使用手机摄像头").performScrollTo().performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasText("需要相机权限") || hasText("返回功能页")
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

        composeRule.onNodeWithText("使用手机摄像头").performClick()
        composeRule.waitUntil(timeoutMillis = 5000) {
            hasText("需要相机权限") || hasText("返回功能页")
        }

        if (hasText("返回功能页")) {
            composeRule.onNodeWithTag("camera_scenario_label").assertExists()
            composeRule.onNodeWithTag("camera_daily_mode_label").assertExists()
            composeRule.onNodeWithTag("risk_explanation_headline").assertExists()
            composeRule.onNodeWithContentDescription("应用安静提醒快捷设置，保留当前场景，使用安静档位、简短语音和轻柔震动").assertExists()
            composeRule.onNodeWithContentDescription("应用敏感提醒快捷设置，保留当前场景，使用敏感档位、标准语音和强震动").assertExists()
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
}
