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
    fun showcaseCenterExposesDemoAndOnboardingActions() {
        prepareMainShell()
        openFeaturesTab()

        composeRule.onNodeWithTag("project_showcase_center").performScrollTo().assertExists()
        composeRule.onNodeWithText("本地识别").assertExists()
        composeRule.onNodeWithText("语音/震动提醒").assertExists()
        composeRule.onNodeWithText("现场测试摘要").assertExists()
        composeRule.onNodeWithText("原型安全边界").assertExists()

        composeRule.onNodeWithTag("showcase_show_onboarding").performScrollTo().performClick()
        composeRule.onNodeWithText("开始使用 BlindAssist").assertExists()
    }

    @Test
    fun showcaseStartDemoUsesExistingCameraEntryPath() {
        prepareMainShell()
        openFeaturesTab()

        composeRule.onNodeWithTag("showcase_start_camera_demo").performScrollTo().performClick()
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
