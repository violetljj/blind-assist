package com.linnan.blindassist.ustrfbenchmark

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performScrollToIndex
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.performTextReplacement
import org.junit.Rule
import org.junit.Test

class KnownHeightCaptureActivityTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<KnownHeightCaptureActivity>()

    @Test
    fun emptyFormExplainsWorkflowAndCannotStart() {
        composeRule.onNodeWithText("快速采集").assertIsDisplayed()
        composeRule.onNodeWithText("固定支架 · 只填一次").assertIsDisplayed()
        composeRule.onNodeWithText("必须实际架高到 80–220 cm；15 cm 低支架不可用。").assertIsDisplayed()
        composeRule.onNodeWithText("打开三星快速测量").assertIsDisplayed()
        composeRule.onNodeWithTag("capture_form_list").performScrollToIndex(4)
        composeRule.onNodeWithTag("start_capture").assertIsNotEnabled()
    }

    @Test
    fun measuredFormEnablesStartWithoutExternalReferenceFile() {
        composeRule.onNodeWithTag("mount_id").performTextReplacement("三脚架A")
        composeRule.onNodeWithTag("height_1").performTextReplacement("143")
        composeRule.onNodeWithText("当前填写：143 cm（1.43 m），请和镜头实际位置核对。").assertIsDisplayed()
        composeRule.onNodeWithTag("capture_form_list").performScrollToIndex(3)
        composeRule.onNodeWithTag("development_distance_cm").performTextReplacement("29")
        composeRule.onNodeWithTag("capture_form_list").performScrollToIndex(4)
        composeRule.onNodeWithTag("start_capture").assertIsEnabled()
    }
}
