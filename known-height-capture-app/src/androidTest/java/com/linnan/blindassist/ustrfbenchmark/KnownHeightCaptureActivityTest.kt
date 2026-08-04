package com.linnan.blindassist.ustrfbenchmark

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performScrollToIndex
import org.junit.Rule
import org.junit.Test

class KnownHeightCaptureActivityTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<KnownHeightCaptureActivity>()

    @Test
    fun emptyFormExplainsWorkflowAndCannotStart() {
        composeRule.onNodeWithText("高度标定采集").assertIsDisplayed()
        composeRule.onNodeWithText("1 · 本次采集").assertIsDisplayed()
        composeRule.onNodeWithText("2 · 现场量高").assertIsDisplayed()
        composeRule.onNodeWithTag("capture_form_list").performScrollToIndex(7)
        composeRule.onNodeWithTag("start_capture").assertIsNotEnabled()
    }
}
