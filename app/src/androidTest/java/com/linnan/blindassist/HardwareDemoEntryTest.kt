package com.linnan.blindassist

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsOff
import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.assertContentDescriptionEquals
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HardwareDemoEntryTest {
    @get:Rule val compose = createAndroidComposeRule<HardwareDemoActivity>()

    @Test fun launcherResolvesToHardwareWithMoreFunctionsAvailable() {
        val activity = compose.activity
        val launch = activity.packageManager.getLaunchIntentForPackage(activity.packageName)
        assertEquals(HardwareDemoActivity::class.java.name, launch?.component?.className)
        compose.onNodeWithTag("hardware_demo_more").assertIsDisplayed()
        compose.onNodeWithTag("hardware_demo_mode").assertIsDisplayed()
    }

    @Test fun feedbackPreferencesSurviveActivityRecreation() {
        val preferences = compose.activity.getSharedPreferences("hardware_demo", 0)
        val speech = preferences.getBoolean("speech", true)
        val vibration = preferences.getBoolean("vibration", true)
        fun assertSwitch(tag: String, enabled: Boolean) {
            val node = compose.onNodeWithTag(tag).performScrollTo()
            node.assertContentDescriptionEquals(if (tag == "hardware_demo_speech") "语音提示" else "震动提示")
            if (enabled) node.assertIsOn() else node.assertIsOff()
        }
        try {
            compose.onNodeWithTag("hardware_demo_speech").performScrollTo().performClick()
            compose.onNodeWithTag("hardware_demo_vibration").performScrollTo().performClick()
            compose.activityRule.scenario.recreate()
            assertSwitch("hardware_demo_speech", !speech)
            assertSwitch("hardware_demo_vibration", !vibration)
        } finally {
            preferences.edit().putBoolean("speech", speech).putBoolean("vibration", vibration).commit()
        }
    }
}
