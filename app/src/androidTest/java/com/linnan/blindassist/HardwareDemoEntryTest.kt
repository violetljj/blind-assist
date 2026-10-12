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
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.rules.ExternalResource
import org.junit.rules.RuleChain

@RunWith(AndroidJUnit4::class)
class HardwareDemoEntryTest {
    val compose = createAndroidComposeRule<HardwareDemoActivity>()
    private val noLiveDevice = object : ExternalResource() {
        private var original: Boolean? = null
        private val preferences get() = InstrumentationRegistry.getInstrumentation().targetContext
            .getSharedPreferences("hardware_demo", 0)
        override fun before() {
            original = if (preferences.contains("wireless")) preferences.getBoolean("wireless", true) else null
            preferences.edit().putBoolean("wireless", false).commit()
        }
        override fun after() {
            preferences.edit().apply {
                original?.let { putBoolean("wireless", it) } ?: remove("wireless")
            }.commit()
        }
    }
    @get:Rule val rules: RuleChain = RuleChain.outerRule(noLiveDevice).around(compose)

    @Test fun launcherResolvesToHomeAndExplicitSessionCanEnd() {
        val activity = compose.activity
        val launch = activity.packageManager.getLaunchIntentForPackage(activity.packageName)
        assertEquals(MainActivity::class.java.name, launch?.component?.className)
        compose.onNodeWithTag("hardware_demo_more").assertIsDisplayed()
        compose.onNodeWithTag("hardware_demo_mode").assertIsDisplayed()
    }

    @Test fun feedbackPreferencesSurviveActivityRecreation() {
        val preferences = compose.activity.getSharedPreferences("hardware_demo", 0)
        val speech = preferences.getBoolean("speech", true)
        val vibration = preferences.getBoolean("vibration", true)
        val shared = compose.activity.getSharedPreferences("blindassist_user_preferences", 0)
        val sharedBefore = shared.all
        val english = preferences.getString("language", "zh") == "en"
        fun assertSwitch(tag: String, enabled: Boolean) {
            val node = compose.onNodeWithTag(tag).performScrollTo()
            node.assertContentDescriptionEquals(if (tag == "hardware_demo_speech") {
                if (english) "Speech" else "语音提示"
            } else if (english) "Vibration" else "震动提示")
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
            shared.edit().apply {
                listOf("speech_enabled", "vibration_enabled").forEach { key ->
                    if (sharedBefore.containsKey(key)) putBoolean(key, sharedBefore[key] as Boolean) else remove(key)
                }
            }.commit()
        }
    }

    @Test fun algorithmSwitchPersistsAndCanReturnToOriginalMode() {
        val preferences = compose.activity.getSharedPreferences("hardware_demo", 0)
        val original = preferences.getBoolean("a_local", true)
        val wireless = preferences.getBoolean("wireless", true)
        try {
            // Preference persistence is independent of live H3, whose unaligned clock disables LOCAL.
            preferences.edit().putBoolean("wireless", false).commit()
            compose.activityRule.scenario.recreate()
            compose.onNodeWithTag("hardware_demo_detail_toggle").performScrollTo().performClick()
            val toggle = compose.onNodeWithTag("hardware_demo_local").performScrollTo()
            toggle.assertContentDescriptionEquals("A+LOCAL 实验模式")
            toggle.performClick()
            compose.activityRule.scenario.recreate()
            compose.onNodeWithTag("hardware_demo_detail_toggle").performScrollTo().performClick()
            val restored = compose.onNodeWithTag("hardware_demo_local").performScrollTo()
            if (original) restored.assertIsOff() else restored.assertIsOn()
        } finally { preferences.edit().putBoolean("a_local", original).putBoolean("wireless", wireless).commit() }
    }
}
