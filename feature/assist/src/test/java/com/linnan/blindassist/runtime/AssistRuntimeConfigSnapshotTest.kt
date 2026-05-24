package com.linnan.blindassist.runtime

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.preferences.DailyUsageMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test

class AssistRuntimeConfigSnapshotTest {
    @Test
    fun getReturnsLatestUpdatedSnapshot() {
        val initial = config(alertProfile = AlertProfile.STANDARD)
        val updated = config(alertProfile = AlertProfile.SENSITIVE)
        val snapshot = AssistRuntimeConfigSnapshot(initial)

        assertSame(initial, snapshot.get())
        assertSame(updated, snapshot.update(updated))
        assertSame(updated, snapshot.get())
    }

    @Test
    fun capturedFrameConfigIsNotChangedByLaterUpdates() {
        val initial = config(assistScenario = AssistScenario.GENERAL, appLanguage = AppLanguage.ZH)
        val snapshot = AssistRuntimeConfigSnapshot(initial)
        val frameConfig = snapshot.get()

        snapshot.update(config(assistScenario = AssistScenario.CORRIDOR, appLanguage = AppLanguage.EN))

        assertEquals(AssistScenario.GENERAL, frameConfig.assistScenario)
        assertEquals(AppLanguage.ZH, frameConfig.appLanguage)
        assertEquals(AssistScenario.CORRIDOR, snapshot.get().assistScenario)
        assertEquals(AppLanguage.EN, snapshot.get().appLanguage)
    }

    private fun config(
        alertProfile: AlertProfile = AlertProfile.STANDARD,
        assistScenario: AssistScenario = AssistScenario.GENERAL,
        appLanguage: AppLanguage = AppLanguage.ZH
    ): AssistRuntimeConfig {
        return AssistRuntimeConfig(
            detectionEnabled = true,
            speechEnabled = true,
            vibrationEnabled = true,
            careModeEnabled = false,
            alertProfile = alertProfile,
            assistScenario = assistScenario,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STANDARD,
            appLanguage = appLanguage,
            dailyUsageMode = DailyUsageMode.GENERAL_DAILY
        )
    }
}
