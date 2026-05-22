package com.linnan.blindassist.alert

import org.junit.Assert.assertEquals
import org.junit.Test

class AlertPolicyTest {
    @Test
    fun quietProfilePolicyMatchesPlannedValues() {
        val policy = AlertPolicy.forProfile(AlertProfile.QUIET)

        assertEquals(3, policy.mediumConfirmFrames)
        assertEquals(450L, policy.holdAlertMs)
        assertEquals(2200L, policy.nearCooldownMs)
        assertEquals(100L, policy.nearVibrationMs)
        assertEquals(1200L, policy.criticalCooldownMs)
        assertEquals(260L, policy.criticalVibrationMs)
    }

    @Test
    fun standardProfilePolicyKeepsExistingBehavior() {
        val policy = AlertPolicy.forProfile(AlertProfile.STANDARD)

        assertEquals(2, policy.mediumConfirmFrames)
        assertEquals(600L, policy.holdAlertMs)
        assertEquals(1500L, policy.nearCooldownMs)
        assertEquals(160L, policy.nearVibrationMs)
        assertEquals(850L, policy.criticalCooldownMs)
        assertEquals(420L, policy.criticalVibrationMs)
    }

    @Test
    fun sensitiveProfilePolicyMatchesPlannedValues() {
        val policy = AlertPolicy.forProfile(AlertProfile.SENSITIVE)

        assertEquals(1, policy.mediumConfirmFrames)
        assertEquals(800L, policy.holdAlertMs)
        assertEquals(1000L, policy.nearCooldownMs)
        assertEquals(220L, policy.nearVibrationMs)
        assertEquals(650L, policy.criticalCooldownMs)
        assertEquals(520L, policy.criticalVibrationMs)
    }

    @Test
    fun profileCyclesThroughQuietStandardSensitive() {
        assertEquals(AlertProfile.STANDARD, AlertProfile.QUIET.next())
        assertEquals(AlertProfile.SENSITIVE, AlertProfile.STANDARD.next())
        assertEquals(AlertProfile.QUIET, AlertProfile.SENSITIVE.next())
    }

    @Test
    fun generalScenarioKeepsExistingPolicy() {
        val original = AlertPolicy.forProfile(AlertProfile.STANDARD)
        val general = AlertPolicy.forProfile(AlertProfile.STANDARD, AssistScenario.GENERAL)

        assertEquals(original, general)
    }

    @Test
    fun indoorScenarioSoftensHoldAndNearCooldown() {
        val policy = AlertPolicy.forProfile(AlertProfile.STANDARD, AssistScenario.INDOOR)

        assertEquals(2, policy.mediumConfirmFrames)
        assertEquals(700L, policy.holdAlertMs)
        assertEquals(1700L, policy.nearCooldownMs)
        assertEquals(160L, policy.nearVibrationMs)
    }

    @Test
    fun corridorScenarioConfirmsEarlierAndStrengthensVibration() {
        val policy = AlertPolicy.forProfile(AlertProfile.STANDARD, AssistScenario.CORRIDOR)

        assertEquals(1, policy.mediumConfirmFrames)
        assertEquals(750L, policy.holdAlertMs)
        assertEquals(1350L, policy.nearCooldownMs)
        assertEquals(180L, policy.nearVibrationMs)
        assertEquals(460L, policy.criticalVibrationMs)
    }

    @Test
    fun crowdedScenarioReducesReminderFatigue() {
        val policy = AlertPolicy.forProfile(AlertProfile.STANDARD, AssistScenario.CROWDED)

        assertEquals(3, policy.mediumConfirmFrames)
        assertEquals(700L, policy.holdAlertMs)
        assertEquals(2200L, policy.nearCooldownMs)
        assertEquals(140L, policy.nearVibrationMs)
    }

    @Test
    fun outdoorSlowScenarioUsesClearerVibrationAndShorterCooldown() {
        val policy = AlertPolicy.forProfile(AlertProfile.STANDARD, AssistScenario.OUTDOOR_SLOW)

        assertEquals(2, policy.mediumConfirmFrames)
        assertEquals(800L, policy.holdAlertMs)
        assertEquals(1300L, policy.nearCooldownMs)
        assertEquals(200L, policy.nearVibrationMs)
        assertEquals(500L, policy.criticalVibrationMs)
    }

    @Test
    fun scenarioStorageFallbackAndCycleAreStable() {
        assertEquals(AssistScenario.GENERAL, AssistScenario.fromStorageValue("unknown"))
        assertEquals(AssistScenario.INDOOR, AssistScenario.GENERAL.next())
        assertEquals(AssistScenario.GENERAL, AssistScenario.OUTDOOR_SLOW.next())
    }
}
