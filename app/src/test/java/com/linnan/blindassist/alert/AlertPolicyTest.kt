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
}
