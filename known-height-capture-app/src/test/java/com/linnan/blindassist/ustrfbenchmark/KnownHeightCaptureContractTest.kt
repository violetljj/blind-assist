package com.linnan.blindassist.ustrfbenchmark

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class KnownHeightCaptureContractTest {
    @Test
    fun completeMeasuredFormCanStart() {
        val form = CaptureFormState(
            sessionId = "P0-20260804-01",
            cameraHeightM = "1.43",
            cameraHeightUncertaintyM = "0.01",
            mountProfileId = "tripod-A",
            referenceDisplayName = "laser-reference.json",
        )
        assertTrue(form.canStart)
        assertEquals(120, form.phase.frameTarget)
    }

    @Test
    fun missingOrEstimatedPhysicalInputsFailClosed() {
        val form = CaptureFormState(sessionId = "x", cameraHeightM = "2.5")
        assertFalse(form.canStart)
        assertEquals(4, form.validationProblems().size)
    }
}
