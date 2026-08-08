package com.linnan.blindassist.ustrfbenchmark

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class KnownHeightCaptureContractTest {
    @Test
    fun simpleMeasuredFormComputesReceiptValuesAndCanStart() {
        val form = CaptureFormState(
            sessionId = "P0-20260804-01",
            phase = CapturePhase.P0,
            mountProfileId = "三脚架A",
            measurementMethod = MeasurementMethod.LASER,
            instrumentErrorCm = "0.5",
            heightReading1Cm = "143",
            heightReading2Cm = "144",
            heightReading3Cm = "143",
            nearDistanceM = "1.0",
            middleDistanceM = "2.0",
            farDistanceM = "3.0",
        )
        assertTrue(form.canStart)
        assertEquals(1.43, form.cameraHeightM!!, 1e-12)
        assertEquals(0.015, form.cameraHeightUncertaintyM!!, 1e-12)
        assertEquals(listOf("near", "middle", "far"), form.referencePoints!!.map { it.id })
    }

    @Test
    fun inconsistentHeightAndUnorderedReferencesFailClosed() {
        val form = CaptureFormState(
            sessionId = "P0-test",
            phase = CapturePhase.P0,
            mountProfileId = "支架A",
            heightReading1Cm = "140",
            heightReading2Cm = "145",
            heightReading3Cm = "150",
            nearDistanceM = "3",
            middleDistanceM = "2",
            farDistanceM = "1",
        )
        assertFalse(form.canStart)
        assertTrue(form.validationProblems().any { it.contains("重测") })
        assertTrue(form.validationProblems().any { it.contains("递增") })
    }

    @Test
    fun developmentFormNeedsOnlySavedHeightAndCurrentQuickMeasureDistance() {
        val form = CaptureFormState(
            sessionId = "DEV-20260804-01",
            phase = CapturePhase.DEV,
            mountProfileId = "固定支架",
            measurementMethod = MeasurementMethod.SAMSUNG_QUICK_MEASURE,
            instrumentErrorCm = "5.0",
            heightReading1Cm = "143",
            developmentDistanceCm = "29",
        )

        assertTrue(form.canStart)
        assertEquals(1.43, form.cameraHeightM!!, 1e-12)
        assertEquals(0.29, form.referencePoints!!.single().distanceM, 1e-12)
        assertEquals("current", form.referencePoints!!.single().id)
    }
}
