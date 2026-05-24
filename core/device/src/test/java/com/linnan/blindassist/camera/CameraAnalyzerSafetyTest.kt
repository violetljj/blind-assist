package com.linnan.blindassist.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class CameraAnalyzerSafetyTest {
    @Test
    fun analyzeFrameReportsNonFatalErrorsAndClosesFrame() {
        val expected = IllegalStateException("bad frame")
        val reported = mutableListOf<Throwable>()
        var closed = false

        CameraAnalyzerSafety.analyzeFrame(
            closeFrame = { closed = true },
            reportError = { reported += it }
        ) {
            throw expected
        }

        assertTrue(closed)
        assertEquals(1, reported.size)
        assertSame(expected, reported.single())
    }

    @Test
    fun analyzeFrameRethrowsFatalErrorsAndStillClosesFrame() {
        val expected = object : VirtualMachineError("fatal") {}
        var closed = false

        try {
            CameraAnalyzerSafety.analyzeFrame(
                closeFrame = { closed = true },
                reportError = { throw AssertionError("fatal errors must not be reported") }
            ) {
                throw expected
            }
            fail("Expected fatal analyzer error to be rethrown")
        } catch (actual: VirtualMachineError) {
            assertSame(expected, actual)
        }

        assertTrue(closed)
    }
}
