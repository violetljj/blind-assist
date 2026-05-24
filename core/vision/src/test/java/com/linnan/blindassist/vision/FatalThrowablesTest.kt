package com.linnan.blindassist.vision

import org.junit.Assert.assertSame
import org.junit.Assert.fail
import org.junit.Test

class FatalThrowablesTest {
    @Test
    fun rethrowIfFatalAllowsRecoverableExceptions() {
        FatalThrowables.rethrowIfFatal(IllegalStateException("recoverable"))
    }

    @Test
    fun rethrowIfFatalRethrowsVirtualMachineError() {
        val expected = object : VirtualMachineError("fatal") {}

        try {
            FatalThrowables.rethrowIfFatal(expected)
            fail("Expected fatal error to be rethrown")
        } catch (actual: VirtualMachineError) {
            assertSame(expected, actual)
        }
    }

    @Test
    fun rethrowIfFatalRethrowsLinkageError() {
        val expected = LinkageError("fatal")

        try {
            FatalThrowables.rethrowIfFatal(expected)
            fail("Expected linkage error to be rethrown")
        } catch (actual: LinkageError) {
            assertSame(expected, actual)
        }
    }
}
