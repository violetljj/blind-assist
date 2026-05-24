package com.linnan.blindassist.vision

internal object FatalThrowables {
    fun rethrowIfFatal(error: Throwable) {
        if (error is VirtualMachineError || error is ThreadDeath || error is LinkageError) {
            throw error
        }
    }
}
