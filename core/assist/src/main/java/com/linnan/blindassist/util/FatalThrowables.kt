package com.linnan.blindassist.util

object FatalThrowables {
    fun rethrowIfFatal(error: Throwable) {
        when (error) {
            is VirtualMachineError,
            is ThreadDeath,
            is LinkageError -> throw error
        }
    }
}
