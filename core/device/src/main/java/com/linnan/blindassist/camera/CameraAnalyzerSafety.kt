package com.linnan.blindassist.camera

import com.linnan.blindassist.util.FatalThrowables

internal object CameraAnalyzerSafety {
    fun analyzeFrame(
        closeFrame: () -> Unit,
        reportError: (Throwable) -> Unit,
        processFrame: () -> Unit
    ) {
        try {
            processFrame()
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            reportError(error)
        } finally {
            closeFrame()
        }
    }
}
