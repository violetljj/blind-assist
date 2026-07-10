package com.linnan.blindassist.camera

import androidx.camera.view.PreviewView
import com.linnan.blindassist.vision.VisionFrame

interface FrameSource {
    fun start(
        previewView: PreviewView?,
        onFrame: (VisionFrame) -> Unit,
        onStarted: () -> Unit,
        onError: (Throwable) -> Unit
    )

    fun stop()
    fun shutdown()
}
