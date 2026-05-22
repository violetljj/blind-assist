package com.linnan.blindassist.camera

import android.graphics.Bitmap
import androidx.camera.view.PreviewView

interface FrameSource {
    fun start(
        previewView: PreviewView,
        onFrame: (Bitmap) -> Unit,
        onStarted: () -> Unit,
        onError: (Throwable) -> Unit
    )

    fun stop()
    fun shutdown()
}
