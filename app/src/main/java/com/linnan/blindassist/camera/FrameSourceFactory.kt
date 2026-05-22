package com.linnan.blindassist.camera

import android.content.Context
import androidx.lifecycle.LifecycleOwner
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class FrameSourceFactory @Inject constructor() {
    fun create(context: Context, lifecycleOwner: LifecycleOwner): FrameSource {
        return CameraXFrameSource(context, lifecycleOwner)
    }
}
