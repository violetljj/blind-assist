package com.linnan.blindassist.camera

import android.content.Context
import androidx.lifecycle.LifecycleOwner
import com.linnan.blindassist.model.AssistInputSource
import com.linnan.blindassist.model.ReplayScenario
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class FrameSourceFactory @Inject constructor() {
    fun create(
        source: AssistInputSource,
        context: Context,
        lifecycleOwner: LifecycleOwner,
        replayScenario: ReplayScenario? = null
    ): FrameSource {
        return when (source) {
            AssistInputSource.PHONE_CAMERA -> CameraXFrameSource(context, lifecycleOwner)
            AssistInputSource.OFFLINE_REPLAY -> ReplayFrameSource(
                context = context,
                scenario = requireNotNull(replayScenario) {
                    "ReplayScenario is required for OFFLINE_REPLAY"
                }
            )
        }
    }
}
