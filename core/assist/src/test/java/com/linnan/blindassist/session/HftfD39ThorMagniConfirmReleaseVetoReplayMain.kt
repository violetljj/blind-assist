package com.linnan.blindassist.session

/**
 * Frozen D39 replay: symmetric bidirectional scene evidence releases the
 * bounded contradiction latch on admitted collective approach.
 */
object HftfD39ThorMagniConfirmReleaseVetoReplayMain {
    @JvmStatic
    fun main(args: Array<String>) {
        HftfD37ThorMagniSceneScaleVetoReplayMain.run(
            args = args,
            label = "D39",
            candidateMode =
                DualLoopRuntimeMode.ACTIVE_CONTRADICT_TTL_CONFIRM_RELEASE,
            expectedSourceId =
                CausalSceneScaleTristateGeometryProducer.BIDIRECTIONAL_SOURCE_ID,
            includeLatchColumn = true,
            includeConfirmReleaseColumn = true
        )
    }
}
