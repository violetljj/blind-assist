package com.linnan.blindassist.session

/**
 * Frozen D38 replay: D37 source and scene evidence with a 250 ms feedback-only
 * contradiction latch. Outcome truth remains outside the Kotlin process.
 */
object HftfD38ThorMagniBoundedTemporalVetoReplayMain {
    @JvmStatic
    fun main(args: Array<String>) {
        HftfD37ThorMagniSceneScaleVetoReplayMain.run(
            args = args,
            label = "D38",
            candidateMode = DualLoopRuntimeMode.ACTIVE_CONTRADICT_TTL,
            expectedSourceId = CausalSceneScaleTristateGeometryProducer.SOURCE_ID,
            includeLatchColumn = true,
            includeConfirmReleaseColumn = false
        )
    }
}
