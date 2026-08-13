package com.linnan.blindassist.ustrfbenchmark

import com.linnan.blindassist.ustrf.TaroBufferedPoseFrame
import com.linnan.blindassist.ustrf.TaroPoseDiverseFrameSelector
import com.linnan.blindassist.ustrf.TaroPoseDiverseSelection
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfVioPoseAdmission
import com.linnan.blindassist.ustrf.UstrfVioPoseAdmissionFailure

sealed interface TaroPoseDiverseCanaryStep {
    data class Evaluated(
        val frame: UstrfFrameStamp,
        val selection: TaroPoseDiverseSelection,
        val bufferedFrameCountAfterAppend: Int
    ) : TaroPoseDiverseCanaryStep

    data class AdmissionRejected(
        val frame: UstrfFrameStamp,
        val failure: UstrfVioPoseAdmissionFailure,
        val bufferedFrameCount: Int
    ) : TaroPoseDiverseCanaryStep
}

/**
 * Benchmark-only bridge from verified VIO admission to the TARO selector.
 *
 * Raw ARCore receipts cannot enter this harness. A caller must first produce an explicit
 * [UstrfVioPoseAdmission.Available], including inter-frame-stable world-frame and independently
 * verified extrinsics evidence. Rejected admissions are never buffered.
 */
class TaroPoseDiverseCanaryHarness(
    private val maximumRetainedAgeNs: Long = 1_000_000_000L,
    private val selector: TaroPoseDiverseFrameSelector = TaroPoseDiverseFrameSelector(enabled = true)
) {
    private val bufferedFrames = ArrayDeque<TaroBufferedPoseFrame>()

    init {
        require(maximumRetainedAgeNs > 0L)
    }

    fun observe(frame: UstrfFrameStamp, admission: UstrfVioPoseAdmission): TaroPoseDiverseCanaryStep {
        if (admission is UstrfVioPoseAdmission.Unavailable) {
            return TaroPoseDiverseCanaryStep.AdmissionRejected(frame, admission.failure, bufferedFrames.size)
        }
        admission as UstrfVioPoseAdmission.Available
        val oldestAllowedNs = (frame.capturedAtNs - maximumRetainedAgeNs).coerceAtLeast(0L)
        while (bufferedFrames.firstOrNull()?.frame?.capturedAtNs?.let { it < oldestAllowedNs } == true) {
            bufferedFrames.removeFirst()
        }
        val selection = selector.select(frame, admission.cameraPose, bufferedFrames.toList())
        bufferedFrames.addLast(TaroBufferedPoseFrame(frame, admission.cameraPose))
        return TaroPoseDiverseCanaryStep.Evaluated(frame, selection, bufferedFrames.size)
    }
}
