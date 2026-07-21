package com.linnan.blindassist.runtime

import android.util.Log
import com.linnan.blindassist.ustrf.UstrfEvidenceState
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfHealth
import com.linnan.blindassist.ustrf.UstrfPerceptionAssembly
import com.linnan.blindassist.ustrf.UstrfPerceptionAssemblyFailure
import com.linnan.blindassist.ustrf.UstrfPoseState
import com.linnan.blindassist.ustrf.UstrfSafetySession
import com.linnan.blindassist.ustrf.UstrfSessionInput
import com.linnan.blindassist.ustrf.UstrfSessionRecord
import com.linnan.blindassist.vision.DetectorFrameResult
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp

/** A write-only seam: USTRF observations can never affect Assist rendering or feedback. */
internal sealed interface AssistUstrfShadowEvent {
    data class Recorded(val record: UstrfSessionRecord) : AssistUstrfShadowEvent
    data class Abstained(val sourceFrame: FrameStamp?, val reason: String) : AssistUstrfShadowEvent
}

internal fun interface AssistUstrfShadowSink {
    fun record(event: AssistUstrfShadowEvent)
}

internal class AssistUstrfShadowAdapter(
    private val session: UstrfSafetySession = UstrfSafetySession(),
    private val sink: AssistUstrfShadowSink = AssistUstrfShadowSink(::logOnly)
) {
    /**
     * Records a fail-closed USTRF shadow tick. Detection boxes are deliberately not promoted to
     * metric geometry; until a calibrated device adapter exists, pose/geometry/motion stay missing.
     */
    fun observe(frame: DetectorFrameResult, decisionAtNs: Long) {
        val stamp = frame.sourceFrame
        if (stamp == null) {
            sinkSafely(AssistUstrfShadowEvent.Abstained(null, "source_frame_missing"))
            return
        }
        if (stamp.clockDomain != FrameClockDomain.ANDROID_ELAPSED_REALTIME) {
            sinkSafely(AssistUstrfShadowEvent.Abstained(stamp, "camera_clock_domain_unmapped"))
            return
        }
        if (decisionAtNs < stamp.capturedAtNs) {
            sinkSafely(AssistUstrfShadowEvent.Abstained(stamp, "decision_clock_precedes_capture"))
            return
        }
        try {
            val source = UstrfFrameStamp(stamp.frameId, stamp.capturedAtNs, stamp.coordinateFrame)
            val record = session.evaluate(
                UstrfSessionInput(
                    frame = source,
                    health = UstrfHealth(
                        pose = UstrfPoseState.MISSING,
                        capture = UstrfEvidenceState.VALID,
                        geometry = UstrfEvidenceState.MISSING,
                        motion = UstrfEvidenceState.MISSING
                    ),
                    perception = UstrfPerceptionAssembly.Unavailable(
                        setOf(
                            UstrfPerceptionAssemblyFailure.POSE_UNAVAILABLE,
                            UstrfPerceptionAssemblyFailure.GEOMETRY_UNAVAILABLE,
                            UstrfPerceptionAssemblyFailure.MOTION_EVIDENCE_UNAVAILABLE
                        )
                    ),
                    route = null,
                    decisionAtNs = decisionAtNs
                )
            )
            sinkSafely(AssistUstrfShadowEvent.Recorded(record))
        } catch (error: RuntimeException) {
            // The shadow path is observational and is forbidden from failing the Assist frame.
            sinkSafely(AssistUstrfShadowEvent.Abstained(stamp, "shadow_adapter_failure:${error.javaClass.simpleName}"))
        }
    }

    fun reset() = session.reset()

    private fun sinkSafely(event: AssistUstrfShadowEvent) {
        try {
            sink.record(event)
        } catch (_: RuntimeException) {
            // A trace sink is never user-feedback authority and may not fail production processing.
        }
    }

    private companion object {
        const val TAG = "UstrfShadow"

        fun logOnly(event: AssistUstrfShadowEvent) {
            try {
                when (event) {
                    is AssistUstrfShadowEvent.Recorded -> Log.d(
                        TAG,
                        "frame=${event.record.frameId} action=${event.record.decision.action} shadowOnly=${event.record.structuredOutput.shadowOnly}"
                    )
                    is AssistUstrfShadowEvent.Abstained -> Log.d(
                        TAG,
                        "frame=${event.sourceFrame?.frameId ?: "none"} abstained=${event.reason}"
                    )
                }
            } catch (_: RuntimeException) {
                // Android Log is unavailable in local JVM tests.
            }
        }
    }
}
