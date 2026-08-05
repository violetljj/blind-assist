package com.linnan.blindassist.vision

import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.session.DetectorMetrics

data class DetectorFrameResult(
    val detections: List<Detection>,
    val frameSize: FrameSize,
    val metrics: DetectorMetrics,
    val sourceFrame: FrameStamp? = null,
    val sourceRanging: RangingSample? = null,
    val stageTiming: DetectorStageTiming? = null
)

/** Android elapsed-realtime boundaries for one detector invocation. */
data class DetectorStageTiming(
    val preprocessStartNs: Long,
    val preprocessCompleteNs: Long,
    /** Host entry into Interpreter.run; delegate-internal enqueue is not exposed. */
    val qnnEnqueueNs: Long,
    val qnnCompleteNs: Long,
    val outputReadCompleteNs: Long,
    val postprocessCompleteNs: Long
) {
    init {
        require(preprocessStartNs >= 0L)
        require(preprocessCompleteNs >= preprocessStartNs)
        require(qnnEnqueueNs >= preprocessCompleteNs)
        require(qnnCompleteNs >= qnnEnqueueNs)
        require(outputReadCompleteNs >= qnnCompleteNs)
        require(postprocessCompleteNs >= outputReadCompleteNs)
    }
}
