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
    val stageTiming: DetectorStageTiming? = null,
    /** Calibration in the same display-oriented coordinates as [frameSize] and [detections]. */
    val cameraIntrinsics: CameraIntrinsics? = null
)

/** Android elapsed-realtime boundaries for one detector invocation. */
data class DetectorStageTiming(
    val preprocessStartNs: Long,
    val preprocessCompleteNs: Long,
    val preprocessLetterboxDrawStartNs: Long? = null,
    val preprocessLetterboxDrawCompleteNs: Long? = null,
    val preprocessBitmapPixelsCompleteNs: Long? = null,
    val preprocessInputWriteCompleteNs: Long? = null,
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
