package com.linnan.blindassist.ustrfbenchmark

import android.os.SystemClock
import com.linnan.blindassist.hftf.metricdepth.D45Yuv420Image
import com.linnan.blindassist.hftf.metricdepth.D45Yuv420ToRgbaDecoder
import com.linnan.blindassist.hftf.metricdepth.D45YuvBytePlane
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.linnan.blindassist.vision.TfliteYoloDetector
import java.nio.ByteBuffer
import java.nio.ByteOrder

enum class TaroOwnedRgbYoloEvidenceFailure {
    DETECTOR_NOT_READY,
    YUV_DECODE_FAILED,
    DETECTOR_SOURCE_FRAME_MISMATCH
}

sealed interface TaroOwnedRgbYoloEvidenceResult {
    data class Available(val receipt: TaroPositiveVisualReceipt) : TaroOwnedRgbYoloEvidenceResult
    data class Unavailable(
        val failure: TaroOwnedRgbYoloEvidenceFailure,
        val detail: String? = null
    ) : TaroOwnedRgbYoloEvidenceResult
}

/** Benchmark-only frozen-YOLO adapter for exact owned YUV payload receipts. */
class TaroOwnedRgbYoloEvidenceAdapter(
    private val detector: TfliteYoloDetector,
    private val elapsedRealtimeNs: () -> Long = SystemClock::elapsedRealtimeNanos
) {
    fun observe(
        payload: TaroOwnedRgbPayload,
        rotationDegrees: Int
    ): TaroOwnedRgbYoloEvidenceResult {
        if (!detector.isReady) {
            return unavailable(TaroOwnedRgbYoloEvidenceFailure.DETECTOR_NOT_READY, detector.statusMessage)
        }
        val decodeStartedNs = elapsedRealtimeNs()
        val rgba = try {
            require(payload.planes.size == EXPECTED_YUV_PLANE_COUNT)
            D45Yuv420ToRgbaDecoder.decode(
                D45Yuv420Image(
                    widthPx = payload.imageWidthPx,
                    heightPx = payload.imageHeightPx,
                    y = payload.planes[0].asDecoderPlane(),
                    u = payload.planes[1].asDecoderPlane(),
                    v = payload.planes[2].asDecoderPlane()
                )
            )
        } catch (error: IllegalArgumentException) {
            return unavailable(TaroOwnedRgbYoloEvidenceFailure.YUV_DECODE_FAILED, error.message)
        }
        val decodedNs = elapsedRealtimeNs()
        val source = payload.sourceFrame.toVisionFrameStamp(decodedNs)
        val frame = OwnedRgbaVisionFrame(
            width = rgba.widthPx,
            height = rgba.heightPx,
            rotationDegrees = rotationDegrees,
            frameStamp = source,
            buffer = ByteBuffer.allocateDirect(rgba.bytes.size)
                .order(ByteOrder.nativeOrder())
                .apply {
                    put(rgba.bytes)
                    rewind()
                }
        )
        val detectorFrame = detector.detect(frame)
        if (detectorFrame.sourceFrame != source) {
            return unavailable(TaroOwnedRgbYoloEvidenceFailure.DETECTOR_SOURCE_FRAME_MISMATCH)
        }
        val tokenization = TaroPositiveVisualEvidence.tokens(detectorFrame.detections)
        return TaroOwnedRgbYoloEvidenceResult.Available(
            TaroPositiveVisualReceipt(
                sourceFrame = payload.sourceFrame,
                tokens = tokenization.tokens,
                focusedTokens = tokenization.focusedTokens,
                decodeLatencyMs = (decodedNs - decodeStartedNs).toDouble() / 1_000_000.0,
                detectorPreprocessLatencyMs = detectorFrame.metrics.preprocessMs,
                detectorInferenceLatencyMs = detectorFrame.metrics.inferenceMs,
                detectorPostprocessLatencyMs = detectorFrame.metrics.postprocessMs,
                detectorTotalLatencyMs = detectorFrame.metrics.totalMs
            )
        )
    }

    private fun TaroOwnedYuvPlane.asDecoderPlane() = D45YuvBytePlane(
        rowStrideBytes = rowStrideBytes,
        pixelStrideBytes = pixelStrideBytes,
        buffer = ByteBuffer.wrap(bytes).asReadOnlyBuffer()
    )

    private fun com.linnan.blindassist.ustrf.UstrfFrameStamp.toVisionFrameStamp(receivedAtNs: Long) = FrameStamp(
        frameId = frameId,
        capturedAtNs = capturedAtNs,
        receivedAtNs = receivedAtNs,
        sourceId = SOURCE_ID,
        coordinateFrame = coordinateFrame,
        clockDomain = FrameClockDomain.CAMERA_HARDWARE_UNMAPPED
    )

    private fun unavailable(failure: TaroOwnedRgbYoloEvidenceFailure, detail: String? = null) =
        TaroOwnedRgbYoloEvidenceResult.Unavailable(failure, detail)

    private data class OwnedRgbaVisionFrame(
        override val width: Int,
        override val height: Int,
        override val rotationDegrees: Int,
        override val frameStamp: FrameStamp,
        override val buffer: ByteBuffer,
        override val rowStride: Int = width * RGBA_CHANNELS,
        override val pixelStride: Int = RGBA_CHANNELS
    ) : RgbaVisionFrame {
        override fun close() = Unit
    }

    private companion object {
        const val EXPECTED_YUV_PLANE_COUNT = 3
        const val RGBA_CHANNELS = 4
        const val SOURCE_ID = "taro-owned-arcore-rgb-r0"
    }
}
