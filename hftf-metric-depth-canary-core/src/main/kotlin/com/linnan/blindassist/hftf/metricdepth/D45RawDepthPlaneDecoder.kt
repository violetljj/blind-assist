package com.linnan.blindassist.hftf.metricdepth

import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class D45StridedBytePlane(
    val widthPx: Int,
    val heightPx: Int,
    val rowStrideBytes: Int,
    val pixelStrideBytes: Int,
    val buffer: ByteBuffer
) {
    init {
        require(widthPx > 0 && heightPx > 0)
        require(rowStrideBytes > 0 && pixelStrideBytes > 0)
        require(rowStrideBytes >= (widthPx - 1) * pixelStrideBytes + 1) {
            "row stride cannot be shorter than one logical row"
        }
    }
}

data class D45DecodedRawDepthRaster(
    val widthPx: Int,
    val heightPx: Int,
    /** Unsigned 16-bit optical-axis depth in millimetres; zero remains invalid. */
    val depthMillimeters: IntArray,
    /** Unsigned 8-bit raw confidence normalized to [0, 1]. */
    val confidence: FloatArray
) {
    init {
        require(widthPx > 0 && heightPx > 0)
        require(depthMillimeters.size == widthPx * heightPx)
        require(confidence.size == depthMillimeters.size)
        require(depthMillimeters.all { it in 0..0xFFFF })
        require(confidence.all { it.isFinite() && it in 0f..1f })
    }
}

/**
 * Decodes source-native ARCore raw depth and confidence without asserting camera registration.
 *
 * The two planes may contain row padding or non-unit pixel stride. Absolute reads are relative
 * to each buffer's current position, matching Android [android.media.Image.Plane] semantics.
 */
object D45RawDepthPlaneDecoder {
    fun decode(
        depth: D45StridedBytePlane,
        confidence: D45StridedBytePlane,
        depthByteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN
    ): D45DecodedRawDepthRaster {
        require(depth.widthPx == confidence.widthPx && depth.heightPx == confidence.heightPx) {
            "raw depth and confidence dimensions must match"
        }
        require(depth.pixelStrideBytes >= Short.SIZE_BYTES) {
            "raw depth pixel stride must contain an unsigned 16-bit sample"
        }
        require(confidence.pixelStrideBytes >= Byte.SIZE_BYTES) {
            "raw confidence pixel stride must contain an unsigned 8-bit sample"
        }
        requirePlaneCapacity(depth, Short.SIZE_BYTES, "depth")
        requirePlaneCapacity(confidence, Byte.SIZE_BYTES, "confidence")

        val depthBuffer = depth.buffer.duplicate().order(depthByteOrder)
        val confidenceBuffer = confidence.buffer.duplicate()
        val depthBase = depthBuffer.position()
        val confidenceBase = confidenceBuffer.position()
        val depthValues = IntArray(depth.widthPx * depth.heightPx)
        val confidenceValues = FloatArray(depthValues.size)
        for (y in 0 until depth.heightPx) {
            for (x in 0 until depth.widthPx) {
                val outputIndex = y * depth.widthPx + x
                val depthOffset =
                    depthBase + y * depth.rowStrideBytes + x * depth.pixelStrideBytes
                val confidenceOffset =
                    confidenceBase +
                        y * confidence.rowStrideBytes +
                        x * confidence.pixelStrideBytes
                depthValues[outputIndex] = depthBuffer.getShort(depthOffset).toInt() and 0xFFFF
                confidenceValues[outputIndex] =
                    (confidenceBuffer.get(confidenceOffset).toInt() and 0xFF) / 255f
            }
        }
        return D45DecodedRawDepthRaster(
            widthPx = depth.widthPx,
            heightPx = depth.heightPx,
            depthMillimeters = depthValues,
            confidence = confidenceValues
        )
    }

    private fun requirePlaneCapacity(
        plane: D45StridedBytePlane,
        sampleBytes: Int,
        label: String
    ) {
        val lastExclusive =
            plane.buffer.position().toLong() +
                (plane.heightPx - 1L) * plane.rowStrideBytes +
                (plane.widthPx - 1L) * plane.pixelStrideBytes +
                sampleBytes
        require(lastExclusive <= plane.buffer.limit().toLong()) {
            "$label plane buffer is shorter than its declared dimensions and strides"
        }
        require(
            plane.rowStrideBytes >=
                (plane.widthPx - 1) * plane.pixelStrideBytes + sampleBytes
        ) {
            "$label row stride truncates its last logical sample"
        }
    }
}

enum class D45RawDepthRegistrationState {
    SOURCE_REGISTRATION_UNVERIFIED
}

/**
 * A decoded, timestamp-aligned raw raster that still cannot enter [MetricDepthTargetSampler].
 *
 * Registration is a type-level boundary: only a separately verified adapter may turn this into
 * [RegisteredMetricDepthFrame].
 */
data class D45UnregisteredRawMetricDepthFrame(
    val sourceFrame: FrameStamp,
    val sourceImageIntrinsics: MetricDepthCameraIntrinsics,
    val raster: D45DecodedRawDepthRaster,
    val depthTimestampNs: Long,
    val confidenceTimestampNs: Long,
    val producedAtNs: Long,
    val registrationState: D45RawDepthRegistrationState =
        D45RawDepthRegistrationState.SOURCE_REGISTRATION_UNVERIFIED
) {
    init {
        require(sourceFrame.clockDomain == FrameClockDomain.ANDROID_ELAPSED_REALTIME)
        require(depthTimestampNs == sourceFrame.capturedAtNs)
        require(confidenceTimestampNs == sourceFrame.capturedAtNs)
        require(producedAtNs >= sourceFrame.receivedAtNs)
    }
}
