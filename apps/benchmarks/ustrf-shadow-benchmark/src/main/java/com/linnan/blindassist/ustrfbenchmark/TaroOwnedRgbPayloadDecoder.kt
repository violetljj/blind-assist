package com.linnan.blindassist.ustrfbenchmark

import com.linnan.blindassist.hftf.metricdepth.D45Yuv420Image
import com.linnan.blindassist.hftf.metricdepth.D45Yuv420ToRgbaDecoder
import com.linnan.blindassist.hftf.metricdepth.D45YuvBytePlane
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import java.nio.ByteBuffer
import java.security.MessageDigest

data class TaroOwnedRgbDecodeReceipt(
    val sourceFrame: UstrfFrameStamp,
    val sourceYuvSha256: String,
    val rgbaSha256: String,
    val widthPx: Int,
    val heightPx: Int,
    val rgbaByteCount: Int
)

/** Delayed decoder for benchmark-owned YUV bytes; no android.media.Image is accepted or retained. */
class TaroOwnedRgbPayloadDecoder {
    fun decode(payload: TaroOwnedRgbPayload): TaroOwnedRgbDecodeReceipt {
        require(payload.imageFormat == YUV_420_888_FORMAT) {
            "expected YUV_420_888 but received ${payload.imageFormat}"
        }
        require(payload.planes.size == EXPECTED_YUV_PLANE_COUNT) {
            "expected three YUV planes but received ${payload.planes.size}"
        }
        val rgba = D45Yuv420ToRgbaDecoder.decode(
            D45Yuv420Image(
                widthPx = payload.imageWidthPx,
                heightPx = payload.imageHeightPx,
                y = payload.planes[0].asDecoderPlane(),
                u = payload.planes[1].asDecoderPlane(),
                v = payload.planes[2].asDecoderPlane()
            )
        )
        return TaroOwnedRgbDecodeReceipt(
            sourceFrame = payload.sourceFrame,
            sourceYuvSha256 = payload.contentSha256,
            rgbaSha256 = sha256(rgba.bytes),
            widthPx = rgba.widthPx,
            heightPx = rgba.heightPx,
            rgbaByteCount = rgba.bytes.size
        )
    }

    private fun TaroOwnedYuvPlane.asDecoderPlane() = D45YuvBytePlane(
        rowStrideBytes = rowStrideBytes,
        pixelStrideBytes = pixelStrideBytes,
        buffer = ByteBuffer.wrap(bytes).asReadOnlyBuffer()
    )

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { "%02x".format(it) }

    private companion object {
        const val EXPECTED_YUV_PLANE_COUNT = 3
        const val YUV_420_888_FORMAT = 35
    }
}
