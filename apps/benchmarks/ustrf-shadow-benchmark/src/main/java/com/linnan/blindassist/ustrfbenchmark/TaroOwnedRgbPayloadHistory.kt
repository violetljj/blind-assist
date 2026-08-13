package com.linnan.blindassist.ustrfbenchmark

import com.linnan.blindassist.ustrf.TaroBufferedPoseFrame
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfPoseSample

data class TaroOwnedYuvPlane(
    val rowStrideBytes: Int,
    val pixelStrideBytes: Int,
    val bytes: ByteArray
) {
    init {
        require(rowStrideBytes > 0)
        require(pixelStrideBytes > 0)
        require(bytes.isNotEmpty())
    }
}

data class TaroOwnedRgbPayloadReceipt(
    val sourceFrame: UstrfFrameStamp,
    val anchorPose: UstrfPoseSample,
    val contentSha256: String,
    val byteCount: Long,
    val imageWidthPx: Int,
    val imageHeightPx: Int,
    val imageFormat: Int,
    val planeCount: Int
)

/** Benchmark-owned YUV bytes whose lifetime is independent of android.media.Image. */
class TaroOwnedRgbPayload(
    val sourceFrame: UstrfFrameStamp,
    val anchorPose: UstrfPoseSample,
    val imageWidthPx: Int,
    val imageHeightPx: Int,
    val imageFormat: Int,
    val planes: List<TaroOwnedYuvPlane>,
    val contentSha256: String
) {
    val byteCount: Long = planes.sumOf { it.bytes.size.toLong() }

    val receipt: TaroOwnedRgbPayloadReceipt = TaroOwnedRgbPayloadReceipt(
        sourceFrame = sourceFrame,
        anchorPose = anchorPose,
        contentSha256 = contentSha256,
        byteCount = byteCount,
        imageWidthPx = imageWidthPx,
        imageHeightPx = imageHeightPx,
        imageFormat = imageFormat,
        planeCount = planes.size
    )

    init {
        require(anchorPose.timestampNs == sourceFrame.capturedAtNs)
        require(anchorPose.cameraFrame == sourceFrame.coordinateFrame)
        require(imageWidthPx > 0 && imageHeightPx > 0)
        require(planes.isNotEmpty())
        require(byteCount > 0L)
        require(SHA256_REGEX.matches(contentSha256))
    }

    internal fun asBufferedPoseFrame(): TaroBufferedPoseFrame =
        TaroBufferedPoseFrame(sourceFrame, anchorPose)

    private companion object {
        val SHA256_REGEX = Regex("[0-9a-f]{64}")
    }
}

data class TaroRgbHistoryMutationReceipt(
    val ageEvictionCount: Int,
    val byteCapEvictionCount: Int,
    val retainedEntryCount: Int,
    val retainedBytes: Long
)

data class TaroRgbHistoryResetReceipt(
    val evictedEntryCount: Int,
    val evictedBytes: Long
)

/**
 * Benchmark-only, exact-identity history for copied RGB payloads.
 *
 * The history is bounded by both source-clock age and owned bytes. Lookups accept the complete
 * [UstrfFrameStamp]; there is deliberately no timestamp-nearest or frame-id-only fallback.
 */
class TaroOwnedRgbPayloadHistory(
    private val maximumRetainedAgeNs: Long = 1_000_000_000L,
    private val maximumRetainedBytes: Long = 32L * 1024L * 1024L
) : AutoCloseable {
    private val payloads = linkedMapOf<UstrfFrameStamp, TaroOwnedRgbPayload>()
    private var lastAppendedTimestampNs = -1L

    var retainedBytes: Long = 0L
        private set

    val retainedEntryCount: Int
        get() = payloads.size

    val oldestRetainedTimestampNs: Long?
        get() = payloads.values.firstOrNull()?.sourceFrame?.capturedAtNs

    init {
        require(maximumRetainedAgeNs > 0L)
        require(maximumRetainedBytes > 0L)
    }

    fun advanceTo(referenceTimestampNs: Long): TaroRgbHistoryMutationReceipt {
        require(referenceTimestampNs >= 0L)
        val oldestAllowedTimestampNs = (referenceTimestampNs - maximumRetainedAgeNs).coerceAtLeast(0L)
        var ageEvictionCount = 0
        while (payloads.values.firstOrNull()?.sourceFrame?.capturedAtNs?.let {
                it < oldestAllowedTimestampNs
            } == true
        ) {
            removeOldest()
            ageEvictionCount++
        }
        return mutationReceipt(ageEvictionCount, byteCapEvictionCount = 0)
    }

    fun append(payload: TaroOwnedRgbPayload): TaroRgbHistoryMutationReceipt {
        require(payload.sourceFrame.capturedAtNs > lastAppendedTimestampNs) {
            "payload source timestamps must be strictly increasing"
        }
        require(payload.byteCount <= maximumRetainedBytes) {
            "one payload exceeds the complete history byte cap"
        }
        val ageReceipt = advanceTo(payload.sourceFrame.capturedAtNs)
        check(payloads.put(payload.sourceFrame, payload) == null) {
            "duplicate source-frame identity"
        }
        retainedBytes += payload.byteCount
        lastAppendedTimestampNs = payload.sourceFrame.capturedAtNs
        var byteCapEvictionCount = 0
        while (retainedBytes > maximumRetainedBytes) {
            removeOldest()
            byteCapEvictionCount++
        }
        return mutationReceipt(ageReceipt.ageEvictionCount, byteCapEvictionCount)
    }

    fun lookupExact(sourceFrame: UstrfFrameStamp): TaroOwnedRgbPayload? = payloads[sourceFrame]

    fun bufferedPoseFrames(): List<TaroBufferedPoseFrame> =
        payloads.values.map(TaroOwnedRgbPayload::asBufferedPoseFrame)

    fun reset(): TaroRgbHistoryResetReceipt {
        val receipt = TaroRgbHistoryResetReceipt(payloads.size, retainedBytes)
        payloads.clear()
        retainedBytes = 0L
        lastAppendedTimestampNs = -1L
        return receipt
    }

    override fun close() {
        reset()
    }

    private fun removeOldest() {
        val oldest = payloads.entries.firstOrNull() ?: return
        retainedBytes -= oldest.value.byteCount
        payloads.remove(oldest.key)
    }

    private fun mutationReceipt(ageEvictionCount: Int, byteCapEvictionCount: Int) =
        TaroRgbHistoryMutationReceipt(
            ageEvictionCount = ageEvictionCount,
            byteCapEvictionCount = byteCapEvictionCount,
            retainedEntryCount = payloads.size,
            retainedBytes = retainedBytes
        )
}
