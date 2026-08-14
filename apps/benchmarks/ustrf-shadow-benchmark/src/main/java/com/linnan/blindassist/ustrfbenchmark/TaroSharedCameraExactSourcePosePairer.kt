package com.linnan.blindassist.ustrfbenchmark

import com.linnan.blindassist.ustrf.UstrfFrameStamp
import java.util.TreeMap

/** Owned app-surface bytes copied before the originating Image is closed. */
data class TaroSharedCameraOwnedYuvFrame(
    val timestampNs: Long,
    val imageWidthPx: Int,
    val imageHeightPx: Int,
    val imageFormat: Int,
    val planes: List<TaroOwnedYuvPlane>,
    val contentSha256: String
) {
    val byteCount: Long = planes.sumOf { it.bytes.size.toLong() }

    init {
        require(timestampNs >= 0L)
        require(imageWidthPx > 0 && imageHeightPx > 0)
        require(planes.isNotEmpty())
        require(byteCount > 0L)
        require(SHA256_REGEX.matches(contentSha256))
    }

    private companion object {
        val SHA256_REGEX = Regex("[0-9a-f]{64}")
    }
}

data class TaroSharedCameraPairingReceipt(
    val matchedPayload: TaroOwnedRgbPayload?,
    val ageEvictedImageCount: Int,
    val ageEvictedPoseCount: Int,
    val byteCapEvictedImageCount: Int,
    val staleInputRejected: Boolean,
    val duplicateInputRejected: Boolean
)

data class TaroSharedCameraPairerSnapshot(
    val exactMatchCount: Int,
    val pendingImageCount: Int,
    val pendingPoseCount: Int,
    val pendingImageBytes: Long,
    val ageEvictedImageCount: Int,
    val ageEvictedPoseCount: Int,
    val byteCapEvictedImageCount: Int,
    val staleInputRejectedCount: Int,
    val duplicateInputRejectedCount: Int
)

data class TaroSharedCameraPairerResetReceipt(
    val evictedImageCount: Int,
    val evictedPoseCount: Int,
    val evictedImageBytes: Long
)

/**
 * Exact timestamp join for an app-owned SharedCamera YUV stream and ARCore pose admissions.
 *
 * Inputs may arrive in either order on different threads. A payload is emitted only when
 * Image.timestamp equals the complete pose source timestamp. There is deliberately no nearest,
 * frame-id-only, or cross-epoch fallback. Pending data is bounded by source age and owned bytes.
 */
class TaroSharedCameraExactSourcePosePairer(
    private val maximumPendingAgeNs: Long = 1_000_000_000L,
    private val maximumPendingImageBytes: Long = 32L * 1024L * 1024L
) {
    private val imagesByTimestamp = TreeMap<Long, TaroSharedCameraOwnedYuvFrame>()
    private val posesByTimestamp = TreeMap<Long, PoseAdmission>()
    private var watermarkNs = -1L
    private var pendingImageBytesInternal = 0L
    private var exactMatchCount = 0
    private var ageEvictedImageCount = 0
    private var ageEvictedPoseCount = 0
    private var byteCapEvictedImageCount = 0
    private var staleInputRejectedCount = 0
    private var duplicateInputRejectedCount = 0

    init {
        require(maximumPendingAgeNs > 0L)
        require(maximumPendingImageBytes > 0L)
    }

    @Synchronized
    fun observeImage(image: TaroSharedCameraOwnedYuvFrame): TaroSharedCameraPairingReceipt {
        val evictions = advanceWatermark(image.timestampNs)
        if (isStale(image.timestampNs)) {
            staleInputRejectedCount++
            return evictions.toReceipt(staleInputRejected = true)
        }
        if (imagesByTimestamp.containsKey(image.timestampNs)) {
            duplicateInputRejectedCount++
            return evictions.toReceipt(duplicateInputRejected = true)
        }
        imagesByTimestamp[image.timestampNs] = image
        pendingImageBytesInternal += image.byteCount
        val match = exactMatch(image.timestampNs)
        var byteEvictions = 0
        while (pendingImageBytesInternal > maximumPendingImageBytes && imagesByTimestamp.isNotEmpty()) {
            removeOldestImage()
            byteEvictions++
            byteCapEvictedImageCount++
        }
        return evictions.toReceipt(match, byteCapEvictedImageCount = byteEvictions)
    }

    @Synchronized
    fun observePose(
        sourceFrame: UstrfFrameStamp,
        admission: TaroArCoreAnchorPoseAdmission.Available
    ): TaroSharedCameraPairingReceipt {
        require(sourceFrame.capturedAtNs == admission.cameraPose.timestampNs)
        require(sourceFrame.coordinateFrame == admission.cameraPose.cameraFrame)
        val timestampNs = sourceFrame.capturedAtNs
        val evictions = advanceWatermark(timestampNs)
        if (isStale(timestampNs)) {
            staleInputRejectedCount++
            return evictions.toReceipt(staleInputRejected = true)
        }
        if (posesByTimestamp.containsKey(timestampNs)) {
            duplicateInputRejectedCount++
            return evictions.toReceipt(duplicateInputRejected = true)
        }
        posesByTimestamp[timestampNs] = PoseAdmission(sourceFrame, admission)
        return evictions.toReceipt(exactMatch(timestampNs))
    }

    @Synchronized
    fun reset(): TaroSharedCameraPairerResetReceipt {
        val receipt = TaroSharedCameraPairerResetReceipt(
            evictedImageCount = imagesByTimestamp.size,
            evictedPoseCount = posesByTimestamp.size,
            evictedImageBytes = pendingImageBytesInternal
        )
        imagesByTimestamp.clear()
        posesByTimestamp.clear()
        pendingImageBytesInternal = 0L
        watermarkNs = -1L
        return receipt
    }

    @Synchronized
    fun snapshot(): TaroSharedCameraPairerSnapshot = TaroSharedCameraPairerSnapshot(
        exactMatchCount = exactMatchCount,
        pendingImageCount = imagesByTimestamp.size,
        pendingPoseCount = posesByTimestamp.size,
        pendingImageBytes = pendingImageBytesInternal,
        ageEvictedImageCount = ageEvictedImageCount,
        ageEvictedPoseCount = ageEvictedPoseCount,
        byteCapEvictedImageCount = byteCapEvictedImageCount,
        staleInputRejectedCount = staleInputRejectedCount,
        duplicateInputRejectedCount = duplicateInputRejectedCount
    )

    private fun exactMatch(timestampNs: Long): TaroOwnedRgbPayload? {
        val image = imagesByTimestamp[timestampNs] ?: return null
        val pose = posesByTimestamp[timestampNs] ?: return null
        imagesByTimestamp.remove(timestampNs)
        posesByTimestamp.remove(timestampNs)
        pendingImageBytesInternal -= image.byteCount
        exactMatchCount++
        return TaroOwnedRgbPayload(
            sourceFrame = pose.sourceFrame,
            anchorPose = pose.admission.cameraPose,
            imageWidthPx = image.imageWidthPx,
            imageHeightPx = image.imageHeightPx,
            imageFormat = image.imageFormat,
            planes = image.planes,
            contentSha256 = image.contentSha256
        )
    }

    private fun advanceWatermark(timestampNs: Long): Evictions {
        watermarkNs = maxOf(watermarkNs, timestampNs)
        val oldestAllowedNs = (watermarkNs - maximumPendingAgeNs).coerceAtLeast(0L)
        var images = 0
        var poses = 0
        while (imagesByTimestamp.firstKeyOrNull()?.let { it < oldestAllowedNs } == true) {
            removeOldestImage()
            images++
            ageEvictedImageCount++
        }
        while (posesByTimestamp.firstKeyOrNull()?.let { it < oldestAllowedNs } == true) {
            posesByTimestamp.pollFirstEntry()
            poses++
            ageEvictedPoseCount++
        }
        return Evictions(images, poses)
    }

    private fun isStale(timestampNs: Long): Boolean =
        watermarkNs >= maximumPendingAgeNs && timestampNs < watermarkNs - maximumPendingAgeNs

    private fun removeOldestImage() {
        val removed = imagesByTimestamp.pollFirstEntry()?.value ?: return
        pendingImageBytesInternal -= removed.byteCount
    }

    private fun <V> TreeMap<Long, V>.firstKeyOrNull(): Long? =
        if (isEmpty()) null else firstKey()

    private data class PoseAdmission(
        val sourceFrame: UstrfFrameStamp,
        val admission: TaroArCoreAnchorPoseAdmission.Available
    )

    private data class Evictions(val images: Int, val poses: Int) {
        fun toReceipt(
            matchedPayload: TaroOwnedRgbPayload? = null,
            byteCapEvictedImageCount: Int = 0,
            staleInputRejected: Boolean = false,
            duplicateInputRejected: Boolean = false
        ) = TaroSharedCameraPairingReceipt(
            matchedPayload = matchedPayload,
            ageEvictedImageCount = images,
            ageEvictedPoseCount = poses,
            byteCapEvictedImageCount = byteCapEvictedImageCount,
            staleInputRejected = staleInputRejected,
            duplicateInputRejected = duplicateInputRejected
        )
    }
}
