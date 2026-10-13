package com.linnan.blindassist

import com.linnan.blindassist.capture.AtomCameraCaptureFrame
import com.linnan.blindassist.capture.ImuCaptureRecord

/** Receipt-time matching only: device clocks and exposure times remain independent.
 *
 * ToF decoding can lag the camera callback. Keep bounded histories so a future camera receipt
 * cannot hide the latest eligible past frame. Matching and epoch changes share the same lock;
 * a snapshot from before a stream reset must never acquire the new epoch.
 */
internal class MobileFusionInputBuffer(
    private val maxAgeNs: Long = 100_000_000L,
    private val cameraCapacity: Int = 16,
    private val imuCapacity: Int = 64,
) {
    data class Match(
        val epoch: Long,
        val image: AtomCameraCaptureFrame?,
        val imu: ImuCaptureRecord?,
    )

    private val cameraHistory = ArrayDeque<AtomCameraCaptureFrame>()
    private val imuHistory = ArrayDeque<ImuCaptureRecord>()
    private var currentEpoch = 0L
    private var retired = false
    val epoch: Long @Synchronized get() = currentEpoch

    init { require(maxAgeNs >= 0 && cameraCapacity > 0 && imuCapacity > 0) }

    @Synchronized
    fun addCamera(frame: AtomCameraCaptureFrame) {
        if (retired) return
        cameraHistory.addLast(frame)
        while (cameraHistory.size > cameraCapacity) cameraHistory.removeFirst()
    }

    @Synchronized
    fun addImu(record: ImuCaptureRecord) {
        if (retired) return
        val frame = record.frame ?: return
        val previous = imuHistory.lastOrNull()?.frame
        if (previous != null && (previous.bootId != frame.bootId || previous.deviceId != frame.deviceId)) {
            invalidate(clearInputs = true)
        }
        imuHistory.addLast(record)
        while (imuHistory.size > imuCapacity) imuHistory.removeFirst()
    }

    @Synchronized
    fun match(tofReceiptNs: Long): Match = Match(
        currentEpoch,
        cameraHistory.lastOrNull { tofReceiptNs - it.receivedElapsedNs in 0..maxAgeNs },
        imuHistory.lastOrNull { tofReceiptNs - it.hostReceivedMonotonicNs in 0..maxAgeNs },
    )

    /** Returns null when an obsolete worker tries to invalidate a newer input generation. */
    @Synchronized
    fun invalidate(expectedEpoch: Long? = null, clearInputs: Boolean = false): Long? {
        if (retired || expectedEpoch != null && expectedEpoch != currentEpoch) return null
        currentEpoch++
        if (clearInputs) { cameraHistory.clear(); imuHistory.clear() }
        return currentEpoch
    }

    /** Freeze the final epoch before publishing shutdown/failure, including late callbacks. */
    @Synchronized
    fun retire() {
        invalidate(clearInputs = true)
        retired = true
    }
}
