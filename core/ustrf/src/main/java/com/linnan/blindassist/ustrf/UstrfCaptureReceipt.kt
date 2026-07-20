package com.linnan.blindassist.ustrf

/**
 * Camera Adapter receipt. `frame.capturedAtNs` is already mapped into the USTRF monotonic clock;
 * `hardwareTimestampNs` is retained for audit and rollback detection, not compared across clocks.
 */
data class UstrfCaptureReceipt(
    val frame: UstrfFrameStamp,
    val hardwareTimestampNs: Long,
    val receivedAtNs: Long,
    val cameraClockDomain: String,
    val calibrationVersion: String
) {
    init {
        require(hardwareTimestampNs >= 0L)
        require(receivedAtNs >= frame.capturedAtNs) { "receipt cannot arrive before mapped capture" }
        require(cameraClockDomain.isNotBlank())
        require(calibrationVersion.isNotBlank())
    }
}

enum class UstrfCaptureReceiptFailure {
    RECEIVED_AFTER_DECISION,
    CAPTURE_AGE_EXCEEDED,
    FRAME_ID_ROLLBACK,
    CAPTURE_TIME_ROLLBACK,
    COORDINATE_FRAME_CHANGED,
    HARDWARE_TIMESTAMP_ROLLBACK,
    CAMERA_CLOCK_DOMAIN_CHANGED,
    CALIBRATION_CHANGED
}

sealed interface UstrfCaptureReceiptValidation {
    data object Valid : UstrfCaptureReceiptValidation
    data class Unavailable(val failure: UstrfCaptureReceiptFailure) : UstrfCaptureReceiptValidation
}

/**
 * Stateful per-camera receipt validator. It deliberately does not estimate a cross-sensor clock
 * transform; that is a future Android Adapter responsibility with measured calibration evidence.
 */
class UstrfCaptureReceiptValidator(private val maximumCaptureAgeNs: Long = 250_000_000L) {
    private var previous: UstrfCaptureReceipt? = null

    init { require(maximumCaptureAgeNs > 0L) }

    fun validate(receipt: UstrfCaptureReceipt, decisionAtNs: Long): UstrfCaptureReceiptValidation {
        require(decisionAtNs >= receipt.frame.capturedAtNs)
        val failure = when {
            receipt.receivedAtNs > decisionAtNs -> UstrfCaptureReceiptFailure.RECEIVED_AFTER_DECISION
            decisionAtNs - receipt.frame.capturedAtNs > maximumCaptureAgeNs -> UstrfCaptureReceiptFailure.CAPTURE_AGE_EXCEEDED
            previous?.let { receipt.frame.frameId <= it.frame.frameId } == true -> UstrfCaptureReceiptFailure.FRAME_ID_ROLLBACK
            previous?.let { receipt.frame.capturedAtNs <= it.frame.capturedAtNs } == true -> UstrfCaptureReceiptFailure.CAPTURE_TIME_ROLLBACK
            previous?.let { receipt.frame.coordinateFrame != it.frame.coordinateFrame } == true -> UstrfCaptureReceiptFailure.COORDINATE_FRAME_CHANGED
            previous?.let { receipt.hardwareTimestampNs <= it.hardwareTimestampNs } == true -> UstrfCaptureReceiptFailure.HARDWARE_TIMESTAMP_ROLLBACK
            previous?.let { receipt.cameraClockDomain != it.cameraClockDomain } == true -> UstrfCaptureReceiptFailure.CAMERA_CLOCK_DOMAIN_CHANGED
            previous?.let { receipt.calibrationVersion != it.calibrationVersion } == true -> UstrfCaptureReceiptFailure.CALIBRATION_CHANGED
            else -> null
        }
        return if (failure == null) UstrfCaptureReceiptValidation.Valid.also { previous = receipt }
        else UstrfCaptureReceiptValidation.Unavailable(failure)
    }

    fun reset() { previous = null }
}
