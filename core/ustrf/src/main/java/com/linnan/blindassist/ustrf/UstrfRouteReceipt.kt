package com.linnan.blindassist.ustrf

/**
 * Slow-loop route/goal receipt. It can propose a short-horizon offset, but it cannot sign a
 * USTRF safety action and is rejected unless it is bound to the fast-loop source frame.
 */
data class UstrfRouteReceipt(
    val queryFrame: UstrfFrameStamp,
    val issuedAtNs: Long,
    val validUntilNs: Long,
    val coordinateFrame: String,
    val desiredOffsetCells: Int,
    val confidence: Float,
    val source: String
) {
    init {
        require(issuedAtNs >= queryFrame.capturedAtNs)
        require(validUntilNs >= issuedAtNs)
        require(coordinateFrame.isNotBlank())
        require(confidence in 0f..1f)
        require(source.isNotBlank())
    }
}

enum class UstrfRouteReceiptFailure {
    SOURCE_FRAME_MISMATCH,
    COORDINATE_FRAME_MISMATCH,
    ISSUED_IN_FUTURE,
    STALE,
    LOW_CONFIDENCE,
    OFFSET_OUT_OF_RANGE
}

sealed interface UstrfRouteReceiptResolution {
    data class Available(val route: UstrfRouteIntent) : UstrfRouteReceiptResolution
    data class Unavailable(val failure: UstrfRouteReceiptFailure) : UstrfRouteReceiptResolution
}

/** Resolves slow-loop data into a bounded fast-loop route candidate without granting control authority. */
class UstrfRouteReceiptResolver(
    private val minimumConfidence: Float = .70f,
    private val maximumAbsoluteOffsetCells: Int = 2
) {
    init {
        require(minimumConfidence in 0f..1f)
        require(maximumAbsoluteOffsetCells >= 0)
    }

    fun resolve(
        receipt: UstrfRouteReceipt,
        frame: UstrfFrameStamp,
        decisionAtNs: Long
    ): UstrfRouteReceiptResolution {
        require(decisionAtNs >= frame.capturedAtNs)
        val failure = when {
            receipt.queryFrame != frame -> UstrfRouteReceiptFailure.SOURCE_FRAME_MISMATCH
            receipt.coordinateFrame != frame.coordinateFrame -> UstrfRouteReceiptFailure.COORDINATE_FRAME_MISMATCH
            receipt.issuedAtNs > decisionAtNs -> UstrfRouteReceiptFailure.ISSUED_IN_FUTURE
            receipt.validUntilNs < decisionAtNs -> UstrfRouteReceiptFailure.STALE
            receipt.confidence < minimumConfidence -> UstrfRouteReceiptFailure.LOW_CONFIDENCE
            kotlin.math.abs(receipt.desiredOffsetCells) > maximumAbsoluteOffsetCells -> UstrfRouteReceiptFailure.OFFSET_OUT_OF_RANGE
            else -> null
        }
        return if (failure == null) UstrfRouteReceiptResolution.Available(
            UstrfRouteIntent(receipt.coordinateFrame, receipt.desiredOffsetCells, receipt.confidence, receipt.validUntilNs)
        ) else UstrfRouteReceiptResolution.Unavailable(failure)
    }
}
