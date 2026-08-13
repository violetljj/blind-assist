package com.linnan.blindassist.ustrfbenchmark

import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import kotlin.math.abs
import kotlin.math.floor

data class TaroPositiveVisualToken(
    val label: String,
    val columnBin: Int,
    val rowBin: Int
)

data class TaroPositiveVisualReceipt(
    val sourceFrame: UstrfFrameStamp,
    val tokens: Set<TaroPositiveVisualToken>,
    val focusedTokens: Set<TaroPositiveVisualToken>,
    val decodeLatencyMs: Double,
    val detectorPreprocessLatencyMs: Long,
    val detectorInferenceLatencyMs: Long,
    val detectorPostprocessLatencyMs: Long,
    val detectorTotalLatencyMs: Long
)

data class TaroPositiveVisualTokenization(
    val tokens: Set<TaroPositiveVisualToken>,
    val focusedTokens: Set<TaroPositiveVisualToken>
) {
    init {
        require(focusedTokens.all(tokens::contains))
    }
}

data class TaroPositiveVisualArmComparison(
    val currentFocusedTokenCount: Int,
    val passiveNewFocusedTokenCount: Int,
    val poseDiverseNewFocusedTokenCount: Int,
    val currentAllTokenCount: Int,
    val passiveNewAllTokenCount: Int,
    val poseDiverseNewAllTokenCount: Int
)

object TaroPositiveVisualEvidence {
    const val PASSIVE_TARGET_GAP_NS = 500_000_000L
    const val MINIMUM_GAP_NS = 150_000_000L
    const val MAXIMUM_GAP_NS = 1_000_000_000L

    fun selectPassive(
        referenceFrame: UstrfFrameStamp,
        candidates: List<TaroOwnedRgbPayload>
    ): TaroOwnedRgbPayload? = candidates
        .asSequence()
        .mapNotNull { payload ->
            val gapNs = referenceFrame.capturedAtNs - payload.sourceFrame.capturedAtNs
            payload.takeIf { gapNs in MINIMUM_GAP_NS..MAXIMUM_GAP_NS }?.let {
                PassiveCandidate(it, gapNs)
            }
        }
        .minWithOrNull(
            compareBy<PassiveCandidate> { abs(it.gapNs - PASSIVE_TARGET_GAP_NS) }
                .thenBy { it.gapNs }
                .thenBy { it.payload.sourceFrame.frameId }
        )
        ?.payload

    fun tokens(detections: List<Detection>): TaroPositiveVisualTokenization {
        val tokens = linkedSetOf<TaroPositiveVisualToken>()
        val focusedTokens = linkedSetOf<TaroPositiveVisualToken>()
        detections.asSequence()
            .filter { it.source == DetectionSource.OBJECT_DETECTOR }
            .forEach { detection ->
            val frameWidth = detection.frameSize.width.toFloat().coerceAtLeast(1f)
            val frameHeight = detection.frameSize.height.toFloat().coerceAtLeast(1f)
            val box = detection.boundingBox.clamped(detection.frameSize)
            val normalizedCenterX = (box.centerX / frameWidth).coerceIn(0f, 1f)
            val normalizedCenterY = (box.centerY / frameHeight).coerceIn(0f, 1f)
            val token = TaroPositiveVisualToken(
                label = detection.label,
                columnBin = bin3(normalizedCenterX),
                rowBin = bin3(normalizedCenterY)
            )
            tokens += token
            if (
                box.right / frameWidth >= FOCUS_LEFT &&
                    box.left / frameWidth <= FOCUS_RIGHT &&
                    box.bottom / frameHeight >= FOCUS_TOP &&
                    box.top / frameHeight <= FOCUS_BOTTOM
            ) {
                focusedTokens += token
            }
        }
        return TaroPositiveVisualTokenization(tokens, focusedTokens)
    }

    fun compare(
        current: TaroPositiveVisualReceipt,
        passive: TaroPositiveVisualReceipt,
        poseDiverse: TaroPositiveVisualReceipt
    ): TaroPositiveVisualArmComparison {
        val currentFocused = current.focusedTokens
        return TaroPositiveVisualArmComparison(
            currentFocusedTokenCount = currentFocused.size,
            passiveNewFocusedTokenCount = (passive.focusedTokens - currentFocused).size,
            poseDiverseNewFocusedTokenCount = (poseDiverse.focusedTokens - currentFocused).size,
            currentAllTokenCount = current.tokens.size,
            passiveNewAllTokenCount = (passive.tokens - current.tokens).size,
            poseDiverseNewAllTokenCount = (poseDiverse.tokens - current.tokens).size
        )
    }

    private fun bin3(normalized: Float): Int =
        floor(normalized.coerceIn(0f, .999999f) * GRID_BIN_COUNT).toInt()

    private data class PassiveCandidate(val payload: TaroOwnedRgbPayload, val gapNs: Long)

    private const val GRID_BIN_COUNT = 3
    private const val FOCUS_LEFT = .25f
    private const val FOCUS_RIGHT = .75f
    private const val FOCUS_TOP = .50f
    private const val FOCUS_BOTTOM = 1f
}
