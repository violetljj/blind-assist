package com.linnan.blindassist.benchmark

import android.content.Intent

/** Benchmark-only parser for an external, non-future route payload. */
internal object AndroidExplicitRoutePayloadProvider {
    const val ACTION = "com.linnan.blindassist.action.EXPLICIT_ROUTE_INTENT"
    const val EXTRA_PROVIDER_ID = "provider_id"
    const val EXTRA_PROJECTION_RECEIPT_ID = "projection_receipt_id"
    const val EXTRA_ISSUED_AT_MS = "issued_at_ms"
    const val EXTRA_VALID_UNTIL_MS = "valid_until_ms"
    const val EXTRA_CONFIDENCE = "confidence"
    const val EXTRA_ROUTE_VALID = "route_valid"
    const val EXTRA_INFERRED_BY_RISK_MODEL = "inferred_by_risk_model"
    const val EXTRA_USES_FUTURE_VIDEO = "uses_future_video"
    const val EXTRA_HORIZONS_MS = "horizons_ms"
    const val EXTRA_X_NORM = "x_norm"
    const val EXTRA_Y_NORM = "y_norm"

    private const val MAXIMUM_AGE_MS = 1_000L
    private const val MINIMUM_CONFIDENCE = 0.5

    fun parse(
        payload: Intent,
        frameTimestampMs: Long,
        frameWidthPx: Int,
        frameHeightPx: Int,
        obstacles: List<ExplicitRouteObstacleBox>
    ): ExplicitRouteFrameInput {
        val invalid = {
            ExplicitRouteFrameInput(
                timestampMs = frameTimestampMs,
                validUntilTimestampMs = frameTimestampMs,
                confidence = 0.0,
                routeValid = false,
                frameWidthPx = frameWidthPx,
                frameHeightPx = frameHeightPx,
                waypoints = emptyList(),
                obstacles = obstacles
            )
        }
        if (payload.action != ACTION) return invalid()
        val providerId = payload.getStringExtra(EXTRA_PROVIDER_ID)
        val receiptId = payload.getStringExtra(EXTRA_PROJECTION_RECEIPT_ID)
        val issuedAt = payload.getLongExtra(EXTRA_ISSUED_AT_MS, Long.MIN_VALUE)
        val validUntil = payload.getLongExtra(EXTRA_VALID_UNTIL_MS, Long.MIN_VALUE)
        val confidence = payload.getDoubleExtra(EXTRA_CONFIDENCE, Double.NaN)
        val routeValid = payload.getBooleanExtra(EXTRA_ROUTE_VALID, false)
        val inferredByRiskModel = payload.getBooleanExtra(EXTRA_INFERRED_BY_RISK_MODEL, true)
        val usesFutureVideo = payload.getBooleanExtra(EXTRA_USES_FUTURE_VIDEO, true)
        val horizons = payload.getLongArrayExtra(EXTRA_HORIZONS_MS)
        val x = payload.getDoubleArrayExtra(EXTRA_X_NORM)
        val y = payload.getDoubleArrayExtra(EXTRA_Y_NORM)
        val timeValid = issuedAt != Long.MIN_VALUE && validUntil != Long.MIN_VALUE &&
            issuedAt <= frameTimestampMs && frameTimestampMs <= validUntil &&
            frameTimestampMs - issuedAt <= MAXIMUM_AGE_MS && validUntil - issuedAt <= MAXIMUM_AGE_MS
        val arraysValid = horizons?.size == 3 && x?.size == 3 && y?.size == 3
        if (providerId.isNullOrBlank() || receiptId.isNullOrBlank() || !timeValid ||
            !confidence.isFinite() || confidence < MINIMUM_CONFIDENCE || !routeValid ||
            inferredByRiskModel || usesFutureVideo || !arraysValid
        ) return invalid()

        val waypoints = horizons!!.indices.map { index ->
            ExplicitRouteWaypoint(horizons[index], x!![index], y!![index])
        }
        return ExplicitRouteFrameInput(
            timestampMs = frameTimestampMs,
            validUntilTimestampMs = validUntil,
            confidence = confidence,
            routeValid = true,
            frameWidthPx = frameWidthPx,
            frameHeightPx = frameHeightPx,
            waypoints = waypoints,
            obstacles = obstacles
        )
    }
}
