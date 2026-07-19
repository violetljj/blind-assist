package com.linnan.blindassist.benchmark

import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AndroidExplicitRoutePayloadProviderTest {
    @Test
    fun validExternalPayloadFeedsGeometryWithoutFutureInput() {
        val frame = AndroidExplicitRoutePayloadProvider.parse(
            validPayload(), 1_000, 1_000, 1_000,
            listOf(ExplicitRouteObstacleBox(0.49, 0.78, 0.51, 0.82))
        )
        val score = ExplicitRouteGeometryFusion.score(frame, obstacleExpansionObjectHeights = 0.0)

        assertTrue(score.routeValid)
        assertEquals(1.0 / 3.0, score.intersectionFraction!!, 1e-12)
    }

    @Test
    fun missingProviderOrProjectionReceiptFailsClosed() {
        val missingProvider = validPayload().also {
            it.removeExtra(AndroidExplicitRoutePayloadProvider.EXTRA_PROVIDER_ID)
        }
        val missingReceipt = validPayload().also {
            it.removeExtra(AndroidExplicitRoutePayloadProvider.EXTRA_PROJECTION_RECEIPT_ID)
        }

        listOf(missingProvider, missingReceipt).forEach { assertClosed(it) }
    }

    @Test
    fun futureIssuedExpiredOrOverlongPayloadFailsClosed() {
        val future = validPayload().putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_ISSUED_AT_MS, 1_001L)
        val expired = validPayload().putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_VALID_UNTIL_MS, 999L)
        val overlong = validPayload().putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_VALID_UNTIL_MS, 2_001L)

        listOf(future, expired, overlong).forEach { assertClosed(it) }
    }

    @Test
    fun riskModelOrFutureVideoProviderFailsClosed() {
        val model = validPayload().putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_INFERRED_BY_RISK_MODEL, true)
        val futureVideo = validPayload().putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_USES_FUTURE_VIDEO, true)

        listOf(model, futureVideo).forEach { assertClosed(it) }
    }

    @Test
    fun lowConfidenceInvalidRouteOrMalformedWaypointsFailClosed() {
        val lowConfidence = validPayload().putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_CONFIDENCE, 0.49)
        val invalidRoute = validPayload().putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_ROUTE_VALID, false)
        val malformed = validPayload().putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_X_NORM, doubleArrayOf(0.5, 0.5))

        listOf(lowConfidence, invalidRoute, malformed).forEach { assertClosed(it) }
    }

    @Test
    fun twoPayloadsCanOpenExistingLifecycle() {
        val obstacle = listOf(ExplicitRouteObstacleBox(0.49, 0.78, 0.51, 0.82))
        val samples = listOf(1_000L, 2_000L).map { timestamp ->
            val payload = validPayload(timestamp - 500, timestamp + 500)
            ExplicitRouteGeometryFusion.score(
                AndroidExplicitRoutePayloadProvider.parse(payload, timestamp, 1_000, 1_000, obstacle),
                obstacleExpansionObjectHeights = 0.0
            )
        }

        assertEquals(
            listOf(ExplicitRouteRiskTransition(ExplicitRouteRiskState.INTERVENTION_NEEDED, 2_000)),
            ExplicitRouteIntentFusion.decode(samples)
        )
    }

    private fun assertClosed(payload: Intent) {
        val result = ExplicitRouteGeometryFusion.score(
            AndroidExplicitRoutePayloadProvider.parse(payload, 1_000, 1_000, 1_000, emptyList())
        )
        assertFalse(result.routeValid)
        assertNull(result.intersectionFraction)
    }

    private fun validPayload(issuedAtMs: Long = 500, validUntilMs: Long = 1_500) =
        Intent(AndroidExplicitRoutePayloadProvider.ACTION)
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_PROVIDER_ID, "external_navigation_test")
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_PROJECTION_RECEIPT_ID, "projection_receipt_test")
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_ISSUED_AT_MS, issuedAtMs)
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_VALID_UNTIL_MS, validUntilMs)
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_CONFIDENCE, 1.0)
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_ROUTE_VALID, true)
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_INFERRED_BY_RISK_MODEL, false)
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_USES_FUTURE_VIDEO, false)
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_HORIZONS_MS, longArrayOf(1_000, 2_000, 3_000))
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_X_NORM, doubleArrayOf(0.5, 0.5, 0.5))
            .putExtra(AndroidExplicitRoutePayloadProvider.EXTRA_Y_NORM, doubleArrayOf(0.92, 0.86, 0.80))
}
