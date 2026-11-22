package com.linnan.blindassist.egoclear.demo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.View
import com.linnan.blindassist.egoclear.EgoClearStreamingEngine
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Live Egocentric BEV Static/Dynamic Mini-Radar + 24-bin Monotonic Distance-Risk Curve overlay.
 */
class EgoClearOverlayView(context: Context) : View(context) {

    data class OverlayState(
        val corrProbs: FloatArray,
        val staticProbGrid: FloatArray,   // BEV_NZ * BEV_NX (max over LOW/BODY/HEAD)
        val dynamicProbGrid: FloatArray,  // BEV_NZ * BEV_NX
        val headHazardProb: Float,
        val pHorizon: Float,
        val horizonMeters: Float,
        val horizonIdx: Int,
        val alertActive: Boolean,
        val alertThreshold: Float,
        val estimatedHitMeters: Float?,   // null if clear within active horizon
        val pitchGated: Boolean = false,
        val isLandscapeGravity: Boolean = false,
        val bypassAdvice: EgoClearStreamingEngine.BypassAdvice = EgoClearStreamingEngine.BypassAdvice.CLEAR,
        val dropOffHitMeters: Float? = null,
        val headHitMeters: Float? = null,
        val dynClosingHitMeters: Float? = null,
        val isUnknownUnobservable: Boolean = false,
    )

    private var state: OverlayState? = null
    private val dp = resources.displayMetrics.density

    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f * dp
    }
    private val panelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(185, 14, 18, 26)
    }
    private val panelStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.2f * dp
        color = Color.argb(120, 255, 255, 255)
    }
    private val cellPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val corridorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * dp
        color = Color.argb(210, 255, 255, 255)
        pathEffect = DashPathEffect(floatArrayOf(5f * dp, 4f * dp), 0f)
    }
    private val gridLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 0.8f * dp
        color = Color.argb(70, 200, 215, 235)
    }
    private val curveStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.6f * dp
    }
    private val curveFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val thrLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.2f * dp
        color = Color.argb(200, 255, 204, 0)
        pathEffect = DashPathEffect(floatArrayOf(6f * dp, 4f * dp), 0f)
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 11.5f * dp
    }
    private val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 13.5f * dp
        isFakeBoldText = true
    }
    private val heroPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 20f * dp
        isFakeBoldText = true
    }

    private val rect = RectF()
    private val curvePath = Path()
    private val fillPath = Path()

    fun updateState(newState: OverlayState?) {
        state = newState
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val s = state ?: return
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        // 1. Alert vignette border around screen
        if (!s.pitchGated) {
            if (s.isUnknownUnobservable) {
                borderPaint.color = Color.argb(190, 142, 142, 147)
                canvas.drawRect(3f * dp, 3f * dp, w - 3f * dp, h - 3f * dp, borderPaint)
            } else if (s.dropOffHitMeters != null) {
                borderPaint.color = Color.argb(240, 255, 128, 0)
                canvas.drawRect(3f * dp, 3f * dp, w - 3f * dp, h - 3f * dp, borderPaint)
            } else if (s.alertActive) {
                borderPaint.color = Color.argb(225, 255, 59, 48)
                canvas.drawRect(3f * dp, 3f * dp, w - 3f * dp, h - 3f * dp, borderPaint)
            } else if (s.dynClosingHitMeters != null) {
                borderPaint.color = Color.argb(200, 50, 215, 255)
                canvas.drawRect(3f * dp, 3f * dp, w - 3f * dp, h - 3f * dp, borderPaint)
            } else if (s.pHorizon >= s.alertThreshold * 0.70f) {
                borderPaint.color = Color.argb(140, 255, 179, 0)
                canvas.drawRect(3f * dp, 3f * dp, w - 3f * dp, h - 3f * dp, borderPaint)
            }

            // Subtle corner brackets showing the exact 16:9 region fed into EgoClear NPU
            val boxW: Float
            val boxH: Float
            if (!s.isLandscapeGravity) {
                boxW = w * 0.92f
                boxH = boxW * (294f / 518f)
            } else {
                boxH = h * 0.62f
                boxW = min(w * 0.92f, boxH * (294f / 518f))
            }
            val bx0 = (w - boxW) * 0.5f
            val by0 = (h - boxH) * 0.5f
            val bx1 = bx0 + boxW
            val by1 = by0 + boxH
            val arm = 18f * dp
            panelStrokePaint.color = Color.argb(130, 255, 255, 255)
            // Top-left, top-right, bottom-left, bottom-right brackets
            canvas.drawLine(bx0, by0, bx0 + arm, by0, panelStrokePaint)
            canvas.drawLine(bx0, by0, bx0, by0 + arm, panelStrokePaint)
            canvas.drawLine(bx1, by0, bx1 - arm, by0, panelStrokePaint)
            canvas.drawLine(bx1, by0, bx1, by0 + arm, panelStrokePaint)
            canvas.drawLine(bx0, by1, bx0 + arm, by1, panelStrokePaint)
            canvas.drawLine(bx0, by1, bx0, by1 - arm, panelStrokePaint)
            canvas.drawLine(bx1, by1, bx1 - arm, by1, panelStrokePaint)
            canvas.drawLine(bx1, by1, bx1, by1 - arm, panelStrokePaint)
            textPaint.color = Color.argb(150, 255, 255, 255)
            textPaint.textSize = 10f * dp
            canvas.drawText(
                if (s.isLandscapeGravity) "模型视野 16:9 (重力横屏已自动校正)" else "模型视野 16:9",
                bx0 + 6f * dp,
                by0 - 5f * dp,
                textPaint,
            )
        }

        // 2. Top-right Egocentric BEV Static/Dynamic Mini-Radar
        val radarW = min(146f * dp, w * 0.37f)
        val radarH = radarW * 1.38f
        val radarRight = w - 12f * dp
        val radarTop = 108f * dp
        val radarLeft = radarRight - radarW
        val radarBottom = radarTop + radarH

        rect.set(radarLeft, radarTop, radarRight, radarBottom)
        canvas.drawRoundRect(rect, 12f * dp, 12f * dp, panelBgPaint)
        canvas.drawRoundRect(rect, 12f * dp, 12f * dp, panelStrokePaint)

        val pad = 8f * dp
        val headerH = 18f * dp
        val footerH = 16f * dp
        val gridL = radarLeft + pad
        val gridT = radarTop + headerH + 4f * dp
        val gridR = radarRight - pad
        val gridB = radarBottom - footerH
        val gridW = gridR - gridL
        val gridH = gridB - gridT
        val cellW = gridW / EgoClearStreamingEngine.BEV_NX
        val cellH = gridH / EgoClearStreamingEngine.BEV_NZ

        titlePaint.color = Color.WHITE
        canvas.drawText("BEV 雷达 (5m)", gridL, radarTop + 15f * dp, titlePaint)
        if (s.headHazardProb >= 0.45f) {
            titlePaint.color = Color.rgb(255, 214, 10)
            canvas.drawText("头顶!", radarRight - 42f * dp, radarTop + 15f * dp, titlePaint)
        }

        // Draw BEV cells: z=0 (0.3m) at bottom, z=23 (5.1m) at top; x=0 (-1.5m left) to x=14 (+1.5m right)
        val nx = EgoClearStreamingEngine.BEV_NX
        val nz = EgoClearStreamingEngine.BEV_NZ
        for (iz in 0 until nz) {
            val rowTop = gridB - (iz + 1) * cellH
            for (ix in 0 until nx) {
                val idx = iz * nx + ix
                val ps = s.staticProbGrid[idx]
                val pd = s.dynamicProbGrid[idx]
                if (ps < 0.14f && pd < 0.18f) continue
                val cLeft = gridL + ix * cellW
                if (ps >= pd) {
                    val alpha = ((ps * 245f).toInt()).coerceIn(35, 245)
                    cellPaint.color = if (ps >= 0.55f) {
                        Color.argb(alpha, 255, 69, 58)
                    } else {
                        Color.argb(alpha, 255, 179, 0)
                    }
                } else {
                    val alpha = ((pd * 230f).toInt()).coerceIn(35, 230)
                    cellPaint.color = Color.argb(alpha, 50, 215, 255)
                }
                canvas.drawRect(cLeft + 0.5f, rowTop + 0.5f, cLeft + cellW - 0.5f, rowTop + cellH - 0.5f, cellPaint)
            }
        }

        // Range reference lines at 1m, 2m, 3m, 4m
        for (m in 1..4) {
            val frac = ((m - 0.3f) / 4.8f).coerceIn(0f, 1f)
            val y = gridB - frac * gridH
            canvas.drawLine(gridL, y, gridR, y, gridLinePaint)
        }
        // Central walking corridor (±0.30 m = columns 6..8)
        val corrLeft = gridL + 6f * cellW
        val corrRight = gridL + 9f * cellW
        canvas.drawLine(corrLeft, gridT, corrLeft, gridB, corridorPaint)
        canvas.drawLine(corrRight, gridT, corrRight, gridB, corridorPaint)

        textPaint.color = Color.argb(215, 220, 230, 242)
        textPaint.textSize = 9.5f * dp
        canvas.drawText("红:静障  青:行人", gridL, radarBottom - 4f * dp, textPaint)

        // 3. Bottom 24-bin Monotonic Distance-Risk Curve (above button bar)
        val curveH = 112f * dp
        val curveBottom = h - 84f * dp
        val curveTop = curveBottom - curveH
        val curveLeft = 12f * dp
        val curveRight = w - 12f * dp

        rect.set(curveLeft, curveTop, curveRight, curveBottom)
        canvas.drawRoundRect(rect, 14f * dp, 14f * dp, panelBgPaint)
        canvas.drawRoundRect(rect, 14f * dp, 14f * dp, panelStrokePaint)

        val statusColor = when {
            s.pitchGated || s.isUnknownUnobservable -> Color.rgb(160, 175, 195)
            s.dropOffHitMeters != null -> Color.rgb(255, 149, 0)
            s.alertActive -> Color.rgb(255, 69, 58)
            s.dynClosingHitMeters != null -> Color.rgb(50, 215, 255)
            s.pHorizon >= s.alertThreshold * 0.70f -> Color.rgb(255, 196, 0)
            else -> Color.rgb(48, 209, 88)
        }
        heroPaint.color = statusColor
        val bypassSuffix = if (s.bypassAdvice.shortZh.isNotEmpty()) " · ${s.bypassAdvice.shortZh}" else ""
        val headline = when {
            s.pitchGated ->
                "镜头俯仰过大 · 暂停预警"
            s.isUnknownUnobservable ->
                "画面受阻/看不清 · 暂停预警"
            s.dropOffHitMeters != null ->
                String.format("⚠ 下台阶/落差 · 约 %.1f m", s.dropOffHitMeters)
            s.headHitMeters != null ->
                String.format("⚠ 上方低障/碰头 · %.1f m%s", s.headHitMeters, bypassSuffix)
            s.estimatedHitMeters != null && s.headHazardProb >= 0.50f ->
                String.format("头顶/前方障碍 · %.1f m%s", s.estimatedHitMeters, bypassSuffix)
            s.estimatedHitMeters != null ->
                String.format("前方障碍 · %.1f m%s", s.estimatedHitMeters, bypassSuffix)
            s.alertActive ->
                String.format("前方障碍 (≤%.1fm)%s", s.horizonMeters, bypassSuffix)
            s.dynClosingHitMeters != null ->
                String.format("前方动态靠近 · 约 %.1f m", s.dynClosingHitMeters)
            else ->
                String.format("前方通道畅通 (%.1fm)", s.horizonMeters)
        }
        canvas.drawText(headline, curveLeft + 12f * dp, curveTop + 26f * dp, heroPaint)

        textPaint.color = Color.argb(220, 225, 232, 240)
        textPaint.textSize = 11f * dp
        val pText = String.format("P(≤%.1fm)=%.0f%%", s.horizonMeters, s.pHorizon * 100f)
        canvas.drawText(pText, curveRight - 104f * dp, curveTop + 24f * dp, textPaint)

        // Plot 24-bin monotonic curve
        val plotL = curveLeft + 14f * dp
        val plotR = curveRight - 14f * dp
        val plotT = curveTop + 36f * dp
        val plotB = curveBottom - 20f * dp
        val plotW = plotR - plotL
        val plotH = plotB - plotT

        // Threshold dashed line + active horizon vertical marker
        val thrY = plotB - s.alertThreshold.coerceIn(0f, 1f) * plotH
        canvas.drawLine(plotL, thrY, plotR, thrY, thrLinePaint)
        val nBins = s.corrProbs.size
        val horizonX = plotL + (s.horizonIdx.coerceIn(0, max(1, nBins - 1)).toFloat() / max(1, nBins - 1)) * plotW
        canvas.drawLine(horizonX, plotT, horizonX, plotB, corridorPaint)

        // Curve path
        curvePath.reset()
        fillPath.reset()
        for (i in 0 until nBins) {
            val x = plotL + (i.toFloat() / max(1, nBins - 1)) * plotW
            val p = s.corrProbs[i].coerceIn(0f, 1f)
            val y = plotB - p * plotH
            if (i == 0) {
                curvePath.moveTo(x, y)
                fillPath.moveTo(x, plotB)
                fillPath.lineTo(x, y)
            } else {
                curvePath.lineTo(x, y)
                fillPath.lineTo(x, y)
            }
        }
        fillPath.lineTo(plotR, plotB)
        fillPath.close()

        curveFillPaint.color = Color.argb(72, Color.red(statusColor), Color.green(statusColor), Color.blue(statusColor))
        curveStrokePaint.color = statusColor
        canvas.drawPath(fillPath, curveFillPaint)
        canvas.drawPath(curvePath, curveStrokePaint)

        // Distance axis ticks (0.5m, 1.5m, 2.5m, 3.5m, 4.5m, 5.1m)
        textPaint.color = Color.argb(190, 200, 212, 226)
        textPaint.textSize = 10f * dp
        val ticks = floatArrayOf(0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.1f)
        for (dist in ticks) {
            val frac = ((dist - 0.5f) / 4.6f).coerceIn(0f, 1f)
            val tx = plotL + frac * plotW - 10f * dp
            canvas.drawText(String.format("%.1fm", dist), tx.coerceIn(plotL, plotR - 26f * dp), curveBottom - 6f * dp, textPaint)
        }
    }

    companion object {
        private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

        fun buildState(
            out: EgoClearStreamingEngine.StepOutput,
            alertThreshold: Float,
            horizonIdx: Int,
            pitchGated: Boolean = false,
            isLandscapeGravity: Boolean = false,
        ): OverlayState {
            val nx = EgoClearStreamingEngine.BEV_NX
            val nz = EgoClearStreamingEngine.BEV_NZ
            val gridSize = nz * nx
            val stGrid = FloatArray(gridSize)
            val dyGrid = FloatArray(gridSize)
            val hIdx = horizonIdx.coerceIn(0, out.corrProbsMonotonic.lastIndex)
            val horizonMeters = 0.30f + (hIdx + 1) * 0.20f

            // staticLogits / dynamicLogits shape: [1, 3, 24, 15] -> 3 height bands (0=LOW, 1=BODY, 2=HEAD)
            var maxHeadInCorr = 0f
            val hasFullStatic = out.staticLogits.size >= 3 * gridSize
            val hasFullDyn = out.dynamicLogits.size >= 3 * gridSize
            if (hasFullStatic) {
                for (i in 0 until gridSize) {
                    val pLow = sigmoid(out.staticLogits[i])
                    val pBody = sigmoid(out.staticLogits[gridSize + i])
                    val pHead = sigmoid(out.staticLogits[2 * gridSize + i])
                    stGrid[i] = max(pLow, max(pBody, pHead))

                    val iz = i / nx
                    val ix = i % nx
                    if (ix in 6..8 && iz <= hIdx && pHead > maxHeadInCorr) {
                        maxHeadInCorr = pHead
                    }
                }
            }
            if (hasFullDyn) {
                for (i in 0 until gridSize) {
                    val p0 = sigmoid(out.dynamicLogits[i])
                    val p1 = sigmoid(out.dynamicLogits[gridSize + i])
                    val p2 = sigmoid(out.dynamicLogits[2 * gridSize + i])
                    dyGrid[i] = max(p0, max(p1, p2))
                }
            }

            // Estimate first-hit distance within active horizon (0..hIdx)
            var estMeters: Float? = null
            for (k in 0..hIdx) {
                if (out.corrProbsMonotonic[k] >= alertThreshold) {
                    estMeters = 0.30f + (k + 1) * 0.20f
                    break
                }
            }

            return OverlayState(
                corrProbs = out.corrProbsMonotonic,
                staticProbGrid = stGrid,
                dynamicProbGrid = dyGrid,
                headHazardProb = max(maxHeadInCorr, out.headHazardProbs?.getOrNull(hIdx) ?: 0f),
                pHorizon = out.corrProbsMonotonic[hIdx],
                horizonMeters = horizonMeters,
                horizonIdx = hIdx,
                alertActive = !pitchGated && !out.isUnknownUnobservable && out.alertActive,
                alertThreshold = alertThreshold,
                estimatedHitMeters = estMeters,
                pitchGated = pitchGated,
                isLandscapeGravity = isLandscapeGravity,
                bypassAdvice = if (pitchGated) EgoClearStreamingEngine.BypassAdvice.CLEAR else out.bypassAdvice,
                dropOffHitMeters = if (pitchGated) null else out.dropOffHitMeters,
                headHitMeters = if (pitchGated) null else out.headHitMeters,
                dynClosingHitMeters = if (pitchGated) null else out.dynClosingHitMeters,
                isUnknownUnobservable = !pitchGated && out.isUnknownUnobservable,
            )
        }
    }
}
