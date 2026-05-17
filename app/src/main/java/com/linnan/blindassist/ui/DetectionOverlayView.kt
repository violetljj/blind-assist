package com.linnan.blindassist.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import kotlin.math.max
import kotlin.math.min

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    private var detections: List<Detection> = emptyList()
    private var frameSize: FrameSize? = null
    private var risk: RiskResult? = null
    private var careMode = false

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }
    private val riskHaloPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 10f
        alpha = 92
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 30f
        style = Paint.Style.FILL
    }
    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(190, 16, 20, 24)
        style = Paint.Style.FILL
    }
    private val zonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(52, 255, 224, 102)
        style = Paint.Style.STROKE
        strokeWidth = 2f
        pathEffect = DashPathEffect(floatArrayOf(18f, 14f), 0f)
    }
    private val zoneFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(14, 255, 224, 102)
        style = Paint.Style.FILL
    }
    private val guidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(80, 255, 255, 255)
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }
    private val zoneLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(150, 255, 244, 198)
        textSize = 24f
        style = Paint.Style.FILL
    }

    fun setCareMode(enabled: Boolean) {
        careMode = enabled
        textPaint.textSize = if (enabled) 34f else 30f
        zonePaint.strokeWidth = if (enabled) 4f else 2f
        zoneLabelPaint.textSize = if (enabled) 28f else 24f
        invalidate()
    }

    fun update(
        detections: List<Detection>,
        frameSize: FrameSize?,
        risk: RiskResult?
    ) {
        this.detections = detections
        this.frameSize = frameSize
        this.risk = risk
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawDangerZone(canvas)

        val sourceSize = frameSize ?: return
        val transform = fillCenterTransform(sourceSize)
        val riskDetection = risk?.sourceDetection

        detections.forEach { detection ->
            val rect = transform.map(detection.boundingBox.left, detection.boundingBox.top)
            rect.right = transform.mapX(detection.boundingBox.right)
            rect.bottom = transform.mapY(detection.boundingBox.bottom)

            val isRiskSource = detection == riskDetection
            val color = colorFor(isRiskSource)
            if (isRiskSource) {
                riskHaloPaint.color = color
                val haloRect = RectF(rect).apply { inset(-6f, -6f) }
                canvas.drawRoundRect(haloRect, 12f, 12f, riskHaloPaint)
            }

            boxPaint.color = color
            boxPaint.strokeWidth = if (isRiskSource) 6f else 2f
            canvas.drawRect(rect, boxPaint)

            val label = fitLabel("${detection.label} ${(detection.confidence * 100).toInt()}%")
            val labelWidth = min(textPaint.measureText(label) + 20f, width - 16f)
            val labelLeft = rect.left.coerceIn(8f, max(8f, width - labelWidth - 8f))
            val labelTop = (rect.top - 38f).coerceAtLeast(0f)
            labelBgPaint.alpha = if (isRiskSource) 210 else 120
            textPaint.alpha = if (isRiskSource) 255 else 190
            canvas.drawRoundRect(
                RectF(labelLeft, labelTop, labelLeft + labelWidth, labelTop + 38f),
                8f,
                8f,
                labelBgPaint
            )
            canvas.drawText(label, labelLeft + 10f, labelTop + 28f, textPaint)
            labelBgPaint.alpha = 255
            textPaint.alpha = 255
        }
    }

    private fun drawDangerZone(canvas: Canvas) {
        val left = width * 0.35f
        val right = width * 0.65f
        val top = height * 0.35f
        val bottom = height * 0.98f
        val zone = RectF(left, top, right, bottom)
        if (careMode) {
            canvas.drawLine(width * 0.5f, height * 0.28f, width * 0.5f, height * 0.98f, guidePaint)
        }
        canvas.drawRoundRect(zone, 16f, 16f, zoneFillPaint)
        canvas.drawRoundRect(zone, 16f, 16f, zonePaint)
        val label = "观察参考区"
        val labelWidth = zoneLabelPaint.measureText(label)
        canvas.drawText(label, zone.centerX() - labelWidth / 2f, top - 10f, zoneLabelPaint)
    }

    private fun fitLabel(label: String): String {
        val maxTextWidth = width - 36f
        if (maxTextWidth <= 0f || textPaint.measureText(label) <= maxTextWidth) {
            return label
        }
        val suffix = "..."
        val suffixWidth = textPaint.measureText(suffix)
        val count = textPaint.breakText(label, true, maxTextWidth - suffixWidth, null)
        return label.take(count.coerceAtLeast(0)) + suffix
    }

    private fun colorFor(isRiskSource: Boolean): Int {
        return when {
            risk?.proximity == ProximityBand.CRITICAL && isRiskSource -> Color.rgb(255, 45, 85)
            risk?.level == RiskLevel.HIGH && isRiskSource -> Color.rgb(255, 59, 48)
            risk?.level == RiskLevel.MEDIUM && isRiskSource -> Color.rgb(255, 149, 0)
            risk?.proximity == ProximityBand.MID && isRiskSource -> Color.rgb(255, 214, 10)
            else -> Color.argb(160, 52, 199, 89)
        }
    }

    private fun fillCenterTransform(sourceSize: FrameSize): ViewTransform {
        val scale = max(
            width.toFloat() / sourceSize.width.toFloat(),
            height.toFloat() / sourceSize.height.toFloat()
        )
        val dx = (width - sourceSize.width * scale) / 2f
        val dy = (height - sourceSize.height * scale) / 2f
        return ViewTransform(scale, dx, dy)
    }

    private data class ViewTransform(
        val scale: Float,
        val dx: Float,
        val dy: Float
    ) {
        fun mapX(x: Float): Float = x * scale + dx
        fun mapY(y: Float): Float = y * scale + dy
        fun map(x: Float, y: Float): RectF = RectF(mapX(x), mapY(y), mapX(x), mapY(y))
    }
}
