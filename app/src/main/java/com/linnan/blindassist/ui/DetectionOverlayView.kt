package com.linnan.blindassist.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import kotlin.math.min

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    private var detections: List<Detection> = emptyList()
    private var frameSize: FrameSize? = null
    private var risk: RiskResult? = null

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 32f
        style = Paint.Style.FILL
    }
    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(190, 16, 20, 24)
        style = Paint.Style.FILL
    }
    private val zonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(90, 255, 214, 10)
        style = Paint.Style.STROKE
        strokeWidth = 3f
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
        val transform = fitCenterTransform(sourceSize)
        val riskDetection = risk?.sourceDetection

        detections.forEach { detection ->
            val rect = transform.map(detection.boundingBox.left, detection.boundingBox.top)
            rect.right = transform.mapX(detection.boundingBox.right)
            rect.bottom = transform.mapY(detection.boundingBox.bottom)

            boxPaint.color = colorFor(detection == riskDetection)
            canvas.drawRect(rect, boxPaint)

            val label = "${detection.label} ${(detection.confidence * 100).toInt()}%"
            val labelWidth = textPaint.measureText(label) + 20f
            val labelTop = (rect.top - 38f).coerceAtLeast(0f)
            canvas.drawRoundRect(
                RectF(rect.left, labelTop, rect.left + labelWidth, labelTop + 38f),
                8f,
                8f,
                labelBgPaint
            )
            canvas.drawText(label, rect.left + 10f, labelTop + 28f, textPaint)
        }
    }

    private fun drawDangerZone(canvas: Canvas) {
        val left = width * 0.35f
        val right = width * 0.65f
        val top = height * 0.35f
        val bottom = height * 0.98f
        canvas.drawRect(left, top, right, bottom, zonePaint)
    }

    private fun colorFor(isRiskSource: Boolean): Int {
        return when {
            risk?.level == RiskLevel.HIGH && isRiskSource -> Color.rgb(255, 59, 48)
            risk?.level == RiskLevel.MEDIUM && isRiskSource -> Color.rgb(255, 149, 0)
            else -> Color.rgb(52, 199, 89)
        }
    }

    private fun fitCenterTransform(sourceSize: FrameSize): ViewTransform {
        val scale = min(
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
