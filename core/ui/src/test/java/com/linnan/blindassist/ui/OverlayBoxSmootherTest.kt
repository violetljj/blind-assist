package com.linnan.blindassist.ui

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Test

class OverlayBoxSmootherTest {
    private val frame = FrameSize(1000, 1000)

    @Test
    fun smoothsSmallBoxMovementForDisplayOnly() {
        val smoother = OverlayBoxSmoother(alpha = 0.5f)
        smoother.smooth(listOf(detection(BoundingBox(100f, 100f, 300f, 300f))))

        val result = smoother.smooth(listOf(detection(BoundingBox(120f, 120f, 320f, 320f)))).single()

        assertEquals(BoundingBox(120f, 120f, 320f, 320f), result.raw.boundingBox)
        assertEquals(110f, result.display.boundingBox.left, 0.01f)
        assertEquals(110f, result.display.boundingBox.top, 0.01f)
        assertEquals(310f, result.display.boundingBox.right, 0.01f)
        assertEquals(310f, result.display.boundingBox.bottom, 0.01f)
    }

    @Test
    fun largeJumpUsesCurrentBoxImmediately() {
        val smoother = OverlayBoxSmoother(alpha = 0.5f)
        smoother.smooth(listOf(detection(BoundingBox(100f, 100f, 300f, 300f))))

        val result = smoother.smooth(listOf(detection(BoundingBox(500f, 500f, 700f, 700f)))).single()

        assertEquals(BoundingBox(500f, 500f, 700f, 700f), result.display.boundingBox)
    }

    private fun detection(box: BoundingBox): Detection {
        return Detection(
            classId = 0,
            label = "person",
            confidence = 0.9f,
            boundingBox = box,
            frameSize = frame
        )
    }
}
