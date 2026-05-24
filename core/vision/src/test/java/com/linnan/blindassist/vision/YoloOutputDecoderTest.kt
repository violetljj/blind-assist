package com.linnan.blindassist.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.tensorflow.lite.DataType

class YoloOutputDecoderTest {
    @Test
    fun parseAppliesThresholdLabelMappingLetterboxAndSameClassNms() {
        val raw = FloatArray(CHANNELS * PREDICTIONS)
        putPrediction(raw, prediction = 0, cx = 160f, cy = 160f, w = 100f, h = 100f, person = 0.92f, chair = 0.10f)
        putPrediction(raw, prediction = 1, cx = 164f, cy = 164f, w = 100f, h = 100f, person = 0.80f, chair = 0.05f)
        putPrediction(raw, prediction = 2, cx = 250f, cy = 120f, w = 40f, h = 40f, person = 0.10f, chair = 0.70f)
        putPrediction(raw, prediction = 3, cx = 40f, cy = 40f, w = 20f, h = 20f, person = 0.20f, chair = 0.10f)

        val result = YoloOutputDecoder.parse(
            raw = raw,
            shape = intArrayOf(1, CHANNELS, PREDICTIONS),
            dataType = DataType.FLOAT32,
            letterbox = LetterboxInfo(
                scale = 0.5f,
                dx = 0f,
                dy = 40f,
                sourceWidth = 640,
                sourceHeight = 480,
                inputSize = 320
            ),
            labels = listOf("person", "chair"),
            confidenceThreshold = 0.35f,
            iouThreshold = 0.45f
        )

        assertNull(result.warning)
        assertEquals(2, result.detections.size)
        assertEquals("person", result.detections[0].label)
        assertEquals(0.92f, result.detections[0].confidence, 0.0001f)
        assertEquals(220f, result.detections[0].boundingBox.left, 0.0001f)
        assertEquals(140f, result.detections[0].boundingBox.top, 0.0001f)
        assertEquals(420f, result.detections[0].boundingBox.right, 0.0001f)
        assertEquals(340f, result.detections[0].boundingBox.bottom, 0.0001f)
        assertEquals("chair", result.detections[1].label)
    }

    @Test
    fun parseReturnsWarningForUnsupportedShape() {
        val result = YoloOutputDecoder.parse(
            raw = FloatArray(0),
            shape = intArrayOf(1, 2),
            dataType = DataType.FLOAT32,
            letterbox = LetterboxInfo(1f, 0f, 0f, 320, 320, 320),
            labels = listOf("person"),
            confidenceThreshold = 0.35f,
            iouThreshold = 0.45f
        )

        assertTrue(result.detections.isEmpty())
        assertEquals("模型输出形状不支持：[1, 2]", result.warning)
    }

    private fun putPrediction(
        raw: FloatArray,
        prediction: Int,
        cx: Float,
        cy: Float,
        w: Float,
        h: Float,
        person: Float,
        chair: Float
    ) {
        raw[0 * PREDICTIONS + prediction] = cx
        raw[1 * PREDICTIONS + prediction] = cy
        raw[2 * PREDICTIONS + prediction] = w
        raw[3 * PREDICTIONS + prediction] = h
        raw[4 * PREDICTIONS + prediction] = person
        raw[5 * PREDICTIONS + prediction] = chair
    }

    private companion object {
        const val CHANNELS = 6
        const val PREDICTIONS = 8
    }
}
