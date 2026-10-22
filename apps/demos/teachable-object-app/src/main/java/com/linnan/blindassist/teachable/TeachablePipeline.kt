package com.linnan.blindassist.teachable

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.Closeable
import java.io.File
import kotlin.math.sqrt

/**
 * Teach -> build memory -> identify, as the Graduation thesis main method (S config, local <= 1.5T budget variant):
 * DINOv2-S/14 backbone on the HTP, matching + pooling head + localiser on the CPU in FP32, one zoomed second look when
 * the top-two z-score gap is below TAU (loc0.5 box, roi_align crop).
 */
class TeachablePipeline(backbone: File, head: File, dspPath: String) : Closeable {
    private val backbone = SnpeEngine(backbone.path, htp = true, dspPath = dspPath)
    private val head = SnpeEngine(head.path, htp = false, dspPath = "", outputTensors = listOf("logits", "loc"))
    /** Output names as SNPE reports them (they need not match the ONNX names); outputs are picked by size in [look]. */
    val headOutputs: List<String> get() = head.outputs

    /** Memory padded to OMAX objects; [valid] marks the user's real objects. */
    class Memory(val objects: Int, val parts: FloatArray, val protos: FloatArray, val valid: FloatArray)

    /** [scores]: the final per-object z-scores (main readout), the input to temporal smoothing. */
    class Result(val pred: Int, val looked: Boolean, val firstLogits: FloatArray, val ms: Map<String, Double>, val scores: FloatArray)

    fun image(file: File): FloatArray {
        val bitmap = requireNotNull(BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888; inScaled = false
        })) { "decode $file" }
        require(bitmap.width == SIZE && bitmap.height == SIZE) { "$file is ${bitmap.width}x${bitmap.height}" }
        return image(bitmap).also { bitmap.recycle() }
    }

    /** SIZE x SIZE ARGB bitmap -> CHW float RGB 0..255 (the backbone folds the normalisation in). */
    fun image(bitmap: Bitmap): FloatArray {
        val px = IntArray(SIZE * SIZE).also { bitmap.getPixels(it, 0, SIZE, 0, 0, SIZE, SIZE) }
        val plane = SIZE * SIZE
        return FloatArray(3 * plane).also { x ->
            for (i in 0 until plane) {
                val p = px[i]
                x[i] = ((p shr 16) and 255).toFloat(); x[plane + i] = ((p shr 8) and 255).toFloat(); x[2 * plane + i] = (p and 255).toFloat()
            }
        }
    }

    /** L2-normalised tokens [257, 384]: CLS then 256 patches. */
    fun tokens(image: FloatArray): FloatArray = backbone.run(mapOf(backbone.inputs.single() to image)).values.single()

    fun memory(objects: Int, parts: FloatArray, cls: List<FloatArray>): Memory {
        require(objects in 2..OMAX && parts.size == objects * PARTS * D)
        val p = FloatArray(OMAX * PARTS * D).also { parts.copyInto(it) }
        val protos = FloatArray(OMAX * D)
        for (o in 0 until objects) {
            val c = cls[o]; var n = 0f
            for (d in 0 until D) n += c[d] * c[d]
            n = sqrt(n).coerceAtLeast(1e-12f)
            for (d in 0 until D) protos[o * D + d] = c[d] / n
        }
        return Memory(objects, p, protos, FloatArray(OMAX) { if (it < objects) 1f else 0f })
    }

    private fun look(tokens: FloatArray, m: Memory): Pair<FloatArray, FloatArray> {
        val out = head.run(mapOf("tokens" to tokens, "parts" to m.parts, "protos" to m.protos, "valid" to m.valid))
        val logits = out.values.single { it.size == OMAX }
        val loc = out.values.single { it.size == GRID * GRID }
        return logits.copyOf(m.objects) to loc
    }

    fun identify(image: FloatArray, tok: FloatArray, m: Memory): Result {
        val ms = LinkedHashMap<String, Double>()
        var t = System.nanoTime()
        val (logits, loc) = look(tok, m)
        ms["head"] = (System.nanoTime() - t) / 1e6
        var s = z(logits)
        val sorted = s.sortedDescending()
        val looked = sorted[0] - sorted[1] < TAU
        if (looked) {
            t = System.nanoTime()
            val b = zoomBox(loc)
            val crop = Native.roiAlign(image, SIZE, b[0], b[1], b[2], b[3], SIZE)
            ms["crop"] = (System.nanoTime() - t) / 1e6; t = System.nanoTime()
            val tok2 = tokens(crop)
            ms["backbone2"] = (System.nanoTime() - t) / 1e6; t = System.nanoTime()
            val z2 = z(look(tok2, m).first)
            ms["head2"] = (System.nanoTime() - t) / 1e6
            s = z(FloatArray(s.size) { s[it] + z2[it] })
        }
        return Result(s.indices.maxBy { s[it] }, looked, logits, ms, s)
    }

    override fun close() {
        backbone.close(); head.close()
    }

    companion object {
        const val OMAX = 12
        const val PARTS = 32
        const val D = 384
        const val SIZE = 224
        const val GRID = 16
        const val TAU = 0.75f
        const val ZOOM = 0.5f
        const val KEEP = 0.15f

        /** z-score over the user's objects (population standard deviation, as numpy). */
        fun z(s: FloatArray): FloatArray {
            val mu = s.average()
            val sd = sqrt(s.sumOf { (it - mu) * (it - mu) } / s.size)
            return FloatArray(s.size) { ((s[it] - mu) / (sd + 1e-6)).toFloat() }
        }

        /** orbit_dump.zoom_boxes on the localiser map: 3x3 mean (no padding counted), argmax, square of ZOOM*SIZE. */
        fun zoomBox(loc: FloatArray): FloatArray {
            var best = -Float.MAX_VALUE; var k = 0
            for (y in 0 until GRID) for (x in 0 until GRID) {
                var sum = 0f; var n = 0
                for (dy in -1..1) for (dx in -1..1) {
                    val yy = y + dy; val xx = x + dx
                    if (yy in 0 until GRID && xx in 0 until GRID) { sum += loc[yy * GRID + xx]; n++ }
                }
                val e = sum / n
                if (e > best) { best = e; k = y * GRID + x }
            }
            val half = ZOOM * SIZE / 2
            val cy = ((k / GRID) + .5f) * SIZE / GRID; val cx = ((k % GRID) + .5f) * SIZE / GRID
            val x = cx.coerceIn(half, SIZE - half); val y = cy.coerceIn(half, SIZE - half)
            return floatArrayOf(x - half, y - half, x + half, y + half)
        }
    }
}
