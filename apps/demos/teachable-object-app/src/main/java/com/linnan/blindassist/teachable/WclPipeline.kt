package com.linnan.blindassist.teachable

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.Closeable
import java.io.File
import kotlin.math.sqrt

/**
 * Whole-frame prototype recognition on SigLIP2-So400m adapted on ORBIT (WCL, Graduation orbit_vitl_wc_20261028), one
 * offline-prepared DLC on the HTP (Graduation orbit_wcl_phone_20261102). Input CHW float RGB 0..255, normalisation in the
 * graph; output the L2-normalised 1152-d embedding. Separate from [TeachablePipeline]: no local memory, heads or relook.
 */
class WclPipeline(model: File, dspPath: String) : Closeable {
    private val engine = SnpeEngine(model.path, htp = true, dspPath = dspPath)

    fun image(file: File): FloatArray = imageOf(file)

    fun image(bitmap: Bitmap): FloatArray = imageOf(bitmap)

    fun embed(image: FloatArray): FloatArray = engine.run(mapOf(engine.inputs.single() to image)).values.single()

    fun prototypes(sums: List<FloatArray>): FloatArray = prototypesOf(sums)

    fun scores(embedding: FloatArray, protos: FloatArray): FloatArray = cosine(embedding, protos)

    override fun close() = engine.close()

    companion object {
        const val SIZE = 224
        const val D = 1152

        fun imageOf(file: File): FloatArray {
            val bitmap = requireNotNull(BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888; inScaled = false
            })) { "decode $file" }
            require(bitmap.width == SIZE && bitmap.height == SIZE) { "$file is ${bitmap.width}x${bitmap.height}" }
            return imageOf(bitmap).also { bitmap.recycle() }
        }

        /** SIZE x SIZE ARGB bitmap -> CHW float RGB 0..255. */
        fun imageOf(bitmap: Bitmap): FloatArray {
            val px = IntArray(SIZE * SIZE).also { bitmap.getPixels(it, 0, SIZE, 0, 0, SIZE, SIZE) }
            val plane = SIZE * SIZE
            return FloatArray(3 * plane).also { x ->
                for (i in 0 until plane) {
                    val p = px[i]
                    x[i] = ((p shr 16) and 255).toFloat(); x[plane + i] = ((p shr 8) and 255).toFloat(); x[2 * plane + i] = (p and 255).toFloat()
                }
            }
        }

        /** Normalised mean of the teaching embeddings of each object: [objects * D]. */
        fun prototypesOf(sums: List<FloatArray>): FloatArray {
            val out = FloatArray(sums.size * D)
            for ((o, c) in sums.withIndex()) {
                var n = 0f
                for (d in 0 until D) n += c[d] * c[d]
                n = sqrt(n).coerceAtLeast(1e-12f)
                for (d in 0 until D) out[o * D + d] = c[d] / n
            }
            return out
        }

        /** Cosine to each prototype. */
        fun cosine(embedding: FloatArray, protos: FloatArray): FloatArray = FloatArray(protos.size / D) { o ->
            var s = 0f
            for (d in 0 until D) s += embedding[d] * protos[o * D + d]
            s
        }
    }
}
