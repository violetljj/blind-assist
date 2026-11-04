package com.linnan.blindassist.teachable

import java.io.Closeable
import java.io.File

/**
 * WCL with the prototype-guided spatial readout (PROTO_FULL, Graduation orbit_wcl_protofull_phone_20261104), two
 * offline-prepared DLCs on the HTP. Body: CHW float RGB 0..255 -> post-LN tokens [1, 256, 1152]. Read head: tokens, up to
 * [SLOTS] teaching prototypes and their mask -> one score per slot and the original MAP embedding, which serves for teaching
 * and for the paired plain WCL score. At least two objects are needed: the spatial field compares each candidate with the others.
 */
class PfPipeline(body: File, head: File, dspPath: String) : Closeable {
    class Read(val scores: FloatArray, val embedding: FloatArray)

    private val bodyEngine = SnpeEngine(body.path, htp = true, dspPath = dspPath)
    private val headEngine = SnpeEngine(head.path, htp = true, dspPath = dspPath, outputTensors = listOf("scores", "embedding"))
    private val proto = FloatArray(SLOTS * WclPipeline.D)
    private val mask = FloatArray(SLOTS)
    private var objects = 0

    fun tokens(image: FloatArray): FloatArray = bodyEngine.run(mapOf(bodyEngine.inputs.single() to image)).values.single()

    /** [objects * D] from [WclPipeline.prototypesOf]; the remaining slots stay empty. */
    fun setPrototypes(protos: FloatArray) {
        objects = protos.size / WclPipeline.D
        require(objects in 2..SLOTS) { "$objects objects, the read head takes 2..$SLOTS" }
        proto.fill(0f); mask.fill(0f)
        protos.copyInto(proto); mask.fill(1f, 0, objects)
    }

    fun read(tokens: FloatArray): Read {
        val out = headEngine.run(mapOf("tokens" to tokens, "proto" to proto, "mask" to mask))
        return Read(requireNotNull(out["scores"]).copyOf(objects), requireNotNull(out["embedding"]))
    }

    /** Teaching embedding, independent of the prototypes in the slots. */
    fun embed(image: FloatArray): FloatArray = read(tokens(image)).embedding

    override fun close() {
        headEngine.close(); bodyEngine.close()
    }

    companion object {
        const val SLOTS = 20
    }
}
