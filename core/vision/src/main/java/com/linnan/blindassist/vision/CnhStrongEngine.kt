package com.linnan.blindassist.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer

data class CnhStrongResult(
    val headStrong: Boolean,
    val bodyStrong: Boolean,
    val ready: Boolean,
    val historyLength: Int,
    val evidencePresent: Boolean,
    val headScore: Double,
    val bodyScore: Double,
    val historyReset: Boolean,
)

internal interface CnhStrongModel : Closeable {
    fun score(input: FloatArray): FloatArray
}

private class CnhStrongOnnx(modelFile: File) : CnhStrongModel {
    private val environment = OrtEnvironment.getEnvironment()
    private val session = OrtSession.SessionOptions().use { options ->
        options.setIntraOpNumThreads(2)
        options.setInterOpNumThreads(1)
        // Default CPU provider, no accelerator precision substitutions.
        environment.createSession(modelFile.absolutePath, options)
    }

    override fun score(input: FloatArray): FloatArray =
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), longArrayOf(1, 5, 24, 17, 33)).use { tensor ->
            session.run(mapOf("voxels" to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                (result[0].value as Array<FloatArray>)[0].copyOf()
            }
        }

    override fun close() = session.close()
}

/** Research strong-only readout. Call from a worker, never the Android UI thread.
 * Includes native absence gating, four-frame inference, eight-frame readiness and
 * frozen last-five 1:2:4:8:16 smoothing. Pose-envelope/output policy is caller-owned.
 */
class CnhStrongEngine internal constructor(
    private val calibration: CnhStrongCalibration,
    private val model: CnhStrongModel,
) : Closeable {
    constructor(modelFile: File, calibrationFile: File) : this(
        CnhStrongCalibration.read(calibrationFile), CnhStrongOnnx(modelFile),
    )

    private val preprocessor = CnhStrongPreprocessor(calibration)
    private val frames = ArrayDeque<FloatArray>()
    private val logits = ArrayDeque<FloatArray>()
    private var previousSequence: Long? = null
    private var previousMs: Long? = null
    private var closed = false

    @Synchronized
    fun reset() {
        frames.clear(); logits.clear()
        previousSequence = null; previousMs = null
    }

    @Synchronized
    fun process(
        histogram: FloatArray,
        ambient: FloatArray,
        sequence: Long,
        sensorTimeMs: Long,
        pitchDeg: Double,
        rollDeg: Double,
    ): CnhStrongResult {
        check(!closed) { "CNH engine closed" }
        try {
            require(pitchDeg.isFinite() && rollDeg.isFinite()) { "Finite CNH pose required" }
            val native = preprocessor.normalize(histogram, ambient)
            val dt = previousMs?.let { sensorTimeMs - it }
            val historyReset = previousSequence == null || sequence - previousSequence!! != 1L ||
                dt == null || dt <= 0L || dt > 500L
            if (historyReset) reset()
            previousSequence = sequence; previousMs = sensorTimeMs
            frames.addLast(native)
            if (frames.size > 8) frames.removeFirst()
            val present = frames.any { f -> f.any { it > 0f } }
            if (frames.size >= 4) {
                val score = model.score(preprocessor.prepare(frames.toList(), pitchDeg, rollDeg))
                require(score.size == 2 && score.all { it.isFinite() }) { "Invalid CNH logits" }
                logits.addLast(score)
                if (logits.size > 5) logits.removeFirst()
            }
            val smooth = DoubleArray(2) { Double.NaN }
            if (logits.isNotEmpty()) {
                smooth.fill(0.0)
                var sum = 0.0
                var weight = 1.0
                for (score in logits) {
                    for (q in 0..1) smooth[q] += score[q].toDouble() * weight
                    sum += weight; weight *= 2
                }
                for (q in 0..1) smooth[q] /= sum
            }
            val ready = frames.size == 8
            return CnhStrongResult(
                ready && present && smooth[0] >= calibration.raisedTau,
                ready && present && smooth[1] >= calibration.raisedTau,
                ready, frames.size, present, smooth[0], smooth[1], historyReset,
            )
        } catch (failure: Exception) {
            // A failed observation cannot leave apparently continuous state behind.
            reset()
            throw failure
        }
    }

    @Synchronized
    override fun close() {
        if (!closed) {
            closed = true
            reset()
            model.close()
        }
    }
}
