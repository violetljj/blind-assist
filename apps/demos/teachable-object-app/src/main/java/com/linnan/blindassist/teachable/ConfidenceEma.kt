package com.linnan.blindassist.teachable

import kotlin.math.exp
import kotlin.math.pow

/**
 * Causal smoothing of the per-object scores: each frame enters the EMA weighted by its confidence, p_max^[power], where p is the
 * softmax of score / [temperature]. power = 0 is the plain EMA. Only the argmax is used, so the weights need no normalisation.
 * Graduation orbit_wcl_clip_20261106: t = 16 was chosen on development users, then registered on stored ORBIT test scores.
 */
class ConfidenceEma(private val alpha: Double, private val power: Int = DEFAULT_POWER, private val temperature: Double = 0.05) {
    private var state: DoubleArray? = null

    /** Adds one frame and returns the smoothed top-1. */
    fun update(score: FloatArray): Int {
        val s = state ?: DoubleArray(score.size).also { state = it }
        val w = (1 - alpha) * weight(score, power, temperature)
        for (i in s.indices) s[i] = alpha * s[i] + w * score[i]
        return s.indices.maxBy { s[it] }
    }

    companion object {
        const val DEFAULT_POWER = 16

        fun weight(score: FloatArray, power: Int, temperature: Double = 0.05): Double {
            if (power == 0) return 1.0
            val top = score.max()
            var sum = 0.0
            for (v in score) sum += exp((v - top) / temperature)
            return (1.0 / sum).pow(power)
        }
    }
}
