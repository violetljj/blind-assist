package com.linnan.blindassist.risk

import java.io.DataInputStream
import java.io.InputStream
import kotlin.math.exp

/** Frozen binary HistGradientBoosting model, exported without refitting or cutoff selection. */
class HardwareLocalModel private constructor(
    val threshold: Double,
    private val baseline: Double,
    private val trees: List<List<Node>>,
) {
    private data class Node(val value: Double, val feature: Int, val threshold: Double,
        val left: Int, val right: Int, val missingLeft: Boolean, val leaf: Boolean)

    fun predict(features: FloatArray): Double {
        require(features.size == HardwareLocalFeatures.FEATURE_SIZE && features.all { it.isFinite() })
        var score = baseline
        for (tree in trees) {
            var index = 0
            while (!tree[index].leaf) {
                val node = tree[index]; val value = features[node.feature].toDouble()
                index = if (if (value.isNaN()) node.missingLeft else value <= node.threshold) node.left else node.right
            }
            score += tree[index].value
        }
        return if (score >= 0) 1.0 / (1.0 + exp(-score)) else exp(score) / (1.0 + exp(score))
    }

    companion object {
        fun read(input: InputStream): HardwareLocalModel {
            val stream = DataInputStream(input)
            require(stream.readInt() == 0x42414c31) { "Unsupported LOCAL model" }
            require(stream.readInt() == HardwareLocalFeatures.FEATURE_SIZE)
            val threshold = stream.readDouble(); val baseline = stream.readDouble()
            require(threshold in 0.0..1.0 && baseline.isFinite())
            val count = stream.readInt(); require(count in 1..1000)
            val trees = List(count) {
                val size = stream.readInt(); require(size in 1..1024)
                List(size) {
                    Node(stream.readDouble(), stream.readInt(), stream.readDouble(), stream.readInt(), stream.readInt(), stream.readBoolean(), stream.readBoolean())
                }.also { nodes -> nodes.forEach { node ->
                    require(node.value.isFinite() && node.threshold.isFinite())
                    require(node.leaf || (node.feature in 0 until HardwareLocalFeatures.FEATURE_SIZE && node.left in nodes.indices && node.right in nodes.indices))
                } }
            }
            require(stream.read() == -1) { "Trailing LOCAL model data" }
            return HardwareLocalModel(threshold, baseline, trees)
        }
    }
}
