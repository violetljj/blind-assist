package com.linnan.blindassist.hftf.metricdepth

import java.util.Random
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/** Benchmark-only Kotlin port of the frozen known-height ground pipeline. */
object KnownHeightGroundPipeline {
    data class Valid(
        val relativeHeight: Double,
        val normalizedMedianResidual: Double,
        val inlierFraction: Double,
        val normal: DoubleArray,
        val features: DoubleArray,
        val studentScale: Double,
    )

    data class Unknown(val reason: String)

    data class Geometry(
        val relativeHeight: Double,
        val normalizedMedianResidual: Double,
        val inlierFraction: Double,
        val normal: DoubleArray,
        val features: DoubleArray,
    )

    fun evaluate(
        depth: FloatArray,
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        cameraHeightM: Double,
    ): Any {
        val geometry = evaluateGeometry(depth, width, height, fx, fy, cx, cy, cameraHeightM)
        if (geometry !is Geometry) return geometry
        val prediction = KnownHeightScaleStudent.frozen().predict(geometry.features)
        if (prediction !is KnownHeightScaleStudent.Prediction.Valid) return Unknown("STUDENT_REJECTED")
        return Valid(
            geometry.relativeHeight,
            geometry.normalizedMedianResidual,
            geometry.inlierFraction,
            geometry.normal,
            geometry.features,
            prediction.scale,
        )
    }

    fun evaluateGeometry(
        depth: FloatArray,
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        cameraHeightM: Double,
    ): Any = workspace.get().evaluate(depth, width, height, fx, fy, cx, cy, cameraHeightM)

    /** Frozen allocation-heavy arm retained only for equivalence benchmarks. */
    fun evaluateGeometryReference(
        depth: FloatArray,
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        cameraHeightM: Double,
    ): Any {
        if (depth.size != width * height || fx <= 0.0 || fy <= 0.0) return Unknown("INVALID_INPUT")
        val candidates = ArrayList<Point>()
        val startRow = ceil(LOWER_ROI_START_FRACTION * height).toInt()
        for (row in startRow until height step STRIDE) {
            for (column in 0 until width step STRIDE) {
                val z = depth[row * width + column].toDouble()
                if (!z.isFinite() || z <= 0.0) continue
                candidates += Point((column - cx) * z / fx, (row - cy) * z / fy, z)
            }
        }
        if (candidates.size < MINIMUM_CANDIDATES) return Unknown("INSUFFICIENT_GROUND_CANDIDATES")
        val points = if (candidates.size <= MAXIMUM_CANDIDATES) {
            candidates
        } else {
            ArrayList<Point>(MAXIMUM_CANDIDATES).apply {
                repeat(MAXIMUM_CANDIDATES) { index ->
                    add(candidates[(index.toLong() * (candidates.lastIndex)) .div(MAXIMUM_CANDIDATES - 1).toInt()])
                }
            }
        }
        val characteristic = median(points.map(Point::norm).toDoubleArray())
        if (!characteristic.isFinite() || characteristic <= 0.0) return Unknown("DEGENERATE_RELATIVE_DEPTH")
        val minimumHeight = max(Double.MIN_VALUE, characteristic * 1e-6)
        val minimumCrossNorm = max(Double.MIN_VALUE, characteristic * characteristic * 1e-12)
        val random = Random(RANSAC_SEED)
        var bestIndices: IntArray? = null
        var bestCount = -1
        var bestResidual = Double.POSITIVE_INFINITY
        repeat(RANSAC_ITERATIONS) {
            val indices = distinctTriple(random, points.size)
            val first = points[indices[0]]
            val second = points[indices[1]]
            val third = points[indices[2]]
            var normal = (second - first).cross(third - first)
            val norm = normal.norm()
            if (!norm.isFinite() || norm <= minimumCrossNorm) return@repeat
            normal /= norm
            if (abs(normal.y) < MINIMUM_ABS_NORMAL_Y) return@repeat
            var offset = -normal.dot(first)
            if (offset < 0.0) {
                normal *= -1.0
                offset = -offset
            }
            if (!offset.isFinite() || offset <= minimumHeight) return@repeat
            val residuals = DoubleArray(points.size) { abs(normal.dot(points[it]) + offset) / offset }
            val inliers = IntArray(points.size)
            var count = 0
            for (index in residuals.indices) if (residuals[index] <= MAXIMUM_NORMALIZED_PLANE_RESIDUAL) {
                inliers[count++] = index
            }
            val residual = if (count == 0) Double.POSITIVE_INFINITY else {
                median(DoubleArray(count) { residuals[inliers[it]] })
            }
            if (count > bestCount || (count == bestCount && residual < bestResidual)) {
                bestCount = count
                bestResidual = residual
                bestIndices = inliers.copyOf(count)
            }
        }
        val required = max(MINIMUM_INLIERS, ceil(MINIMUM_INLIER_FRACTION * points.size).toInt())
        val selected = bestIndices ?: return Unknown("NO_GROUND_CONSENSUS")
        if (selected.size < required) return Unknown("NO_GROUND_CONSENSUS")

        val center = Point(
            selected.sumOf { points[it].x } / selected.size,
            selected.sumOf { points[it].y } / selected.size,
            selected.sumOf { points[it].z } / selected.size,
        )
        val covariance = Array(3) { DoubleArray(3) }
        for (index in selected) {
            val delta = points[index] - center
            val values = doubleArrayOf(delta.x, delta.y, delta.z)
            for (row in 0..2) for (column in 0..2) covariance[row][column] += values[row] * values[column]
        }
        var normal = smallestEigenvector(covariance)
        var offset = -normal.dot(center)
        if (offset < 0.0) {
            normal *= -1.0
            offset = -offset
        }
        if (!offset.isFinite() || offset <= minimumHeight) return Unknown("DEGENERATE_RELATIVE_HEIGHT")
        if (abs(normal.y) < MINIMUM_ABS_NORMAL_Y) return Unknown("GROUND_ORIENTATION_REJECTED")
        val residuals = DoubleArray(points.size) { abs(normal.dot(points[it]) + offset) / offset }
        val accepted = residuals.filter { it <= MAXIMUM_NORMALIZED_PLANE_RESIDUAL }.toDoubleArray()
        val fraction = accepted.size.toDouble() / points.size
        if (accepted.size < required || fraction < MINIMUM_INLIER_FRACTION) return Unknown("GROUND_SUPPORT_REJECTED")
        val medianResidual = median(accepted)
        if (medianResidual > MAXIMUM_NORMALIZED_PLANE_RESIDUAL) return Unknown("GROUND_RESIDUAL_REJECTED")
        val scale = cameraHeightM / offset
        if (!scale.isFinite() || scale !in MINIMUM_SCALE..MAXIMUM_SCALE) return Unknown("SCALE_OUT_OF_RANGE")

        val finiteBuffer = DoubleArray(depth.size)
        var finiteCount = 0
        for (value in depth) {
            val converted = value.toDouble()
            if (converted.isFinite() && converted > 0.0) finiteBuffer[finiteCount++] = converted
        }
        if (finiteCount < 500) return Unknown("INSUFFICIENT_VALID_DEPTH")
        val finiteDepth = finiteBuffer.copyOf(finiteCount).also(DoubleArray::sort)
        val q10 = quantile(finiteDepth, 0.10)
        val q50 = quantile(finiteDepth, 0.50)
        val q90 = quantile(finiteDepth, 0.90)
        val features = doubleArrayOf(
            ln(scale), ln(cameraHeightM), normal.x, normal.y, normal.z, medianResidual,
            ln(q10), ln(q50), ln(q90), ln(q90 / q10),
        )
        return Geometry(
            offset,
            medianResidual,
            fraction,
            doubleArrayOf(normal.x, normal.y, normal.z),
            features,
        )
    }

    private class Workspace {
        private var rayWidth = -1
        private var rayHeight = -1
        private var rayFx = Double.NaN
        private var rayFy = Double.NaN
        private var rayCx = Double.NaN
        private var rayCy = Double.NaN
        private var rayX = DoubleArray(0)
        private var rayY = DoubleArray(0)
        private var candidateX = DoubleArray(0)
        private var candidateY = DoubleArray(0)
        private var candidateZ = DoubleArray(0)
        private val sampledX = DoubleArray(MAXIMUM_CANDIDATES)
        private val sampledY = DoubleArray(MAXIMUM_CANDIDATES)
        private val sampledZ = DoubleArray(MAXIMUM_CANDIDATES)
        private val firstIndices = IntArray(MAXIMUM_CANDIDATES)
        private val secondIndices = IntArray(MAXIMUM_CANDIDATES)
        private val residualScratch = DoubleArray(MAXIMUM_CANDIDATES)
        private var finiteScratch = DoubleArray(0)

        fun evaluate(
            depth: FloatArray,
            width: Int,
            height: Int,
            fx: Double,
            fy: Double,
            cx: Double,
            cy: Double,
            cameraHeightM: Double,
        ): Any {
            if (depth.size != width * height || fx <= 0.0 || fy <= 0.0) return Unknown("INVALID_INPUT")
            ensureRays(width, height, fx, fy, cx, cy)
            var candidateCount = 0
            var rayIndex = 0
            val startRow = ceil(LOWER_ROI_START_FRACTION * height).toInt()
            for (row in startRow until height step STRIDE) {
                for (column in 0 until width step STRIDE) {
                    val z = depth[row * width + column].toDouble()
                    if (z.isFinite() && z > 0.0) {
                        candidateX[candidateCount] = rayX[rayIndex] * z
                        candidateY[candidateCount] = rayY[rayIndex] * z
                        candidateZ[candidateCount] = z
                        candidateCount++
                    }
                    rayIndex++
                }
            }
            if (candidateCount < MINIMUM_CANDIDATES) return Unknown("INSUFFICIENT_GROUND_CANDIDATES")
            val pointCount: Int
            val px: DoubleArray
            val py: DoubleArray
            val pz: DoubleArray
            if (candidateCount <= MAXIMUM_CANDIDATES) {
                pointCount = candidateCount
                px = candidateX; py = candidateY; pz = candidateZ
            } else {
                pointCount = MAXIMUM_CANDIDATES
                repeat(MAXIMUM_CANDIDATES) { index ->
                    val selected = (index.toLong() * (candidateCount - 1)).div(MAXIMUM_CANDIDATES - 1).toInt()
                    sampledX[index] = candidateX[selected]
                    sampledY[index] = candidateY[selected]
                    sampledZ[index] = candidateZ[selected]
                }
                px = sampledX; py = sampledY; pz = sampledZ
            }
            for (index in 0 until pointCount) {
                residualScratch[index] = sqrt(px[index] * px[index] + py[index] * py[index] + pz[index] * pz[index])
            }
            val characteristic = medianRange(residualScratch, pointCount)
            if (!characteristic.isFinite() || characteristic <= 0.0) return Unknown("DEGENERATE_RELATIVE_DEPTH")
            val minimumHeight = max(Double.MIN_VALUE, characteristic * 1e-6)
            val minimumCrossNorm = max(Double.MIN_VALUE, characteristic * characteristic * 1e-12)
            val random = Random(RANSAC_SEED)
            var currentIndices = firstIndices
            var bestIndices = secondIndices
            var bestCount = -1
            var bestResidual = Double.POSITIVE_INFINITY
            repeat(RANSAC_ITERATIONS) {
                val a = random.nextInt(pointCount)
                var b = random.nextInt(pointCount)
                while (b == a) b = random.nextInt(pointCount)
                var c = random.nextInt(pointCount)
                while (c == a || c == b) c = random.nextInt(pointCount)
                val abx = px[b] - px[a]
                val aby = py[b] - py[a]
                val abz = pz[b] - pz[a]
                val acx = px[c] - px[a]
                val acy = py[c] - py[a]
                val acz = pz[c] - pz[a]
                var nx = aby * acz - abz * acy
                var ny = abz * acx - abx * acz
                var nz = abx * acy - aby * acx
                val normalNorm = sqrt(nx * nx + ny * ny + nz * nz)
                if (!normalNorm.isFinite() || normalNorm <= minimumCrossNorm) return@repeat
                nx /= normalNorm; ny /= normalNorm; nz /= normalNorm
                if (abs(ny) < MINIMUM_ABS_NORMAL_Y) return@repeat
                var offset = -(nx * px[a] + ny * py[a] + nz * pz[a])
                if (offset < 0.0) { nx = -nx; ny = -ny; nz = -nz; offset = -offset }
                if (!offset.isFinite() || offset <= minimumHeight) return@repeat
                var count = 0
                for (index in 0 until pointCount) {
                    val residual = abs(nx * px[index] + ny * py[index] + nz * pz[index] + offset) / offset
                    if (residual <= MAXIMUM_NORMALIZED_PLANE_RESIDUAL) {
                        currentIndices[count] = index
                        residualScratch[count] = residual
                        count++
                    }
                }
                val residual = if (count == 0) Double.POSITIVE_INFINITY else medianRange(residualScratch, count)
                if (count > bestCount || (count == bestCount && residual < bestResidual)) {
                    bestCount = count
                    bestResidual = residual
                    val swap = bestIndices; bestIndices = currentIndices; currentIndices = swap
                }
            }
            val required = max(MINIMUM_INLIERS, ceil(MINIMUM_INLIER_FRACTION * pointCount).toInt())
            if (bestCount < required) return Unknown("NO_GROUND_CONSENSUS")

            var centerX = 0.0; var centerY = 0.0; var centerZ = 0.0
            for (position in 0 until bestCount) {
                val index = bestIndices[position]
                centerX += px[index]; centerY += py[index]; centerZ += pz[index]
            }
            centerX /= bestCount; centerY /= bestCount; centerZ /= bestCount
            val covariance = Array(3) { DoubleArray(3) }
            for (position in 0 until bestCount) {
                val index = bestIndices[position]
                val dx = px[index] - centerX
                val dy = py[index] - centerY
                val dz = pz[index] - centerZ
                covariance[0][0] += dx * dx; covariance[0][1] += dx * dy; covariance[0][2] += dx * dz
                covariance[1][0] += dy * dx; covariance[1][1] += dy * dy; covariance[1][2] += dy * dz
                covariance[2][0] += dz * dx; covariance[2][1] += dz * dy; covariance[2][2] += dz * dz
            }
            var normal = smallestEigenvector(covariance)
            var offset = -(normal.x * centerX + normal.y * centerY + normal.z * centerZ)
            if (offset < 0.0) { normal *= -1.0; offset = -offset }
            if (!offset.isFinite() || offset <= minimumHeight) return Unknown("DEGENERATE_RELATIVE_HEIGHT")
            if (abs(normal.y) < MINIMUM_ABS_NORMAL_Y) return Unknown("GROUND_ORIENTATION_REJECTED")
            var acceptedCount = 0
            for (index in 0 until pointCount) {
                val residual = abs(normal.x * px[index] + normal.y * py[index] + normal.z * pz[index] + offset) / offset
                if (residual <= MAXIMUM_NORMALIZED_PLANE_RESIDUAL) residualScratch[acceptedCount++] = residual
            }
            val fraction = acceptedCount.toDouble() / pointCount
            if (acceptedCount < required || fraction < MINIMUM_INLIER_FRACTION) return Unknown("GROUND_SUPPORT_REJECTED")
            val medianResidual = medianRange(residualScratch, acceptedCount)
            if (medianResidual > MAXIMUM_NORMALIZED_PLANE_RESIDUAL) return Unknown("GROUND_RESIDUAL_REJECTED")
            val scale = cameraHeightM / offset
            if (!scale.isFinite() || scale !in MINIMUM_SCALE..MAXIMUM_SCALE) return Unknown("SCALE_OUT_OF_RANGE")

            if (finiteScratch.size < depth.size) finiteScratch = DoubleArray(depth.size)
            var finiteCount = 0
            for (value in depth) {
                val converted = value.toDouble()
                if (converted.isFinite() && converted > 0.0) finiteScratch[finiteCount++] = converted
            }
            if (finiteCount < 500) return Unknown("INSUFFICIENT_VALID_DEPTH")
            val q10 = quantileRange(finiteScratch, finiteCount, 0.10)
            val q50 = quantileRange(finiteScratch, finiteCount, 0.50)
            val q90 = quantileRange(finiteScratch, finiteCount, 0.90)
            val features = doubleArrayOf(
                ln(scale), ln(cameraHeightM), normal.x, normal.y, normal.z, medianResidual,
                ln(q10), ln(q50), ln(q90), ln(q90 / q10),
            )
            return Geometry(offset, medianResidual, fraction,
                doubleArrayOf(normal.x, normal.y, normal.z), features)
        }

        private fun ensureRays(width: Int, height: Int, fx: Double, fy: Double, cx: Double, cy: Double) {
            if (width == rayWidth && height == rayHeight && fx == rayFx && fy == rayFy && cx == rayCx && cy == rayCy) return
            rayWidth = width; rayHeight = height; rayFx = fx; rayFy = fy; rayCx = cx; rayCy = cy
            val rows = ((height - ceil(LOWER_ROI_START_FRACTION * height).toInt()) + STRIDE - 1) / STRIDE
            val columns = (width + STRIDE - 1) / STRIDE
            val count = rows * columns
            if (rayX.size < count) {
                rayX = DoubleArray(count); rayY = DoubleArray(count)
                candidateX = DoubleArray(count); candidateY = DoubleArray(count); candidateZ = DoubleArray(count)
            }
            var index = 0
            val startRow = ceil(LOWER_ROI_START_FRACTION * height).toInt()
            for (row in startRow until height step STRIDE) for (column in 0 until width step STRIDE) {
                rayX[index] = (column - cx) / fx
                rayY[index] = (row - cy) / fy
                index++
            }
        }

        private fun medianRange(values: DoubleArray, count: Int): Double {
            val middle = count / 2
            return if (count % 2 == 1) select(values, count, middle) else {
                val upper = select(values, count, middle)
                val lower = select(values, count, middle - 1)
                0.5 * (lower + upper)
            }
        }

        private fun quantileRange(values: DoubleArray, count: Int, fraction: Double): Double {
            val position = fraction * (count - 1)
            val lower = position.toInt()
            val upper = minOf(lower + 1, count - 1)
            val lowerValue = select(values, count, lower)
            val upperValue = if (upper == lower) lowerValue else select(values, count, upper)
            return lowerValue * (1.0 - (position - lower)) + upperValue * (position - lower)
        }

        private fun select(values: DoubleArray, count: Int, target: Int): Double {
            var left = 0
            var right = count - 1
            while (left < right) {
                val pivot = values[(left + right) ushr 1]
                var low = left
                var high = right
                while (low <= high) {
                    while (values[low] < pivot) low++
                    while (values[high] > pivot) high--
                    if (low <= high) {
                        val swap = values[low]
                        values[low] = values[high]
                        values[high] = swap
                        low++
                        high--
                    }
                }
                when {
                    target <= high -> right = high
                    target >= low -> left = low
                    else -> return values[target]
                }
            }
            return values[target]
        }
    }

    private fun distinctTriple(random: Random, bound: Int): IntArray {
        val a = random.nextInt(bound)
        var b = random.nextInt(bound)
        while (b == a) b = random.nextInt(bound)
        var c = random.nextInt(bound)
        while (c == a || c == b) c = random.nextInt(bound)
        return intArrayOf(a, b, c)
    }

    private fun smallestEigenvector(input: Array<DoubleArray>): Point {
        val matrix = Array(3) { input[it].copyOf() }
        val vectors = Array(3) { row -> DoubleArray(3) { column -> if (row == column) 1.0 else 0.0 } }
        repeat(32) {
            var p = 0
            var q = 1
            if (abs(matrix[0][2]) > abs(matrix[p][q])) { p = 0; q = 2 }
            if (abs(matrix[1][2]) > abs(matrix[p][q])) { p = 1; q = 2 }
            if (abs(matrix[p][q]) < 1e-12) return@repeat
            val angle = 0.5 * kotlin.math.atan2(2.0 * matrix[p][q], matrix[q][q] - matrix[p][p])
            val cosine = kotlin.math.cos(angle)
            val sine = kotlin.math.sin(angle)
            for (index in 0..2) {
                val mip = matrix[index][p]
                val miq = matrix[index][q]
                matrix[index][p] = cosine * mip - sine * miq
                matrix[index][q] = sine * mip + cosine * miq
            }
            for (index in 0..2) {
                val mpi = matrix[p][index]
                val mqi = matrix[q][index]
                matrix[p][index] = cosine * mpi - sine * mqi
                matrix[q][index] = sine * mpi + cosine * mqi
            }
            for (index in 0..2) {
                val vip = vectors[index][p]
                val viq = vectors[index][q]
                vectors[index][p] = cosine * vip - sine * viq
                vectors[index][q] = sine * vip + cosine * viq
            }
        }
        val eigenIndex = (0..2).minBy { matrix[it][it] }
        val result = Point(vectors[0][eigenIndex], vectors[1][eigenIndex], vectors[2][eigenIndex])
        return result / result.norm()
    }

    private fun median(values: DoubleArray): Double {
        values.sort()
        val middle = values.size / 2
        return if (values.size % 2 == 1) values[middle] else 0.5 * (values[middle - 1] + values[middle])
    }

    private fun quantile(sorted: DoubleArray, fraction: Double): Double {
        val position = fraction * (sorted.size - 1)
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.lastIndex)
        return sorted[lower] * (1.0 - (position - lower)) + sorted[upper] * (position - lower)
    }

    private data class Point(val x: Double, val y: Double, val z: Double) {
        operator fun minus(other: Point) = Point(x - other.x, y - other.y, z - other.z)
        operator fun div(value: Double) = Point(x / value, y / value, z / value)
        operator fun times(value: Double) = Point(x * value, y * value, z * value)
        fun dot(other: Point) = x * other.x + y * other.y + z * other.z
        fun cross(other: Point) = Point(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x)
        fun norm() = sqrt(x * x + y * y + z * z)
    }

    private const val LOWER_ROI_START_FRACTION = 0.55
    private const val STRIDE = 4
    private const val RANSAC_SEED = 1729L
    private const val RANSAC_ITERATIONS = 240
    private const val MAXIMUM_CANDIDATES = 5000
    private const val MINIMUM_CANDIDATES = 100
    private const val MINIMUM_INLIERS = 80
    private const val MINIMUM_INLIER_FRACTION = 0.08
    private const val MINIMUM_ABS_NORMAL_Y = 0.55
    private const val MAXIMUM_NORMALIZED_PLANE_RESIDUAL = 0.035
    private const val MINIMUM_SCALE = 0.25
    private const val MAXIMUM_SCALE = 4.0
    private val workspace = ThreadLocal.withInitial(::Workspace)
}
