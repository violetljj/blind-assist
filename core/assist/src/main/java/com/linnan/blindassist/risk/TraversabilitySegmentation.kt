package com.linnan.blindassist.risk

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import kotlin.math.max
import kotlin.math.abs

enum class TraversabilityClass {
    SAFE_TO_WALK,
    NOT_SAFE_TO_WALK,
    OBSTACLE
}

/**
 * Keeps the connected-component/risk corridor logic independent from the
 * originating segmentation dataset.  The production-facing code only sees
 * [DetectionSource.SEGMENTATION]; SANPO ids and learned-model ids remain
 * benchmark implementation details.
 */
interface TraversabilityTaxonomy {
    fun traversabilityFor(classId: Int): TraversabilityClass
    fun isNavigationHazard(classId: Int): Boolean
    fun riskLabelFor(classId: Int): String?

    /** A boundary may be emitted only when it enters the central walking corridor. */
    fun permitsBoundaryDetection(classId: Int): Boolean = false
    fun isBoundaryEvidence(classId: Int): Boolean = false
}

/**
 * BlindAssist navigation mapping for SANPO semantic ids.
 *
 * This intentionally differs from SANPO's paper accessibility collapse for stairs:
 * SANPO maps stairs to safe-to-walk, while BlindAssist keeps stairs as an explicit
 * mobility hazard so a step transition cannot disappear inside the free-space mask.
 */
object BlindAssistSanpoTaxonomy : TraversabilityTaxonomy {
    private val safeToWalkIds = setOf(3, 5, 6, 17, 30)
    private val navigationHazardIds = setOf(2, 4, 9, 10, 11, 15, 18, 20, 24, 26)
    private val obstacleIds = setOf(
        4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 28, 29
    )

    override fun traversabilityFor(classId: Int): TraversabilityClass {
        return when {
            classId in safeToWalkIds -> TraversabilityClass.SAFE_TO_WALK
            classId in obstacleIds -> TraversabilityClass.OBSTACLE
            else -> TraversabilityClass.NOT_SAFE_TO_WALK
        }
    }

    override fun isNavigationHazard(classId: Int): Boolean = classId in navigationHazardIds

    override fun isBoundaryEvidence(classId: Int): Boolean = classId == 2

    override fun riskLabelFor(classId: Int): String? {
        return when (classId) {
            2 -> "curb"
            4 -> "road barrier"
            9 -> "hand rail"
            10 -> "opening door"
            11 -> "opening gate"
            15 -> "stairs"
            18 -> "inaccessible surface"
            20 -> "generic obstacle"
            24 -> "pole"
            26 -> "bike rack"
            else -> null
        }
    }
}

/** Four-class contract used by the benchmark-only MobileNetV3 LR-ASPP model. */
object BlindAssistLearnedTraversabilityTaxonomy : TraversabilityTaxonomy {
    const val WALKABLE = 0
    const val BOUNDARY_STEP_CURB = 1
    const val OBSTACLE = 2
    const val UNKNOWN_NONWALKABLE = 3

    override fun traversabilityFor(classId: Int): TraversabilityClass = when (classId) {
        WALKABLE -> TraversabilityClass.SAFE_TO_WALK
        OBSTACLE -> TraversabilityClass.OBSTACLE
        BOUNDARY_STEP_CURB, UNKNOWN_NONWALKABLE -> TraversabilityClass.NOT_SAFE_TO_WALK
        else -> TraversabilityClass.NOT_SAFE_TO_WALK
    }

    override fun isNavigationHazard(classId: Int): Boolean =
        classId == BOUNDARY_STEP_CURB || classId == OBSTACLE

    override fun riskLabelFor(classId: Int): String? = when (classId) {
        BOUNDARY_STEP_CURB -> "boundary step curb"
        OBSTACLE -> "segmentation obstacle"
        else -> null
    }

    override fun permitsBoundaryDetection(classId: Int): Boolean = classId == BOUNDARY_STEP_CURB
    override fun isBoundaryEvidence(classId: Int): Boolean = classId == BOUNDARY_STEP_CURB
}

data class TraversabilityAnalyzerConfig(
    val corridorTopRatio: Float = 0.42f,
    val corridorTopHalfWidthRatio: Float = 0.16f,
    val corridorBottomHalfWidthRatio: Float = 0.42f,
    val minimumRegionPixels: Int = 12,
    val minimumRegionAreaRatio: Float = 0.0008f,
    val minimumCenterOverlapRatio: Float = 0.10f,
    val curbMinimumCenterOverlapRatio: Float = 0.35f,
    val minimumBottomRatio: Float = 0.42f,
    val curbMinimumBottomRatio: Float = 0.62f,
    val boundaryMaximumCenterOverlapRatio: Float = 0.34f,
    val boundaryMinimumAspectRatio: Float = 3.0f,
    val boundaryEdgeAttachmentRatio: Float = 0.18f,
    val analysisSize: Int = 256
) {
    init {
        require(corridorTopRatio in 0f..<1f)
        require(corridorTopHalfWidthRatio in 0f..0.5f)
        require(corridorBottomHalfWidthRatio in 0f..0.5f)
        require(minimumRegionPixels > 0)
        require(minimumRegionAreaRatio >= 0f)
        require(analysisSize in 32..512)
    }
}

data class DenseSemanticMask(
    val width: Int,
    val height: Int,
    val classIds: IntArray
) {
    init {
        require(width > 0 && height > 0)
        require(classIds.size == width * height)
    }
}

data class TraversabilityAnalysis(
    val safeCoverage: Float,
    val notSafeCoverage: Float,
    val obstacleCoverage: Float,
    val corridorPixelCount: Int,
    val riskDetections: List<Detection>
)

class TraversabilitySegmentationAnalyzer(
    private val config: TraversabilityAnalyzerConfig = TraversabilityAnalyzerConfig(),
    private val taxonomy: TraversabilityTaxonomy = BlindAssistSanpoTaxonomy
) {
    private var corridorBuffer = BooleanArray(0)
    private var visitedBuffer = BooleanArray(0)
    private var queueBuffer = IntArray(0)
    private var bufferAllocationCount = 0

    val allocations: Int get() = bufferAllocationCount

    fun analyze(mask: DenseSemanticMask, frameSize: FrameSize): TraversabilityAnalysis {
        ensureBuffers(mask.classIds.size)
        val corridor = corridorBuffer.also { it.fill(false) }
        var safe = 0
        var notSafe = 0
        var obstacle = 0
        var corridorCount = 0
        val top = (mask.height * config.corridorTopRatio).toInt().coerceIn(0, mask.height - 1)
        for (y in top until mask.height) {
            val progress = (y - top).toFloat() / max(1, mask.height - 1 - top)
            val halfWidthRatio = config.corridorTopHalfWidthRatio +
                (config.corridorBottomHalfWidthRatio - config.corridorTopHalfWidthRatio) * progress
            val left = (mask.width * (0.5f - halfWidthRatio)).toInt().coerceIn(0, mask.width - 1)
            val right = (mask.width * (0.5f + halfWidthRatio)).toInt().coerceIn(left + 1, mask.width)
            for (x in left until right) {
                val index = y * mask.width + x
                corridor[index] = true
                corridorCount += 1
                when (taxonomy.traversabilityFor(mask.classIds[index])) {
                    TraversabilityClass.SAFE_TO_WALK -> safe += 1
                    TraversabilityClass.NOT_SAFE_TO_WALK -> notSafe += 1
                    TraversabilityClass.OBSTACLE -> obstacle += 1
                }
            }
        }

        val detections = extractRiskRegions(mask, corridor, frameSize)
        return TraversabilityAnalysis(
            safeCoverage = ratio(safe, corridorCount),
            notSafeCoverage = ratio(notSafe, corridorCount),
            obstacleCoverage = ratio(obstacle, corridorCount),
            corridorPixelCount = corridorCount,
            riskDetections = detections
        )
    }

    private fun extractRiskRegions(
        mask: DenseSemanticMask,
        corridor: BooleanArray,
        frameSize: FrameSize
    ): List<Detection> {
        val visited = visitedBuffer.also { it.fill(false) }
        val queue = queueBuffer
        val detections = mutableListOf<Detection>()
        for (start in mask.classIds.indices) {
            val classId = mask.classIds[start]
            if (visited[start] || !taxonomy.isNavigationHazard(classId)) continue
            visited[start] = true
            var head = 0
            var tail = 0
            queue[tail++] = start
            var pixels = 0
            var corridorPixels = 0
            var minX = mask.width
            var minY = mask.height
            var maxX = 0
            var maxY = 0
            while (head < tail) {
                val index = queue[head++]
                val x = index % mask.width
                val y = index / mask.width
                pixels += 1
                if (corridor[index]) corridorPixels += 1
                minX = minOf(minX, x)
                minY = minOf(minY, y)
                maxX = maxOf(maxX, x)
                maxY = maxOf(maxY, y)
                enqueue(index - 1, x > 0, classId, mask, visited, queue, tail).also { tail = it }
                enqueue(index + 1, x + 1 < mask.width, classId, mask, visited, queue, tail).also { tail = it }
                enqueue(index - mask.width, y > 0, classId, mask, visited, queue, tail).also { tail = it }
                enqueue(index + mask.width, y + 1 < mask.height, classId, mask, visited, queue, tail).also { tail = it }
            }
            val areaRatio = pixels.toFloat() / mask.classIds.size
            val centerOverlapRatio = corridorPixels.toFloat() / max(1, pixels)
            val bottomRatio = (maxY + 1).toFloat() / mask.height
            val label = taxonomy.riskLabelFor(classId)
            val isBoundaryEvidence = taxonomy.isBoundaryEvidence(classId)
            val minimumOverlap = if (isBoundaryEvidence) config.curbMinimumCenterOverlapRatio else config.minimumCenterOverlapRatio
            val minimumBottom = if (isBoundaryEvidence) config.curbMinimumBottomRatio else config.minimumBottomRatio
            val isBoundaryLikeGenericObstacle = (label == "generic obstacle" || isBoundaryEvidence) &&
                (maxX - minX + 1).toFloat() / max(1, maxY - minY + 1) >= config.boundaryMinimumAspectRatio &&
                centerOverlapRatio <= config.boundaryMaximumCenterOverlapRatio &&
                (
                    minX <= mask.width * config.boundaryEdgeAttachmentRatio ||
                    maxX + 1 >= mask.width * (1f - config.boundaryEdgeAttachmentRatio))
            // SANPO curb stays diagnostic-only. The learned boundary class can enter the
            // risk path only after it intrudes into the central corridor; temporal logic
            // still decides whether it becomes actionable.
            val passesGate = centerOverlapRatio >= minimumOverlap && bottomRatio >= minimumBottom &&
                (!isBoundaryEvidence || (taxonomy.permitsBoundaryDetection(classId) && !isBoundaryLikeGenericObstacle))
            if (passesGate && pixels >= config.minimumRegionPixels && areaRatio >= config.minimumRegionAreaRatio && label != null) {
                val scaleX = frameSize.width.toFloat() / mask.width
                val scaleY = frameSize.height.toFloat() / mask.height
                detections += Detection(
                    classId = SANPO_CLASS_ID_OFFSET + classId,
                    label = label,
                    confidence = 1f,
                    boundingBox = BoundingBox(
                        left = minX * scaleX,
                        top = minY * scaleY,
                        right = (maxX + 1) * scaleX,
                        bottom = (maxY + 1) * scaleY
                    ),
                    frameSize = frameSize,
                    source = DetectionSource.SEGMENTATION,
                    temporalPromotionEligible = !isBoundaryLikeGenericObstacle
                )
            }
        }
        // Forward only the strongest center-path segmentation candidate. This prevents broad
        // peripheral regions from outranking a smaller obstacle in the walking line.
        return detections.sortedWith(
            compareBy<Detection> { abs(it.boundingBox.centerX / it.frameSize.width - 0.5f) }
                .thenByDescending { it.boundingBox.bottom / it.frameSize.height }
                .thenBy { it.areaRatio }
        ).take(1)
    }

    private fun enqueue(
        index: Int,
        valid: Boolean,
        classId: Int,
        mask: DenseSemanticMask,
        visited: BooleanArray,
        queue: IntArray,
        tail: Int
    ): Int {
        if (!valid || visited[index] || mask.classIds[index] != classId) return tail
        visited[index] = true
        queue[tail] = index
        return tail + 1
    }

    private fun ratio(value: Int, total: Int): Float = if (total == 0) 0f else value.toFloat() / total

    private fun ensureBuffers(size: Int) {
        if (corridorBuffer.size == size) return
        corridorBuffer = BooleanArray(size)
        visitedBuffer = BooleanArray(size)
        queueBuffer = IntArray(size)
        bufferAllocationCount += 1
    }

    companion object {
        private const val SANPO_CLASS_ID_OFFSET = 10_000
    }
}
