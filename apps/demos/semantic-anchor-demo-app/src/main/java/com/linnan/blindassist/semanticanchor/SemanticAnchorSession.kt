package com.linnan.blindassist.semanticanchor

internal enum class AnchorMode { MARKER, OCR }

internal enum class AnchorPhase { SEARCH, TARGET_FOUND, LOCKED, LOST, REACQUIRED, NEAR, TASK_COMPLETE }

internal data class NormalizedBox(
    val left: Double,
    val top: Double,
    val right: Double,
    val bottom: Double,
) {
    init {
        require(left in 0.0..1.0 && top in 0.0..1.0)
        require(right in 0.0..1.0 && bottom in 0.0..1.0)
        require(right > left && bottom > top)
    }

    val centerX: Double get() = (left + right) / 2.0
    val centerY: Double get() = (top + bottom) / 2.0
    val area: Double get() = (right - left) * (bottom - top)
    val visualScale: Double get() = kotlin.math.sqrt(area)

    fun iou(other: NormalizedBox): Double {
        val intersectionWidth = (minOf(right, other.right) - maxOf(left, other.left)).coerceAtLeast(0.0)
        val intersectionHeight = (minOf(bottom, other.bottom) - maxOf(top, other.top)).coerceAtLeast(0.0)
        val intersection = intersectionWidth * intersectionHeight
        return intersection / (area + other.area - intersection).coerceAtLeast(1e-9)
    }

    fun shiftX(delta: Double): NormalizedBox {
        val width = right - left
        val shiftedLeft = (left + delta).coerceIn(0.0, 1.0 - width)
        return copy(left = shiftedLeft, right = shiftedLeft + width)
    }

    fun blend(other: NormalizedBox, alpha: Double): NormalizedBox {
        val weight = alpha.coerceIn(0.0, 1.0)
        return NormalizedBox(
            left = (1.0 - weight) * left + weight * other.left,
            top = (1.0 - weight) * top + weight * other.top,
            right = (1.0 - weight) * right + weight * other.right,
            bottom = (1.0 - weight) * bottom + weight * other.bottom,
        )
    }

    companion object {
        fun fromPixels(
            left: Int,
            top: Int,
            right: Int,
            bottom: Int,
            frameWidth: Int,
            frameHeight: Int,
        ): NormalizedBox? {
            if (frameWidth <= 0 || frameHeight <= 0) return null
            val normalizedLeft = (left.toDouble() / frameWidth).coerceIn(0.0, 1.0)
            val normalizedTop = (top.toDouble() / frameHeight).coerceIn(0.0, 1.0)
            val normalizedRight = (right.toDouble() / frameWidth).coerceIn(0.0, 1.0)
            val normalizedBottom = (bottom.toDouble() / frameHeight).coerceIn(0.0, 1.0)
            if (normalizedRight <= normalizedLeft || normalizedBottom <= normalizedTop) return null
            return NormalizedBox(normalizedLeft, normalizedTop, normalizedRight, normalizedBottom)
        }
    }
}

internal data class AnchorCandidate(
    val value: String,
    val bounds: NormalizedBox? = null,
)

internal data class AnchorTarget(
    val mode: AnchorMode,
    val value: String,
) {
    val normalizedValue: String = normalize(value)

    init {
        require(normalizedValue.isNotEmpty()) { "semantic target must not be blank" }
    }

    fun matches(candidate: String): Boolean {
        val normalizedCandidate = normalize(candidate)
        return when (mode) {
            AnchorMode.MARKER -> normalizedCandidate == normalizedValue
            AnchorMode.OCR -> normalizedCandidate.contains(normalizedValue)
        }
    }

    fun matches(candidate: AnchorCandidate): Boolean = matches(candidate.value)

    companion object {
        fun normalize(value: String): String = value
            .uppercase()
            .replace(Regex("[^A-Z0-9]+"), " ")
            .trim()
            .replace(Regex("\\s+"), " ")
    }
}

internal data class AnchorObservation(
    val candidates: List<AnchorCandidate>,
    val source: String = "LIVE",
    val markerPoses: List<MarkerPoseEstimate> = emptyList(),
)

internal data class AnchorUiState(
    val target: AnchorTarget,
    val phase: AnchorPhase = AnchorPhase.SEARCH,
    val frameCount: Int = 0,
    val hitStreak: Int = 0,
    val missStreak: Int = 0,
    val lockCount: Int = 0,
    val reacquisitionCount: Int = 0,
    val targetVisible: Boolean = false,
    val evidence: String = "等待语义证据",
    val source: String = "IDLE",
    val beliefScore: Double = -2.2,
    val completionEvidenceFrames: Int = 0,
    val guidanceArm: GuidanceArm = GuidanceArm.PNP_POSE,
    val guidance: MarkerGuidance = MarkerGuidance(GuidancePhase.SEARCH, "SEARCH", "等待 exact QR ID"),
)

/**
 * Identity authority is the semantic target match only. Camera continuity and appearance never
 * promote SEARCH/LOST to an identity-bearing state.
 */
internal class SemanticAnchorSession(
    initialTarget: AnchorTarget,
    private val hitsToAcquire: Int = 2,
    private val missesToLose: Int = 5,
    private val poseController: MarkerPoseController = MarkerPoseController(),
) {
    private val ocrController = OcrGoalLockController(hitsToAcquire, missesToLose)

    var state: AnchorUiState = initialState(initialTarget)
        private set

    init {
        require(hitsToAcquire > 0)
        require(missesToLose > 0)
    }

    fun reset(target: AnchorTarget = state.target): AnchorUiState {
        poseController.reset()
        ocrController.reset()
        state = initialState(target)
        return state
    }

    fun setGuidanceArm(arm: GuidanceArm): AnchorUiState {
        poseController.reset()
        poseController.arm = arm
        state = state.copy(
            guidanceArm = arm,
            guidance = poseController.update(state.phase, state.guidance.estimate),
        )
        return state
    }

    fun observe(observation: AnchorObservation): AnchorUiState {
        return when (state.target.mode) {
            AnchorMode.MARKER -> observeMarker(observation)
            AnchorMode.OCR -> observeOcr(observation)
        }
    }

    private fun observeMarker(observation: AnchorObservation): AnchorUiState {
        val matches = observation.candidates.filter(state.target::matches)
        // Repeated exact IDs are not physical-instance authority. The controlled live setup must
        // keep one visible installation; ambiguity remains SEARCH/LOST.
        val hit = matches.size == 1
        val nextHits = if (hit) state.hitStreak + 1 else 0
        val nextMisses = if (hit) 0 else state.missStreak + 1
        var nextPhase = state.phase
        var nextLocks = state.lockCount
        var nextReacquisitions = state.reacquisitionCount

        when (state.phase) {
            AnchorPhase.SEARCH,
            AnchorPhase.TARGET_FOUND,
            -> if (nextHits >= hitsToAcquire) {
                nextPhase = AnchorPhase.LOCKED
                nextLocks += 1
            }
            AnchorPhase.LOCKED,
            AnchorPhase.REACQUIRED,
            AnchorPhase.NEAR,
            AnchorPhase.TASK_COMPLETE,
            -> if (nextMisses >= missesToLose) nextPhase = AnchorPhase.LOST
            AnchorPhase.LOST -> if (nextHits >= hitsToAcquire) {
                nextPhase = AnchorPhase.REACQUIRED
                nextReacquisitions += 1
            }
        }

        val matchingPose = observation.markerPoses.singleOrNull { state.target.matches(it.payload) }
        val nextGuidance = poseController.update(nextPhase, matchingPose)
        state = state.copy(
            phase = nextPhase,
            frameCount = state.frameCount + 1,
            hitStreak = nextHits,
            missStreak = nextMisses,
            lockCount = nextLocks,
            reacquisitionCount = nextReacquisitions,
            targetVisible = hit,
            evidence = when {
                hit -> "MATCH · ${matches.first().value}"
                matches.size > 1 -> "AMBIGUOUS · repeated exact ID"
                observation.candidates.isEmpty() -> "NO SEMANTIC EVIDENCE"
                else -> "NON-TARGET · ${observation.candidates.take(2).joinToString(" | ") { it.value }}"
            },
            source = observation.source,
            guidanceArm = poseController.arm,
            guidance = nextGuidance,
        )
        return state
    }

    private fun observeOcr(observation: AnchorObservation): AnchorUiState {
        val update = ocrController.update(state.target, observation.candidates)
        state = state.copy(
            phase = update.phase,
            frameCount = state.frameCount + 1,
            hitStreak = update.hitStreak,
            missStreak = update.missStreak,
            lockCount = state.lockCount + if (update.lockAcquired) 1 else 0,
            reacquisitionCount = state.reacquisitionCount + if (update.reacquired) 1 else 0,
            targetVisible = update.selected != null,
            evidence = update.evidence,
            source = observation.source,
            beliefScore = update.beliefScore,
            completionEvidenceFrames = update.completionEvidenceFrames,
            guidance = update.guidance,
        )
        return state
    }

    private fun initialState(target: AnchorTarget): AnchorUiState = AnchorUiState(
        target = target,
        guidanceArm = poseController.arm,
        guidance = if (target.mode == AnchorMode.OCR) {
            ocrController.initialGuidance()
        } else {
            MarkerGuidance(GuidancePhase.SEARCH, "SEARCH", "等待 exact QR ID")
        },
    )
}
