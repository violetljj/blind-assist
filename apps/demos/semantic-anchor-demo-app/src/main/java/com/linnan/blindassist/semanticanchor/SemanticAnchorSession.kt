package com.linnan.blindassist.semanticanchor

internal enum class AnchorMode { MARKER, OCR }

internal enum class AnchorPhase { SEARCH, LOCKED, LOST, REACQUIRED }

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

    companion object {
        fun normalize(value: String): String = value
            .uppercase()
            .replace(Regex("[^A-Z0-9]+"), " ")
            .trim()
            .replace(Regex("\\s+"), " ")
    }
}

internal data class AnchorObservation(
    val candidates: List<String>,
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
    var state: AnchorUiState = AnchorUiState(target = initialTarget)
        private set

    init {
        require(hitsToAcquire > 0)
        require(missesToLose > 0)
    }

    fun reset(target: AnchorTarget = state.target): AnchorUiState {
        poseController.reset()
        state = AnchorUiState(target = target, guidanceArm = poseController.arm)
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
            AnchorPhase.SEARCH -> if (nextHits >= hitsToAcquire) {
                nextPhase = AnchorPhase.LOCKED
                nextLocks += 1
            }
            AnchorPhase.LOCKED,
            AnchorPhase.REACQUIRED,
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
                hit -> "MATCH · ${matches.first()}"
                matches.size > 1 -> "AMBIGUOUS · repeated exact ID"
                observation.candidates.isEmpty() -> "NO SEMANTIC EVIDENCE"
                else -> "NON-TARGET · ${observation.candidates.take(2).joinToString(" | ")}"
            },
            source = observation.source,
            guidanceArm = poseController.arm,
            guidance = nextGuidance,
        )
        return state
    }
}
