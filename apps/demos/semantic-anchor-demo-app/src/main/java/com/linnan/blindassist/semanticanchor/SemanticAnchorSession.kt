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
)

/**
 * Identity authority is the semantic target match only. Camera continuity and appearance never
 * promote SEARCH/LOST to an identity-bearing state.
 */
internal class SemanticAnchorSession(
    initialTarget: AnchorTarget,
    private val hitsToAcquire: Int = 2,
    private val missesToLose: Int = 5,
) {
    var state: AnchorUiState = AnchorUiState(target = initialTarget)
        private set

    init {
        require(hitsToAcquire > 0)
        require(missesToLose > 0)
    }

    fun reset(target: AnchorTarget = state.target): AnchorUiState {
        state = AnchorUiState(target = target)
        return state
    }

    fun observe(observation: AnchorObservation): AnchorUiState {
        val matches = observation.candidates.filter(state.target::matches)
        val hit = matches.isNotEmpty()
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
                observation.candidates.isEmpty() -> "NO SEMANTIC EVIDENCE"
                else -> "NON-TARGET · ${observation.candidates.take(2).joinToString(" | ")}"
            },
            source = observation.source,
        )
        return state
    }
}
