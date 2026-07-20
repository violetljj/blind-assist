package com.linnan.blindassist.ustrf

/** Explicit, deterministic faults for synthetic or later receipt-backed fast-loop replays. */
enum class UstrfFastLoopFault {
    POSE_LOST,
    CAPTURE_UNAVAILABLE,
    GEOMETRY_UNAVAILABLE,
    MOTION_UNAVAILABLE,
    PERCEPTION_SOURCE_FRAME_MISMATCH,
    PERCEPTION_EXPIRED
}

data class UstrfFaultScenario(
    val input: UstrfSessionInput,
    val faults: Set<UstrfFastLoopFault> = emptySet()
)

data class UstrfFaultReplayRecord(
    val frame: UstrfFrameStamp,
    val injectedFaults: Set<UstrfFastLoopFault>,
    val result: UstrfSessionRecord
)

/**
 * Runs explicit fault cases through the same [UstrfSafetySession] composition root as an Adapter.
 * It is intentionally unable to convert a fault into a recovery; scenarios must supply the next
 * receipt explicitly, preserving monotonic frame ordering and repeatable traces.
 */
class UstrfControlledFaultReplay(
    private val session: UstrfSafetySession = UstrfSafetySession()
) {
    fun run(scenarios: List<UstrfFaultScenario>): List<UstrfFaultReplayRecord> = scenarios.map { scenario ->
        val injected = scenario.input.withFaults(scenario.faults)
        UstrfFaultReplayRecord(scenario.input.frame, scenario.faults, session.evaluate(injected))
    }

    private fun UstrfSessionInput.withFaults(faults: Set<UstrfFastLoopFault>): UstrfSessionInput {
        var injectedHealth = health
        if (UstrfFastLoopFault.POSE_LOST in faults) injectedHealth = injectedHealth.copy(pose = UstrfPoseState.LOST)
        if (UstrfFastLoopFault.CAPTURE_UNAVAILABLE in faults) injectedHealth = injectedHealth.copy(capture = UstrfEvidenceState.MISSING)
        if (UstrfFastLoopFault.GEOMETRY_UNAVAILABLE in faults) injectedHealth = injectedHealth.copy(geometry = UstrfEvidenceState.MISSING)
        if (UstrfFastLoopFault.MOTION_UNAVAILABLE in faults) injectedHealth = injectedHealth.copy(motion = UstrfEvidenceState.MISSING)

        var injectedPerception = perception
        var injectedDecisionAtNs = decisionAtNs
        if (injectedPerception is UstrfPerceptionAssembly.Available) {
            var packet = injectedPerception.packet
            if (UstrfFastLoopFault.PERCEPTION_SOURCE_FRAME_MISMATCH in faults) {
                packet = packet.copy(sourceFrame = packet.sourceFrame.copy(frameId = packet.sourceFrame.frameId + 1L))
            }
            if (UstrfFastLoopFault.PERCEPTION_EXPIRED in faults) {
                injectedDecisionAtNs = maxOf(injectedDecisionAtNs, packet.validUntilNs + 1L)
            }
            injectedPerception = UstrfPerceptionAssembly.Available(packet)
        }
        return copy(health = injectedHealth, perception = injectedPerception, decisionAtNs = injectedDecisionAtNs)
    }
}
