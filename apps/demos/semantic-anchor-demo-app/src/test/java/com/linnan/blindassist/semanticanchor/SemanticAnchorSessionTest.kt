package com.linnan.blindassist.semanticanchor

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class SemanticAnchorSessionTest {
    @Test
    fun markerRequiresFreshSemanticEvidenceForLockAndReacquisition() {
        val target = AnchorTarget(AnchorMode.MARKER, "BLINDASSIST:ANCHOR:17")
        val session = SemanticAnchorSession(target)

        assertThat(session.observe(observation("BLINDASSIST:ANCHOR:99")).phase).isEqualTo(AnchorPhase.SEARCH)
        assertThat(session.observe(observation(target.value)).phase).isEqualTo(AnchorPhase.SEARCH)
        assertThat(session.observe(observation(target.value)).phase).isEqualTo(AnchorPhase.LOCKED)

        repeat(4) { assertThat(session.observe(observation()).phase).isEqualTo(AnchorPhase.LOCKED) }
        assertThat(session.observe(observation()).phase).isEqualTo(AnchorPhase.LOST)
        assertThat(session.observe(observation("BLINDASSIST:ANCHOR:99")).phase).isEqualTo(AnchorPhase.LOST)
        assertThat(session.observe(observation(target.value)).phase).isEqualTo(AnchorPhase.LOST)
        val reacquired = session.observe(observation(target.value))

        assertThat(reacquired.phase).isEqualTo(AnchorPhase.REACQUIRED)
        assertThat(reacquired.lockCount).isEqualTo(1)
        assertThat(reacquired.reacquisitionCount).isEqualTo(1)
    }

    @Test
    fun ocrUsesNormalizedSubstringButNeverNonTargetText() {
        val session = SemanticAnchorSession(AnchorTarget(AnchorMode.OCR, "Room 302"))

        assertThat(session.observe(observation("ROOM 301 EXIT")).targetVisible).isFalse()
        assertThat(session.observe(observation("Welcome — room 302 / east wing")).targetVisible).isTrue()
        assertThat(session.observe(observation("ROOM\n302")).phase).isEqualTo(AnchorPhase.LOCKED)
    }

    @Test
    fun resetClearsAllIdentityState() {
        val target = AnchorTarget(AnchorMode.MARKER, "A17")
        val session = SemanticAnchorSession(target)
        session.observe(observation(target.value))
        session.observe(observation(target.value))

        val reset = session.reset(AnchorTarget(AnchorMode.OCR, "ROOM 302"))

        assertThat(reset.phase).isEqualTo(AnchorPhase.SEARCH)
        assertThat(reset.frameCount).isEqualTo(0)
        assertThat(reset.lockCount).isEqualTo(0)
        assertThat(reset.reacquisitionCount).isEqualTo(0)
    }

    @Test
    fun repeatedExactIdRemainsAmbiguous() {
        val target = AnchorTarget(AnchorMode.MARKER, "A17")
        val session = SemanticAnchorSession(target)

        repeat(3) { session.observe(observation("A17", "A17")) }

        assertThat(session.state.phase).isEqualTo(AnchorPhase.SEARCH)
        assertThat(session.state.evidence).contains("AMBIGUOUS")
        assertThat(session.state.lockCount).isEqualTo(0)
    }

    @Test
    fun ocrGoalLockRejectsSilentSwitchThenReacquiresAndCompletes() {
        val target = AnchorTarget(AnchorMode.OCR, "ROOM 302")
        val session = SemanticAnchorSession(target)
        val left = candidate(target.value, 0.08, 0.40, 0.28, 0.52)

        assertThat(session.observe(boxedObservation(left)).phase).isEqualTo(AnchorPhase.TARGET_FOUND)
        val locked = session.observe(boxedObservation(left))
        assertThat(locked.phase).isEqualTo(AnchorPhase.LOCKED)
        assertThat(locked.guidance.command).isEqualTo("LEFT")

        val farRight = candidate(target.value, 0.72, 0.40, 0.92, 0.52)
        val rejected = session.observe(boxedObservation(farRight))
        assertThat(rejected.targetVisible).isFalse()
        assertThat(rejected.evidence).contains("SWITCH REJECTED")
        repeat(4) { session.observe(observation()) }
        assertThat(session.state.phase).isEqualTo(AnchorPhase.LOST)
        assertThat(session.state.guidance.command).startsWith("SCAN")

        assertThat(session.observe(boxedObservation(left)).phase).isEqualTo(AnchorPhase.LOST)
        val reacquired = session.observe(boxedObservation(left))
        assertThat(reacquired.phase).isEqualTo(AnchorPhase.REACQUIRED)
        assertThat(reacquired.reacquisitionCount).isEqualTo(1)

        val path = listOf(
            candidate(target.value, 0.16, 0.39, 0.36, 0.53),
            candidate(target.value, 0.25, 0.38, 0.47, 0.54),
            candidate(target.value, 0.34, 0.35, 0.66, 0.57),
            candidate(target.value, 0.34, 0.35, 0.66, 0.57),
            candidate(target.value, 0.34, 0.35, 0.66, 0.57),
        )
        path.forEach { session.observe(boxedObservation(it)) }

        assertThat(session.state.phase).isEqualTo(AnchorPhase.TASK_COMPLETE)
        assertThat(session.state.guidance.command).isEqualTo("TASK COMPLETE")
        assertThat(session.state.completionEvidenceFrames).isEqualTo(3)
    }

    private fun observation(vararg candidates: String) =
        AnchorObservation(candidates.map(::AnchorCandidate), "TEST")

    private fun boxedObservation(vararg candidates: AnchorCandidate) = AnchorObservation(candidates.toList(), "TEST")

    private fun candidate(
        value: String,
        left: Double,
        top: Double,
        right: Double,
        bottom: Double,
    ) = AnchorCandidate(value, NormalizedBox(left, top, right, bottom))
}
