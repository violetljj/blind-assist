package com.linnan.blindassist.semanticanchor

import com.google.common.truth.Truth.assertThat
import org.junit.Test

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

    private fun observation(vararg candidates: String) = AnchorObservation(candidates.toList(), "TEST")
}
