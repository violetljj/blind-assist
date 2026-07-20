package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path

class UstrfSyntheticCorridorSafetyReplayTest {
    @Test
    fun generatedBodyCapsuleTruthReplaysThroughTheKernelWithNoFalseClearStops() {
        val root = System.getenv("USTRF_SYNTHETIC_CORRIDOR_SAFETY_REPLAY_ROOT") ?: return
        val metrics = UstrfSyntheticCorridorSafetyReplay().replay(Path.of(root))
        assertEquals(256, metrics.sceneCount)
        assertEquals(metrics.sceneCount, metrics.actionMatchCount)
        assertEquals(metrics.eligibleCorridorSelectionCount, metrics.matchingCorridorSelectionCount)
        assertEquals(metrics.expectedStopCount, metrics.actualStopCount)
        assertEquals(0, metrics.clearStopCount)
        assertEquals(metrics.faultSceneCount, metrics.faultStopCount)
        System.getenv("USTRF_SYNTHETIC_CORRIDOR_SAFETY_REPLAY_OUTPUT")?.let { output ->
            Path.of(output).parent?.let(Files::createDirectories)
            Files.writeString(Path.of(output), """{
  "format": "blindassist_ustrf_synthetic_corridor_safety_kotlin_replay_v1",
  "scene_count": ${metrics.sceneCount},
  "expected_stop_count": ${metrics.expectedStopCount},
  "actual_stop_count": ${metrics.actualStopCount},
  "clear_stop_count": ${metrics.clearStopCount},
  "action_match_count": ${metrics.actionMatchCount},
  "eligible_corridor_selection_count": ${metrics.eligibleCorridorSelectionCount},
  "matching_corridor_selection_count": ${metrics.matchingCorridorSelectionCount},
  "fault_scene_count": ${metrics.faultSceneCount},
  "fault_stop_count": ${metrics.faultStopCount},
  "production_authority": false
}
""")
        }
    }
}
