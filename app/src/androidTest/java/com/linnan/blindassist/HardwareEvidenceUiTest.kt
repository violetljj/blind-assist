package com.linnan.blindassist

import android.os.SystemClock
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.device.glasses.HardwareDemoSnapshot
import com.linnan.blindassist.device.glasses.HardwareEvidenceRecorder
import com.linnan.blindassist.device.glasses.HardwareEvidenceStore
import com.linnan.blindassist.risk.DemoTofCell
import com.linnan.blindassist.risk.TofCorridorDemo
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class HardwareEvidenceUiTest {
    @get:Rule val compose = createAndroidComposeRule<HardwareDemoActivity>()

    @Test fun savedScalarRecordReplaysWithoutLiveOverwriteAndCanReturnToLive() {
        val store = HardwareEvidenceStore(File(compose.activity.noBackupFilesDir, "hardware-evidence"))
        val cells = (0 until 64).map { DemoTofCell(it, 1000.0, 5, 1, "KNOWN") }
        val snapshot = HardwareDemoSnapshot("live", "synthetic-ui-test", null, 1, null,
            cells, 8, 8, TofCorridorDemo.evaluate(8, 8, cells, true), false, "synthetic fixture",
            cameraUsable = false, tofUsable = true)
        val saved = HardwareEvidenceRecorder("synthetic-ui-test").use { recorder ->
            recorder.offer(snapshot, 0)
            recorder.offer(snapshot.copy(tofSequence = 2), 2000)
            store.save(recorder.freeze())
        }
        try {
            compose.activityRule.scenario.recreate()
            compose.onNodeWithTag("hardware_evidence_toggle").performScrollTo().performClick()
            compose.waitUntil(5000) {
                compose.onAllNodesWithTag("hardware_evidence_open_0").fetchSemanticsNodes().isNotEmpty()
            }
            compose.onNodeWithTag("hardware_evidence_open_0").performScrollTo().performClick()
            compose.waitUntil(5000) {
                compose.onAllNodesWithTag("hardware_evidence_replay").fetchSemanticsNodes().isNotEmpty()
            }
            compose.onNodeWithTag("hardware_demo_speech").performScrollTo().assertIsNotEnabled()
            compose.onNodeWithTag("hardware_demo_vibration").performScrollTo().assertIsNotEnabled()
            compose.onNodeWithTag("hardware_evidence_play").performScrollTo().performClick()
            SystemClock.sleep(1200)
            compose.onNodeWithTag("hardware_evidence_replay").performScrollTo().assertIsDisplayed()
            compose.activityRule.scenario.recreate()
            compose.waitUntil(5000) {
                compose.onAllNodesWithTag("hardware_evidence_replay").fetchSemanticsNodes().isNotEmpty()
            }
            compose.onNodeWithTag("hardware_demo_speech").performScrollTo().assertIsNotEnabled()
            compose.onNodeWithTag("hardware_evidence_live").performScrollTo().performClick()
            compose.waitUntil(5000) {
                compose.onAllNodesWithTag("hardware_evidence_replay").fetchSemanticsNodes().isEmpty()
            }
        } finally { store.delete(saved.id) }
    }
}
