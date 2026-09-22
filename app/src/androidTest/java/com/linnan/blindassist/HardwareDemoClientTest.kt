package com.linnan.blindassist

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.device.glasses.HardwareDemoClient
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream

@RunWith(AndroidJUnit4::class)
class HardwareDemoClientTest {
    private fun fixture(mode: String = "live", active: Boolean = true): JSONObject = JSONObject().apply {
        put("mode", mode); put("run_id", "test-run"); put("duration_ms", 2000)
        put("recording", JSONObject().put("active", active))
        put("camera", JSONObject().put("seq", 1).put("age_ms", 0).put("stale", false).put("url", "/api/frame?run=test-run"))
        put("tof", JSONObject().put("seq", 1).put("age_ms", 0).put("stale", false).put("rows", 8).put("cols", 8)
            .put("cells", JSONArray().apply {
                repeat(64) { zone -> put(JSONObject().put("zone", zone).put("raw_mm", 1000)
                    .put("status", 5).put("targets", 1).put("quality", "KNOWN")) }
            }))
    }

    private fun jpeg(): ByteArray = ByteArrayOutputStream().use { output ->
        val bitmap = Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, output)
        bitmap.recycle()
        output.toByteArray()
    }

    @Test fun liveAlertsButFrozenSequencesExpireAndRunChangeResets() {
        var now = 100L
        val state = fixture()
        val image = jpeg()
        val client = HardwareDemoClient({ if (it.startsWith("/api/state?")) state.toString().toByteArray() else image }, { now })
        assertTrue(client.poll().liveUsable)
        assertTrue(client.poll().decision.alert)
        now += 1501
        val frozen = client.poll()
        assertFalse(frozen.liveUsable)
        assertEquals("UNKNOWN", frozen.decision.state)
        state.put("run_id", "new-run")
        assertTrue(client.poll().liveUsable)
    }

    @Test fun replayCanDisplayDecisionsButIsNeverLiveAndStoppedCaptureIsUnknown() {
        val image = jpeg()
        for (mode in listOf("replay", "live")) {
            val state = fixture(mode, active = false)
            val client = HardwareDemoClient({ if (it.startsWith("/api/state?")) state.toString().toByteArray() else image }, { 100 })
            val result = client.poll()
            assertFalse(result.liveUsable)
            assertEquals(mode == "replay", result.decision.alert)
        }
    }

    @Test fun staleTofOrSlowNetworkCannotProduceLiveAlert() {
        val image = jpeg()
        for (slow in listOf(false, true)) {
            var now = 100L
            val state = fixture()
            if (!slow) state.getJSONObject("tof").put("stale", true)
            val client = HardwareDemoClient({ path ->
                if (path.startsWith("/api/state?")) state.toString().toByteArray()
                else { if (slow) now += 1501; image }
            }, { now })
            val result = client.poll()
            assertFalse(result.liveUsable)
            assertFalse(result.decision.alert)
        }
    }

    @Test fun malformedZoneGridFailsClosedWithoutInventingMissingCells() {
        val state = fixture()
        state.getJSONObject("tof").getJSONArray("cells").getJSONObject(63).put("zone", 62)
        val image = jpeg()
        val client = HardwareDemoClient({ if (it.startsWith("/api/state?")) state.toString().toByteArray() else image }, { 100 })
        assertEquals("UNKNOWN", client.poll().decision.state)
    }
}
