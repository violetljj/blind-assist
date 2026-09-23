package com.linnan.blindassist

import android.graphics.Bitmap
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.device.glasses.*
import com.linnan.blindassist.risk.DemoTofCell
import com.linnan.blindassist.risk.DemoTofDecision
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class HardwareEvidenceRecorderTest {
    private fun snapshot(camera: Boolean = true, tof: Boolean = true, image: Bitmap? = null) =
        HardwareDemoSnapshot("live", "synthetic-test", 12, 23, image,
            listOf(DemoTofCell(0, 1000.0, 5, 1, "KNOWN"),
                DemoTofCell(1, Double.NaN, 0, 0, "INVALID")),
            8, 8, DemoTofDecision(tof, if (tof) "SAVED_TEST_DECISION" else "UNKNOWN",
                if (tof) 1000 else null, if (tof) listOf(0) else emptyList(), if (tof) 1 else 0),
            camera && tof, "synthetic input", camera, tof, 25, 40)

    private fun <T> withStore(block: (HardwareEvidenceStore, File) -> T): T {
        val cache = InstrumentationRegistry.getInstrumentation().targetContext.cacheDir.canonicalFile
        val dir = File(cache, "evidence-test-${UUID.randomUUID()}").canonicalFile
        check(dir.parentFile == cache && !dir.exists())
        check(dir.mkdir())
        try { return block(HardwareEvidenceStore(dir), dir) }
        finally {
            check(dir.deleteRecursively())
            check(!dir.exists())
        }
    }

    @Test fun ringBoundsAgeAndCountEvenWhenAvailabilityFlaps() {
        HardwareEvidenceRecorder("test").use { recorder ->
            repeat(350) { recorder.offer(snapshot(camera = it % 2 == 0), it.toLong()) }
            val limited = recorder.freeze()
            assertEquals(256, limited.frames.size)
            assertEquals(94L, limited.frames.first().atMs)
            recorder.clear()
            repeat(250) { recorder.offer(snapshot(), it * 100L) }
            val aged = recorder.freeze()
            assertTrue(aged.durationMs <= 20_000)
            assertTrue(aged.frames.size <= 256)
            assertEquals(4900L, aged.frames.first().atMs)
        }
    }

    @Test fun replayIsNotRecordedAndFrozenClipSurvivesRingChanges() {
        HardwareEvidenceRecorder("test").use { recorder ->
            recorder.offer(snapshot(), 1000)
            val frozen = recorder.freeze(55)
            recorder.offer(snapshot().copy(mode = "replay"), 1200)
            assertEquals(1, recorder.freeze().frames.size)
            recorder.offer(snapshot(camera = false), 1500)
            recorder.clear()
            assertEquals(1, frozen.frames.size)
            assertEquals(1000L, frozen.frames.single().atMs)
            assertEquals(55L, frozen.createdAtMs)
            assertTrue(runCatching { recorder.freeze() }.isFailure)
        }
    }

    @Test fun newTofAndDecisionTransitionsAreNotLostToDisplayRateLimit() {
        HardwareEvidenceRecorder("test").use { recorder ->
            recorder.offer(snapshot(), 1000)
            recorder.offer(snapshot().copy(tofSequence = 24), 1050)
            recorder.offer(snapshot().copy(tofSequence = 24,
                decision = DemoTofDecision(false, "UNKNOWN", null, emptyList(), 0)), 1060)
            assertEquals(listOf(1000L, 1050L, 1060L), recorder.freeze().frames.map { it.atMs })
        }
    }

    @Test fun roundTripPreservesInvalidCellsSavedDecisionAndOutputRequests() = withStore { store, _ ->
        HardwareEvidenceRecorder("synthetic-version").use { recorder ->
            val original = snapshot(camera = false)
            recorder.offer(original, 1000)
            val event = HardwareEvidenceEvent(1005, "ALERT", "测试提示", true, false)
            recorder.recordEvent(event)
            val id = store.save(recorder.freeze(123456)).id
            val loaded = store.load(id)
            assertEquals("synthetic-version", loaded.appVersion)
            assertEquals(123456L, loaded.createdAtMs)
            assertEquals(listOf(event), loaded.events)
            assertEquals(original.cells, loaded.frames.single().cells)
            assertTrue(loaded.frames.single().cells[1].rangeMm.isNaN())
            assertEquals(original.decision, loaded.frames.single().decision)
            assertEquals(original.decision, loaded.snapshotAt(0).decision)
            assertFalse(loaded.snapshotAt(0).liveUsable)
            assertEquals("replay", loaded.snapshotAt(0).mode)
            val json = JSONObject(store.readBytes(id).toString(Charsets.UTF_8))
            assertEquals("hardware-evidence-v1", json.getString("schema"))
            assertEquals("android_elapsed_realtime_ms", json.getString("clock"))
            assertTrue(json.getJSONArray("frames").getJSONObject(0).getJSONArray("cells")
                .getJSONObject(1).isNull("mm"))
            assertTrue(json.getJSONArray("events").getJSONObject(0).getBoolean("speech_queued"))
            assertEquals(1, store.list().size)
            store.delete(id)
            assertTrue(store.list().isEmpty())
        }
    }

    @Test fun replayNeverCarriesThumbnailAcrossCameraDropOrRecovery() {
        val bitmap = Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888)
        val jpeg = try { ByteArrayOutputStream().use {
            assertTrue(bitmap.compress(Bitmap.CompressFormat.JPEG, 80, it)); it.toByteArray()
        } } finally { bitmap.recycle() }
        HardwareEvidenceRecorder("test").use { recorder ->
            recorder.offer(snapshot(), 1000)
            val first = recorder.freeze().frames.single().copy(thumbnail = jpeg)
            val clip = HardwareEvidenceClip(1, "test", listOf(first,
                first.copy(atMs = 1100, cameraUsable = false, thumbnail = null),
                first.copy(atMs = 1200, thumbnail = null)), emptyList())
            val displayed = clip.snapshotAt(0).image
            assertNotNull(displayed)
            displayed?.recycle()
            assertNull(clip.snapshotAt(100).image)
            assertNull(clip.snapshotAt(200).image)
            assertFalse(clip.snapshotAt(200).liveUsable)
        }
    }

    @Test fun imagesAreOptInAndDisablingRemovesRingThumbnails() {
        val bitmap = Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888)
        try {
            HardwareEvidenceRecorder("test").use { recorder ->
                assertFalse(recorder.captureImages)
                recorder.offer(snapshot(image = bitmap), 1000)
                assertNull(recorder.freeze().frames.single().thumbnail)
                recorder.captureImages = true
                recorder.offer(snapshot(image = bitmap), 1500)
                val deadline = SystemClock.elapsedRealtime() + 3000
                while (recorder.freeze().frames.last().thumbnail == null && SystemClock.elapsedRealtime() < deadline) {
                    Thread.sleep(10)
                }
                assertNotNull("Thumbnail worker did not complete within 3 seconds", recorder.freeze().frames.last().thumbnail)
                recorder.captureImages = false
                assertTrue(recorder.freeze().frames.all { it.thumbnail == null })
            }
        } finally { bitmap.recycle() }
    }

    @Test fun replayThumbnailExpiryIncludesOriginalCameraAge() {
        val bitmap = Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888)
        val jpeg = try { ByteArrayOutputStream().use {
            assertTrue(bitmap.compress(Bitmap.CompressFormat.JPEG, 80, it)); it.toByteArray()
        } } finally { bitmap.recycle() }
        HardwareEvidenceRecorder("test").use { recorder ->
            recorder.offer(snapshot(), 1000)
            val first = recorder.freeze().frames.single().copy(thumbnail = jpeg, cameraAgeMs = 700)
            val clip = HardwareEvidenceClip(1, "test", listOf(first,
                first.copy(atMs = 1050, thumbnail = null),
                first.copy(atMs = 1051, thumbnail = null)), emptyList())
            val boundary = clip.snapshotAt(50).image
            assertNotNull("700 ms source age plus 50 ms reaches the accepted boundary", boundary)
            boundary?.recycle()
            assertNull("700 ms source age plus 51 ms must expire", clip.snapshotAt(51).image)
            val unknownAge = clip.copy(frames = clip.frames.map { it.copy(cameraAgeMs = null) })
            val original = unknownAge.snapshotAt(0).image
            assertNotNull(original)
            original?.recycle()
            assertNull("Unknown source age cannot support later reuse", unknownAge.snapshotAt(50).image)
        }
    }

    @Test fun storeRejectsTraversalAndUnsupportedSchema() = withStore { store, dir ->
        for (id in listOf("../clip-a.json", "/tmp/clip-a.json", "clip-a.json/../clip-b.json")) {
            assertTrue(runCatching { store.readBytes(id) }.isFailure)
            assertTrue(runCatching { store.delete(id) }.isFailure)
        }
        HardwareEvidenceRecorder("test").use { recorder ->
            recorder.offer(snapshot(), 1000)
            val id = store.save(recorder.freeze()).id
            val json = JSONObject(store.readBytes(id).toString(Charsets.UTF_8))
            json.put("schema", "hardware-evidence-v999")
            File(dir, id).writeText(json.toString())
            assertTrue(runCatching { store.load(id) }.isFailure)
        }
    }

    @Test fun loaderRejectsStaleAlertAndReversedTimeline() = withStore { store, dir ->
        HardwareEvidenceRecorder("test").use { recorder ->
            recorder.offer(snapshot(), 1000)
            recorder.offer(snapshot(), 1500)
            val id = store.save(recorder.freeze()).id
            val original = store.readBytes(id).toString(Charsets.UTF_8)
            val stale = JSONObject(original)
            stale.getJSONArray("frames").getJSONObject(0).put("tof_usable", false)
            File(dir, id).writeText(stale.toString())
            assertTrue(runCatching { store.load(id) }.isFailure)
            val reversed = JSONObject(original)
            reversed.getJSONArray("frames").getJSONObject(1).put("at_ms", 999)
            File(dir, id).writeText(reversed.toString())
            assertTrue(runCatching { store.load(id) }.isFailure)
        }
    }
}
