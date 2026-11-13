package com.linnan.blindassist.teachable

import android.Manifest
import android.content.Intent
import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID

/** Exercises both camera and TTS paths with a disposable library made from existing development images. */
@RunWith(AndroidJUnit4::class)
class WclPresentationDeviceTest {
    @get:Rule val camera: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun bothArmsStartFreshAndRecordSpeech() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val id = "trial_smoke_${UUID.randomUUID().toString().replace('-', '_')}"
        val store = WclTeachingStore(WclTeachingStore.directory(context.filesDir, id))
        val source = File(context.filesDir, "m0/P189")
        val meta = JSONObject(File(source, "meta.json").readText())
        val model = File(context.filesDir, WclPipeline.MODEL)
        val root = File(context.filesDir, "wcl/presentation_trials")
        val before = root.list().orEmpty().toSet()
        try {
            for ((index, name) in listOf("red object", "blue object").withIndex()) {
                val dir = store.newObject(name)
                val frames = meta.getJSONArray("teach").getJSONArray(index)
                repeat(5) { i ->
                    File(source, "img/${frames.getString(i)}").copyTo(File(dir, "frame_$i.jpg"))
                }
                assertTrue(store.finishObject(dir))
            }
            WclPipeline(model, context.applicationInfo.nativeLibraryDir).use { pipeline ->
                store.build(pipeline, WclTeachingStore.modelIdentity(model))
            }
            val intent = Intent(context, WclLiveActivity::class.java)
                .putExtra(WclLiveActivity.EXTRA_LIBRARY, id)
                .putExtra(WclLiveActivity.EXTRA_PRESENTATION_TRIALS, true)
            ActivityScenario.launch<WclLiveActivity>(intent).use { scenario ->
                var previousLibrary: String? = null
                var previousModel: String? = null
                for ((index, arm) in listOf("natural", "guided").withIndex()) {
                    await("trial controls unavailable") {
                        var ready = false
                        scenario.onActivity { ready = it.trialControlsReady() }
                        ready
                    }
                    scenario.onActivity { it.startTrial(guided = arm == "guided") }
                    val output = "${index + 1}.json"
                    var rowFile: File? = null
                    await("trial $arm did not write its receipt") {
                        rowFile = root.listFiles().orEmpty().filter { it.name !in before }
                            .map { File(it, output) }.singleOrNull { it.isFile }
                        rowFile != null
                    }
                    val row = JSONObject(requireNotNull(rowFile).readText())
                    assertEquals(arm, row.getString("arm"))
                    assertEquals("deadline", row.getString("end_reason"))
                    assertTrue(row.getBoolean("cue_started"))
                    val started = row.getLong("started_ns")
                    val deadline = row.getLong("deadline_ns")
                    assertEquals(6_000_000_000L, deadline - started)
                    val frames = row.getJSONArray("frames")
                    assertTrue("camera did not score at least three frames", frames.length() >= 3)
                    for (i in 0 until frames.length()) {
                        val frame = frames.getJSONObject(i)
                        assertTrue(frame.getLong("capture_ns") in started until deadline)
                        assertTrue(frame.getLong("decision_ns") in started until deadline)
                    }
                    val speech = row.getJSONArray("speech")
                    assertTrue("no TTS utterance finished in $arm", (0 until speech.length()).any {
                        val event = speech.getJSONObject(it)
                        event.optLong("tts_start_ns", 0L) in started until deadline &&
                            event.optLong("tts_done_ns", 0L) in started until deadline
                    })
                    val library = row.getString("library_sha256")
                    val modelHash = row.getString("model_sha256")
                    previousLibrary?.let { assertEquals(it, library) }
                    previousModel?.let { assertEquals(it, modelHash) }
                    previousLibrary = library
                    previousModel = modelHash
                    println("WCL trial evidence ${requireNotNull(rowFile).relativeTo(context.filesDir)}")
                }
            }
        } finally {
            store.root.deleteRecursively()
        }
    }

    private fun await(message: String, ready: () -> Boolean) {
        val deadline = SystemClock.elapsedRealtime() + 30_000
        while (SystemClock.elapsedRealtime() < deadline) {
            if (ready()) return
            Thread.sleep(100)
        }
        error(message)
    }
}
