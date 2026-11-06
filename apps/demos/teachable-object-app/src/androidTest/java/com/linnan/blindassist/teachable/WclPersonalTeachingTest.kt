package com.linnan.blindassist.teachable

import android.Manifest
import android.app.KeyguardManager
import android.content.Intent
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import androidx.lifecycle.Lifecycle
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeFalse
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Device integration only; replay teaching and unlabelled camera frames are not a real-object accuracy trial. */
@RunWith(AndroidJUnit4::class)
class WclPersonalTeachingTest {
    @get:Rule val camera: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    /** Engine capability query, not an assertion about audible pronunciation or user understanding. */
    @Test
    fun inspectSpeechLanguages() {
        val ready = CountDownLatch(1)
        var status = TextToSpeech.ERROR
        val speech = TextToSpeech(context) { result -> status = result; ready.countDown() }
        try {
            assertTrue("TTS did not initialise", ready.await(15, TimeUnit.SECONDS))
            assertEquals(TextToSpeech.SUCCESS, status)
            val result = JSONObject().put("engine", speech.defaultEngine)
                .put("zh_CN", speech.isLanguageAvailable(Locale.SIMPLIFIED_CHINESE))
                .put("en_US", speech.isLanguageAvailable(Locale.US))
                .put("scope", "engine availability only; no audible or user evaluation")
            File(context.filesDir, "wcl/tts_languages.json").writeText(result.toString())
        } finally { speech.shutdown() }
    }

    @Test
    fun teachingCameraRebindsAfterReturning() {
        assumeFalse("Teaching UI requires the user to unlock the device", context.getSystemService(KeyguardManager::class.java).isKeyguardLocked)
        ActivityScenario.launch<WclTeachActivity>(Intent(context, WclTeachActivity::class.java)).use { scenario ->
            fun awaitCamera() {
                val deadline = SystemClock.elapsedRealtime() + 15_000
                var ready = false
                while (!ready && SystemClock.elapsedRealtime() < deadline) {
                    scenario.onActivity { ready = it.isCameraReady() }
                    if (!ready) Thread.sleep(100)
                }
                assertTrue("teaching camera did not bind", ready)
            }
            awaitCamera()
            scenario.moveToState(Lifecycle.State.CREATED)
            scenario.onActivity { assertFalse(it.isCameraReady()) }
            scenario.moveToState(Lifecycle.State.RESUMED)
            awaitCamera()
        }
    }

    @Test
    fun persistenceRejectsChangedModelOrTeachingAndKeepsInterruptedCaptureOut() {
        val root = File(context.cacheDir, "wcl_store_${UUID.randomUUID()}")
        try {
            val store = WclTeachingStore(root)
            val draft = store.newObject("未完成")
            File(draft, "frame_000.jpg").writeText("draft")
            assertFalse(store.finishObject(draft))
            assertTrue(store.objects().isEmpty())
            val obj = store.newObject("我的钥匙")
            repeat(5) { File(obj, "frame_$it.jpg").writeText("frame$it") }
            assertTrue(store.finishObject(obj))
            val proto = FloatArray(WclPipeline.D).apply { this[0] = 1f }
            val model = File(root, "model.dlc").apply { writeText("first model") }
            val identity = WclTeachingStore.modelIdentity(model)
            store.save(WclTeachingStore.Library(listOf("我的钥匙"), proto, listOf(5)), identity)
            val reopened = WclTeachingStore(root).load(identity)
            assertEquals(listOf("我的钥匙"), reopened.names)
            assertArrayEquals(proto, reopened.prototypes, 0f)
            model.writeText("other model")
            try { store.load(WclTeachingStore.modelIdentity(model)); fail("accepted a different model") }
            catch (_: IllegalStateException) { }
            File(obj, "frame_5.jpg").writeText("new teaching")
            try { store.load(identity); fail("accepted stale teaching") }
            catch (_: IllegalStateException) { }
        } finally { root.deleteRecursively() }
    }

    @Test
    fun developmentTeachingPersistsAndStartsPersonalLiveChain() {
        val library = "verification_${UUID.randomUUID().toString().replace('-', '_')}"
        val store = WclTeachingStore(WclTeachingStore.directory(context.filesDir, library))
        val source = File(context.filesDir, "m0/P189")
        val meta = JSONObject(File(source, "meta.json").readText())
        val model = File(context.filesDir, "wcl/wcl_cached.dlc")
        val started = SystemClock.elapsedRealtime()
        try {
            val identity = WclTeachingStore.modelIdentity(model)
            repeat(2) { o ->
                val dir = store.newObject(meta.getJSONArray("objects").getString(o))
                val frames = meta.getJSONArray("teach").getJSONArray(o)
                repeat(minOf(8, frames.length())) { i ->
                    File(source, "img/" + frames.getString(i)).copyTo(File(dir, "frame_%03d.jpg".format(i)))
                }
                assertTrue(store.finishObject(dir))
            }
            val built = WclPipeline(model, context.applicationInfo.nativeLibraryDir).use { store.build(it, identity) }
            val reopened = WclTeachingStore(store.root).load(identity)
            assertEquals(built.names, reopened.names)
            assertArrayEquals(built.prototypes, reopened.prototypes, 0f)
            val intent = Intent(context, WclLiveActivity::class.java).putExtra(WclLiveActivity.EXTRA_LIBRARY, library)
            ActivityScenario.launch<WclLiveActivity>(intent).use { scenario ->
                val deadline = SystemClock.elapsedRealtime() + 60_000
                var frameCount = 0
                while (frameCount < 3 && SystemClock.elapsedRealtime() < deadline) {
                    scenario.onActivity { frameCount = it.analyzedFrameCount() }
                    if (frameCount < 3) Thread.sleep(100)
                }
                assertTrue("personal live chain did not analyse camera frames", frameCount >= 3)
                scenario.onActivity { activity ->
                    val result = JSONObject(activity.stopAndDump().readText())
                    assertEquals(library, result.getString("personal_library"))
                    assertEquals(built.names, List(result.getJSONArray("objects").length()) { result.getJSONArray("objects").getString(it) })
                    result.put("validation_kind", "existing Development teaching; unlabelled camera smoke, no accuracy claim")
                        .put("model_sha256", identity).put("total_ms", SystemClock.elapsedRealtime() - started)
                    File(context.filesDir, "wcl/personal_teaching_validation_${System.currentTimeMillis()}.json").writeText(result.toString())
                }
            }
        } finally { store.root.deleteRecursively() }
    }
}
