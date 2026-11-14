package com.linnan.blindassist.teachable

import android.Manifest
import android.app.KeyguardManager
import android.content.Intent
import android.os.Build
import android.os.SystemClock
import android.view.View
import android.view.ViewGroup
import android.view.inspector.WindowInspector
import android.widget.Button
import android.widget.CheckBox
import android.widget.ListView
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SdkSuppress
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry
import androidx.test.runner.lifecycle.Stage
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Locale

/** Real teaching-page actions with P189 development images and the production W8A16 WCL model. */
@RunWith(AndroidJUnit4::class)
@SdkSuppress(minSdkVersion = Build.VERSION_CODES.Q)
class WclTeachingWarningDeviceTest {
    @get:Rule val camera: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    @Test
    fun warnDeferAndUndoSpecifiedClip() {
        check(!context.getSystemService(KeyguardManager::class.java).isKeyguardLocked) {
            "Unlock the device for the personal teaching test."
        }
        val runId = requireNotNull(InstrumentationRegistry.getArguments().getString("runId"))
        require(runId.matches(Regex("[A-Za-z0-9_-]+")))
        val library = "bui_$runId"
        val store = WclTeachingStore(WclTeachingStore.directory(context.filesDir, library))
        check(!store.root.exists()) { "Use a fresh run ID." }
        val fixture = fixture()
        val model = File(context.filesDir, WclPipeline.MODEL)
        val identity = WclTeachingStore.modelIdentity(model)
        val started = SystemClock.elapsedRealtime()
        val originalIds = fixture.names.take(3).mapIndexed { i, name ->
            val clip = store.newObject(name)
            copyClip(store, clip, fixture.teaching[i])
            clip.name
        }.toSet()
        val queries = WclPipeline(model, context.applicationInfo.nativeLibraryDir).use { pipeline ->
            store.build(pipeline, identity)
            fixture.queries.map { pipeline.embed(WclPipeline.imageOf(it)) }
        }
        val original = reopened(store, identity, originalIds)
        check(original.counts == List(3) { FRAMES })
        check(original.names.toSet() == fixture.names.take(3).toSet())
        val before = snapshot(original, queries)
        val savedFile = File(store.root, "prototypes.json")
        val savedHash = sha256(savedFile.readBytes())
        val intended = fixture.names[1]
        val sourceName = fixture.names[0]
        val target = original.names.indexOf(intended)
        val wrongClip = store.newClip(intended)
        copyClip(store, wrongClip, fixture.teaching[0])
        check(store.pendingClips().map { it.directory.name } == listOf(wrongClip.name))
        val intent = Intent(context, WclTeachActivity::class.java)
            .putExtra(WclTeachActivity.EXTRA_TEST_LIBRARY, library)

        val deferred = ActivityScenario.launch<WclTeachActivity>(intent).use { scenario ->
            try {
                clickSave(scenario)
                val warning = answerWarning(intended, sourceName, "暂缓保存")
                check(sha256(savedFile.readBytes()) == savedHash) { "Deferral rewrote the saved library." }
                val prior = snapshot(reopened(store, identity, originalIds), queries)
                assertEqual(before, prior, "defer")
                check(store.pendingClips().map { it.directory.name } == listOf(wrongClip.name))
                openSaved(scenario)
                StageEvidence(prior, awaitLive(library), warning)
            } finally {
                finishLive(library)
            }
        }

        val accepted = ActivityScenario.launch<WclTeachActivity>(intent).use { scenario ->
            try {
                clickSave(scenario)
                val warning = answerWarning(intended, sourceName, "仍然保存")
                val liveFrames = awaitLive(library)
                check(store.pendingClips().isEmpty())
                val inserted = reopened(store, identity, originalIds + wrongClip.name)
                check(inserted.names == original.names)
                check(inserted.counts == original.counts.mapIndexed { i, n -> if (i == target) n + FRAMES else n })
                val corrupted = snapshot(inserted, queries)
                checkChangedOnlyTarget(before, corrupted, target)
                StageEvidence(corrupted, liveFrames, warning)
            } finally {
                finishLive(library)
            }
        }

        val pending = store.savedClips().single { it.directory == wrongClip }
        val cached = WclEmbeddingCache.read(File(wrongClip, WclEmbeddingCache.FILE_NAME),
            WclEmbeddingCache.identity(identity, pending.frames), FRAMES)
        val match = checkNotNull(WclTeachingWarning.compare(checkNotNull(cached), intended, original))
        check(match.suggested == sourceName && match.retained == FRAMES / 2 && match.gap > WclTeachingWarning.THRESHOLD)
        val restoredLiveFrames = ActivityScenario.launch<WclTeachActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val undo = activity.findViewById<Button>(WclTeachActivity.UNDO_BUTTON_ID)
                check(undo.isEnabled && undo.performClick()) { "Cannot open the clip selector." }
            }
            selectNewest(intended)
            await("Undo confirmation did not appear.") {
                onMain { clickVisibleButton("撤销此片段") }
            }
            await("Undo did not restore only the selected clip's prior library state.") {
                store.savedClips().map { it.directory.name }.toSet() == originalIds &&
                    runCatching { store.load(identity).counts == original.counts }.getOrDefault(false)
            }
            try {
                openSaved(scenario)
                awaitLive(library)
            } finally {
                finishLive(library)
            }
        }
        check(store.pendingClips().isEmpty()) { "The revoked clip returned to the pending list." }
        val restoredQueries = WclPipeline(model, context.applicationInfo.nativeLibraryDir).use { pipeline ->
            fixture.queries.map { pipeline.embed(WclPipeline.imageOf(it)) }
        }
        val restored = snapshot(reopened(store, identity, originalIds), restoredQueries)
        assertEqual(before, restored, "undo")
        val report = JSONObject()
            .put("device", Build.MODEL).put("android_sdk", Build.VERSION.SDK_INT).put("fingerprint", Build.FINGERPRINT)
            .put("model_sha256", identity).put("manifest_sha256", fixture.manifestSha)
            .put("recognition_scope", "WCL W8A16, five fixed P189 Development query JPEGs, raw cosine scores")
            .put("warning_threshold", WclTeachingWarning.THRESHOLD.toDouble())
            .put("warning_gap", match.gap.toDouble()).put("warning_retained_frames", match.retained)
            .put("source_name", sourceName).put("intended_name", intended).put("mistaught_clip_id", wrongClip.name)
            .put("original_clip_ids", JSONArray(originalIds.toList()))
            .put("saved_prototypes_sha256_before_and_deferred", savedHash)
            .put("teaching_images", JSONArray((fixture.teaching.take(3).flatten()).map(::imageReceipt)))
            .put("fixed_queries", JSONArray(fixture.queries.map(::imageReceipt)))
            .put("before", before.json()).put("deferred", deferred.json()).put("accepted", accepted.json())
            .put("restored", restored.json().put("live_frames_observed", restoredLiveFrames))
            .put("elapsed_ms", SystemClock.elapsedRealtime() - started)
        File(store.root, "warning_device_result.json").writeText(report.toString(2))
    }

    private fun clickSave(scenario: ActivityScenario<WclTeachActivity>) {
        scenario.onActivity { activity ->
            activity.findViewById<CheckBox>(WclTeachActivity.COMPLETE_RECOGNITION_ID).isChecked = false
            val save = activity.findViewById<Button>(WclTeachActivity.SAVE_BUTTON_ID)
            check(save.isEnabled && save.performClick()) { "Cannot submit personal teaching." }
        }
    }

    private fun openSaved(scenario: ActivityScenario<WclTeachActivity>) {
        scenario.onActivity { it.findViewById<CheckBox>(WclTeachActivity.COMPLETE_RECOGNITION_ID).isChecked = false }
        await("Cannot open the previously saved library.") {
            onMain { clickVisibleButton("继续识别已保存的物品") }
        }
    }

    private fun answerWarning(intended: String, source: String, answer: String): String {
        var warning = ""
        await("Wrong-teaching warning did not appear for $intended.") {
            onMain {
                val window = visibleWindows().firstOrNull { w ->
                    views(w).filterIsInstance<TextView>().any { it.text.toString() == "请核对教学名称" }
                } ?: return@onMain false
                val text = views(window).filterIsInstance<TextView>().joinToString("\n") { it.text.toString() }
                check(intended in text && source in text) { "Warning did not identify the requested and suggested names." }
                val button = views(window).filterIsInstance<Button>().singleOrNull { it.text.toString() == answer }
                    ?: return@onMain false
                warning = text
                check(button.isEnabled && button.performClick()) { "Cannot choose $answer in the warning." }
                true
            }
        }
        return warning
    }

    private fun selectNewest(intended: String) {
        await("The saved teaching clip selector did not become available.") {
            onMain {
                val list = visibleWindows().flatMap { views(it).filterIsInstance<ListView>().asIterable() }
                    .firstOrNull { it.childCount > 0 } ?: return@onMain false
                check(list.firstVisiblePosition == 0)
                val row = list.getChildAt(0)
                val label = views(row).filterIsInstance<TextView>().joinToString { it.text.toString() }
                check(intended in label && "${FRAMES}帧" in label.replace(" ", "")) { "Unexpected first clip: $label" }
                check(list.performItemClick(row, 0, list.adapter.getItemId(0))) { "Cannot choose the saved clip." }
                true
            }
        }
    }

    private fun clickVisibleButton(text: String): Boolean {
        val button = visibleWindows().flatMap { views(it).filterIsInstance<Button>().asIterable() }
            .firstOrNull { it.text.toString() == text && it.isEnabled && it.isShown } ?: return false
        check(button.performClick()) { "Cannot choose $text." }
        return true
    }

    private fun awaitLive(library: String): Int {
        var frames = 0
        await("The personal recognition page did not process a camera frame.") {
            onMain {
                val live = ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(Stage.RESUMED)
                    .filterIsInstance<WclLiveActivity>()
                    .firstOrNull { it.intent.getStringExtra(WclLiveActivity.EXTRA_LIBRARY) == library }
                frames = live?.analyzedFrameCount() ?: 0
                frames > 0
            }
        }
        return frames
    }

    private fun finishLive(library: String) = instrumentation.runOnMainSync {
        val monitor = ActivityLifecycleMonitorRegistry.getInstance()
        listOf(Stage.RESUMED, Stage.STARTED, Stage.PAUSED, Stage.STOPPED)
            .flatMap(monitor::getActivitiesInStage)
            .filterIsInstance<WclLiveActivity>()
            .filter { it.intent.getStringExtra(WclLiveActivity.EXTRA_LIBRARY) == library }
            .forEach { it.finish() }
    }

    private fun <T : Any> onMain(block: () -> T): T {
        var value: T? = null
        instrumentation.runOnMainSync { value = block() }
        return requireNotNull(value)
    }

    private fun visibleWindows(): List<View> =
        WindowInspector.getGlobalWindowViews().filter { it.isAttachedToWindow && it.isShown }

    private fun views(view: View): Sequence<View> = sequence {
        yield(view)
        if (view is ViewGroup) for (i in 0 until view.childCount) yieldAll(views(view.getChildAt(i)))
    }

    private fun await(message: String, ready: () -> Boolean) {
        val deadline = SystemClock.elapsedRealtime() + 45_000
        while (SystemClock.elapsedRealtime() < deadline) {
            if (ready()) return
            Thread.sleep(50)
        }
        error(message)
    }

    private fun reopened(store: WclTeachingStore, identity: String, ids: Set<String>): WclTeachingStore.Library {
        val reopened = WclTeachingStore(store.root)
        check(reopened.savedClips().map { it.directory.name }.toSet() == ids) { "Wrong saved clip membership." }
        return reopened.load(identity)
    }

    private fun copyClip(store: WclTeachingStore, clip: File, images: List<File>) {
        check(images.size == FRAMES)
        images.forEachIndexed { i, source ->
            val dest = File(clip, "frame_%03d.jpg".format(Locale.ROOT, i))
            source.copyTo(dest)
            check(dest.readBytes().contentEquals(source.readBytes())) { "Teaching image changed while copying: ${source.name}" }
        }
        check(store.finishObject(clip)) { "Could not finish the P189 teaching clip." }
    }

    private fun fixture(): Fixture {
        val root = File(context.filesDir, "m0/P189")
        val manifest = File(root, "meta.json").readBytes()
        val meta = JSONObject(manifest.toString(Charsets.UTF_8))
        val objects = meta.getJSONArray("objects")
        val teach = meta.getJSONArray("teach")
        check(meta.getString("user") == "P189" && objects.length() == 5 && teach.length() == 5)
        val names = List(objects.length()) { objects.getString(it) }
        val images = File(root, "img").canonicalFile
        fun image(name: String): File = File(images, name).canonicalFile.also {
            check(it.parentFile == images && it.isFile) { "Missing development image: $name" }
        }
        val teaching = List(names.size) { i -> teach.getJSONArray(i).let { clip ->
            check(clip.length() == FRAMES)
            List(clip.length()) { image(clip.getString(it)) }
        } }
        val byLabel = arrayOfNulls<File>(names.size)
        val queries = meta.getJSONArray("queries")
        for (i in 0 until queries.length()) {
            val row = queries.getJSONObject(i)
            val label = row.getInt("y")
            if (label in byLabel.indices && byLabel[label] == null) byLabel[label] = image(row.getString("name"))
        }
        val fixed = byLabel.map { requireNotNull(it) { "Missing fixed clutter query." } }
        val teachingSet = teaching.flatten().toSet()
        check(teachingSet.size == 150 && fixed.toSet().size == names.size && fixed.none { it in teachingSet })
        return Fixture(names, teaching, fixed, sha256(manifest))
    }

    private fun snapshot(library: WclTeachingStore.Library, queries: List<FloatArray>): Snapshot =
        Snapshot(library.names, library.counts, bits(library.prototypes), queries.map { bits(WclPipeline.cosine(it, library.prototypes)) })

    private fun assertEqual(expected: Snapshot, actual: Snapshot, phase: String) {
        check(actual.names == expected.names && actual.counts == expected.counts) { "Library membership changed after $phase." }
        check(actual.prototypes.contentEquals(expected.prototypes)) { "Prototype bits did not match after $phase." }
        check(actual.scores.size == expected.scores.size)
        check(actual.scores.indices.all { actual.scores[it].contentEquals(expected.scores[it]) }) {
            "Fixed query score bits did not match after $phase."
        }
    }

    private fun checkChangedOnlyTarget(before: Snapshot, after: Snapshot, target: Int) {
        check(before.names == after.names && target in before.names.indices)
        for (i in before.names.indices) {
            val range = i * WclPipeline.D until (i + 1) * WclPipeline.D
            check(range.any { before.prototypes[it] != after.prototypes[it] } == (i == target)) {
                "Unexpected prototype change after forced acceptance for ${before.names[i]}."
            }
            check(before.scores.indices.any { before.scores[it][i] != after.scores[it][i] } == (i == target)) {
                "Unexpected score change after forced acceptance for ${before.names[i]}."
            }
        }
    }

    private data class Fixture(
        val names: List<String>, val teaching: List<List<File>>, val queries: List<File>, val manifestSha: String,
    )

    private data class StageEvidence(val snapshot: Snapshot, val frames: Int, val warning: String) {
        fun json(): JSONObject = snapshot.json().put("live_frames_observed", frames).put("warning_text", warning)
    }

    private data class Snapshot(val names: List<String>, val counts: List<Int>, val prototypes: IntArray, val scores: List<IntArray>) {
        fun json(): JSONObject = JSONObject().put("names", JSONArray(names)).put("counts", JSONArray(counts))
            .put("prototype_word_count", prototypes.size).put("prototype_words_sha256", sha256(words(prototypes)))
            .put("fixed_query_score_bits", JSONArray(scores.map { JSONArray(it.toList()) }))
    }

    companion object {
        private const val FRAMES = 30

        private fun bits(values: FloatArray): IntArray {
            check(values.all { it.isFinite() }) { "Non-finite WCL output." }
            return IntArray(values.size) { values[it].toRawBits() }
        }

        private fun words(values: IntArray): ByteArray = ByteBuffer.allocate(values.size * Int.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN).apply { values.forEach { putInt(it) } }.array()

        private fun imageReceipt(file: File): JSONObject = JSONObject()
            .put("name", file.name).put("sha256", sha256(file.readBytes()))

        private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it) }
    }
}
