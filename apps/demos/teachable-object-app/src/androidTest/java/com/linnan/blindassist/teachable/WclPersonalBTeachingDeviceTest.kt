package com.linnan.blindassist.teachable

import android.Manifest
import android.app.KeyguardManager
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import android.widget.Button
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
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
import java.security.MessageDigest
import java.util.Locale

/** P189 teaching replay with a live-camera timing endpoint and optional fixed-JPEG score checks. */
@RunWith(AndroidJUnit4::class)
class WclPersonalBTeachingDeviceTest {
    @get:Rule val camera: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    @Test
    fun prepareAndAppend() {
        val run = readRun()
        check(!run.store.root.exists() && !run.output.exists()) { "Use a fresh run ID" }
        check(run.output.mkdirs())
        addTeaching(run, 0, 3)
        saveAndRecognize(run, 3)
        addTeaching(run, 3, 4)
        saveAndRecognize(run, 4)
    }

    @Test
    fun appendAfterRestart() {
        val run = readRun()
        val prior = JSONObject(File(run.output, "4.json").readText())
        val initial = JSONObject(File(run.output, "3.json").readText())
        check(prior.getInt("pid") == initial.getInt("pid") &&
            prior.getLong("process_start_ticks") == initial.getLong("process_start_ticks")) {
            "The first two stages ran in different processes"
        }
        check(prior.getLong("process_start_ticks") != processStartTicks() || prior.getInt("pid") != Process.myPid()) {
            "Stop the application process after completing stage 4 before running stage 5"
        }
        addTeaching(run, 4, 5)
        saveAndRecognize(run, 5)
    }

    private data class Run(
        val arm: String, val library: String, val store: WclTeachingStore, val output: File,
        val names: List<String>, val sources: List<List<File>>, val fixedQueries: List<File>, val manifestSha: String,
    )

    private fun readRun(): Run {
        check(!context.getSystemService(KeyguardManager::class.java).isKeyguardLocked) { "Unlock the device for the camera test" }
        val args = InstrumentationRegistry.getArguments()
        val runId = requireNotNull(args.getString("runId"))
        require(runId.matches(Regex("[A-Za-z0-9_-]+")))
        val arm = requireNotNull(args.getString("arm"))
        require(arm in setOf("full", "incremental", "wcl_full", "wcl_incremental"))
        val library = "bui_${runId}_$arm"
        val source = File(context.filesDir, "m0/P189")
        val manifest = File(source, "meta.json").readBytes()
        val meta = JSONObject(manifest.toString(Charsets.UTF_8))
        val objects = meta.getJSONArray("objects")
        val teach = meta.getJSONArray("teach")
        check(meta.getString("user") == "P189" && objects.length() == 5 && teach.length() == 5)
        val names = List(5) { objects.getString(it) }
        val imageRoot = File(source, "img")
        val sources = List(5) { index ->
            val frames = teach.getJSONArray(index)
            check(frames.length() == 30)
            List(frames.length()) { File(imageRoot, frames.getString(it)) }
        }
        check(sources.flatten().toSet().size == 150 && sources.flatten().all(File::isFile))
        val fixedQueries = if (arm.startsWith("wcl_")) {
            // Freeze the first manifest query per class, including at stages before its class is taught.
            val queries = meta.getJSONArray("queries")
            val first = arrayOfNulls<File>(names.size)
            val canonicalImages = imageRoot.canonicalFile
            for (index in 0 until queries.length()) {
                val query = queries.getJSONObject(index)
                val y = query.getInt("y")
                if (y in first.indices && first[y] == null) {
                    first[y] = File(canonicalImages, query.getString("name")).canonicalFile
                }
            }
            val selected = first.map { requireNotNull(it) { "Missing fixed P189 query" } }
            val teaching = sources.flatten().mapTo(HashSet()) { it.canonicalFile }
            check(selected.toSet().size == names.size && selected.all {
                it.isFile && it.parentFile == canonicalImages && it !in teaching
            }) { "Fixed P189 queries must be distinct images outside the teaching list" }
            selected
        } else emptyList()
        return Run(arm, library, WclTeachingStore(WclTeachingStore.directory(context.filesDir, library)),
            File(context.filesDir, "b_ui_results/$runId/$arm"), names, sources, fixedQueries, sha256(manifest))
    }

    private fun addTeaching(run: Run, previous: Int, objects: Int) {
        check(run.store.objects().map { it.name } == run.names.take(previous)) { "Unexpected existing teaching objects" }
        for (objectIndex in previous until objects) {
            val directory = run.store.newObject(run.names[objectIndex])
            run.sources[objectIndex].forEachIndexed { index, image ->
                image.copyTo(File(directory, "frame_%03d.jpg".format(Locale.ROOT, index)))
            }
            check(run.store.finishObject(directory)) { "Could not complete the replay teaching object" }
        }
        check(run.store.objects().map { it.name } == run.names.take(objects)) { "Teaching order differs from the P189 manifest" }
    }

    private fun saveAndRecognize(run: Run, objects: Int) {
        val teachingHashes = verifyTeaching(run, objects)
        val fixedQueries = run.fixedQueries
        val fixedQueryHashes = fixedQueries.map { sha256(it.readBytes()) }
        val receipt = File(run.store.root, "b_ui_result.json")
        val stage = File(run.output, "$objects.json")
        check(!stage.exists()) { "Result already exists: $stage" }
        val intent = Intent(context, WclTeachActivity::class.java)
            .putExtra(WclTeachActivity.EXTRA_TEST_LIBRARY, run.library)
            .putExtra(WclTeachActivity.EXTRA_TEST_FORCE_FULL_B, run.arm == "full")
            .putExtra(WclTeachActivity.EXTRA_TEST_FORCE_FULL_WCL, run.arm != "wcl_incremental")
        if (fixedQueries.isNotEmpty()) {
            intent.putStringArrayListExtra(WclTeachActivity.EXTRA_TEST_FIXED_QUERY, ArrayList(fixedQueries.map { it.path }))
        }
        ActivityScenario.launch<WclTeachActivity>(intent).use { scenario ->
            await(15_000, "Teaching camera did not bind") {
                var ready = false
                scenario.onActivity { ready = it.isCameraReady() }
                ready
            }
            val thermalBefore = thermalSnapshot()
            var clickLowerBound = 0L
            scenario.onActivity { activity ->
                val save = activity.findViewById<Button>(WclTeachActivity.SAVE_BUTTON_ID)
                check(save.isEnabled) { "Personal teaching save button is disabled" }
                clickLowerBound = SystemClock.elapsedRealtimeNanos()
                check(save.performClick()) { "Personal teaching save button did not invoke its listener" }
            }
            try {
                val result = awaitResult(receipt, clickLowerBound)
                result.put("library", run.library)
                    .put("arm", run.arm)
                    .put("test_click_lower_bound_ns", clickLowerBound)
                    .put("process_start_ticks", processStartTicks())
                    .put("p189_manifest_sha256", run.manifestSha)
                    .put("teaching_objects", JSONArray(run.names.take(objects)))
                    .put("teaching_jpeg_sha256", teachingHashes)
                    .put("device", Build.MODEL)
                    .put("android_version", Build.VERSION.RELEASE)
                    .put("android_sdk", Build.VERSION.SDK_INT)
                    .put("fingerprint", Build.FINGERPRINT)
                    .put("thermal_before", thermalBefore)
                    .put("thermal_after", thermalSnapshot())
                if (fixedQueries.isNotEmpty()) result.put("fixed_query_images", JSONArray(fixedQueries.map { it.name }))
                stage.writeText(result.toString(2))
                check(!result.has("error")) { "Personal teaching failed: ${result.optString("error")}" }
                check(result.getInt("objects") == objects && result.getInt("pid") == Process.myPid())
                val saveNs = result.getLong("save_ns")
                val readyNs = result.getLong("ready_ns")
                val firstNs = result.getLong("first_result_ns")
                val displayedNs = result.getLong("displayed_ns")
                check(saveNs >= clickLowerBound && readyNs >= saveNs && firstNs >= readyNs && displayedNs >= firstNs)
                val mode = if (run.arm == "full") "full_cutoff" else if (objects == 3) "full_seed" else "restored_append"
                check(result.getString("mode") == mode) { "Expected $mode, got ${result.getString("mode")}" }
                check(result.getInt("retained_count") == 1_152 * objects)
                if (mode == "restored_append") {
                    check(result.getLong("reused_witness_rows") > 0 && result.getDouble("restore_ms") > 0)
                    check(result.getLong("evicted_witness_rows") == if (objects == 4) 0L else 912L)
                }
                for (key in listOf("retained_sha256", "parts_sha256", "cls_sha256", "padded_parts_sha256", "valid_sha256")) {
                    check(result.getString(key).matches(Regex("[a-fA-F0-9]{64}"))) { "Invalid $key" }
                }
                if (fixedQueries.isNotEmpty()) verifyWcl(run, objects, fixedQueryHashes, result)
            } finally {
                finishPersonalLive(run.library)
            }
        }
    }

    private fun verifyWcl(run: Run, objects: Int, fixedQueryHashes: List<String>, result: JSONObject) {
        val expectedReused = if (run.arm == "wcl_incremental" && objects > 3) (objects - 1) * 30 else 0
        val encoded = result.getInt("wcl_encoded_frames")
        val reused = result.getInt("wcl_reused_frames")
        check(reused == expectedReused && encoded == objects * 30 - expectedReused) {
            "Unexpected WCL work at $objects objects (${run.arm}): encoded $encoded, reused $reused"
        }
        check(result.getDouble("wcl_build_ms") > 0)
        verifyFloatWords(result, "wcl_prototype_bits", objects * WclPipeline.D)
        val scores = result.getJSONArray("fixed_queries")
        check(scores.length() == fixedQueryHashes.size)
        for ((index, hash) in fixedQueryHashes.withIndex()) {
            val row = scores.getJSONObject(index)
            check(row.getString("sha256") == hash) { "Fixed query identity differs at index $index" }
            for (key in listOf("wcl_score_bits", "b_score_bits", "fused_score_bits")) verifyFloatWords(row, key, objects)
        }
    }

    private fun verifyFloatWords(result: JSONObject, key: String, size: Int) {
        val bits = result.getJSONArray(key)
        check(bits.length() == size) { "Unexpected $key size" }
        for (index in 0 until size) {
            val word = bits.get(index)
            check(word is Int && Float.fromBits(word).isFinite()) { "Invalid FP32 word in $key at index $index" }
        }
    }

    private fun verifyTeaching(run: Run, objects: Int): JSONObject {
        val identities = JSONObject()
        val taught = run.store.objects()
        check(taught.map { it.name } == run.names.take(objects))
        for (index in taught.indices) {
            val frames = taught[index].frames
            check(frames.size == run.sources[index].size)
            frames.zip(run.sources[index]).forEach { (copy, source) ->
                val sha = sha256(source.readBytes())
                check(sha256(copy.readBytes()) == sha) { "Copied teaching JPEG differs: ${source.name}" }
                identities.put(source.name, sha)
            }
        }
        return identities
    }

    private fun awaitResult(receipt: File, clickLowerBound: Long): JSONObject {
        var result: JSONObject? = null
        await(120_000, "The live camera did not produce the first recognition result") {
            result = runCatching { JSONObject(receipt.readText()) }.getOrNull()?.takeIf {
                it.optLong("save_ns") >= clickLowerBound && (it.has("error") || it.optLong("first_result_ns") > 0L)
            }
            result != null
        }
        return requireNotNull(result)
    }

    private fun finishPersonalLive(library: String) = instrumentation.runOnMainSync {
        val monitor = ActivityLifecycleMonitorRegistry.getInstance()
        listOf(Stage.RESUMED, Stage.STARTED, Stage.PAUSED, Stage.STOPPED).flatMap { monitor.getActivitiesInStage(it) }
            .filterIsInstance<WclLiveActivity>()
            .filter { it.intent.getStringExtra(WclLiveActivity.EXTRA_LIBRARY) == library }
            .forEach { it.finish() }
    }

    private fun await(timeoutMs: Long, message: String, ready: () -> Boolean) {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (SystemClock.elapsedRealtime() < deadline) {
            if (ready()) return
            Thread.sleep(50)
        }
        error(message)
    }

    private fun processStartTicks(): Long =
        File("/proc/self/stat").readText().substringAfterLast(") ").split(' ')[19].toLong()

    private fun thermalSnapshot(): JSONObject {
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val temperature = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        return JSONObject().apply {
            if (temperature != null && temperature != Int.MIN_VALUE) put("battery_c", temperature / 10.0)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                context.getSystemService(PowerManager::class.java)?.let { put("thermal_status", it.currentThermalStatus) }
            }
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }
}
