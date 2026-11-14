package com.linnan.blindassist.teachable

import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.util.AtomicFile
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest

/** Paired Development JPEG replay through the production gallery and warning arithmetic on HTP. */
@RunWith(AndroidJUnit4::class)
class WclTeachingWarningBatchDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    @Test
    fun replayWarningPairs() {
        val args = InstrumentationRegistry.getArguments()
        val root = fixtureRoot(requireNotNull(args.getString("warningBatchFixture")))
        val modelName = args.getString("warningBatchModel") ?: "w8a16"
        val asset = when (modelName) {
            "w8a16" -> WclPipeline.MODEL
            "fp16" -> "wcl/wcl_cached.dlc"
            else -> error("Use warningBatchModel=w8a16 or fp16.")
        }
        val model = File(context.filesDir, asset)
        val manifest = File(root, "manifest.json").readBytes()
        val users = JSONObject(String(manifest, Charsets.UTF_8)).getJSONArray("users")
        check(users.length() > 0) { "Empty replay fixture." }
        val output = File(root, "scores_$modelName.json")
        check(!output.exists()) { "Use a fresh fixture or output file: ${output.name}" }

        val power = context.getSystemService(PowerManager::class.java)
        val userScores = JSONArray()
        val report = JSONObject()
            .put("model", modelName)
            .put("model_asset", asset)
            .put("model_sha256", WclTeachingStore.modelIdentity(model))
            .put("backend", "HTP")
            .put("manifest_sha256", sha256(manifest))
            .put("threshold", WclTeachingWarning.THRESHOLD.toDouble())
            .put("embedding_dimensions", WclPipeline.D)
            .put("preprocessing", WclTeachingStore.PREPROCESSING)
            .put("device", Build.MODEL)
            .put("android_sdk", Build.VERSION.SDK_INT)
            .put("users", userScores)
        val started = SystemClock.elapsedRealtime()
        var encoded = 0
        var pairs = 0
        val seen = HashSet<String>()
        WclPipeline(model, context.applicationInfo.nativeLibraryDir).use { pipeline ->
            for (i in 0 until users.length()) {
                val user = users.getJSONObject(i)
                val name = user.getString("user")
                check(seen.add(name)) { "Repeated fixture user: $name" }
                val result = replayUser(name, user, root, pipeline, power)
                userScores.put(result)
                encoded += result.getInt("encoded_images")
                pairs += result.getJSONArray("cases").length()
                report.put("encoded_images", encoded).put("pairs", pairs).put("complete", false)
                report.put("elapsed_ms", SystemClock.elapsedRealtime() - started)
                val stamp = "$modelName $name: ${result.getInt("encoded_images")} JPEGs, " +
                    "${result.getJSONArray("cases").length()} pairs, ${result.getLong("elapsed_ms")} ms"
                writeReceipt(output, report)
                instrumentation.sendStatus(2, Bundle().apply { putString("stream", "\n$stamp\n") })
            }
        }
        report.put("complete", true)
        report.put("elapsed_ms", SystemClock.elapsedRealtime() - started)
        writeReceipt(output, report)
    }

    private fun fixtureRoot(name: String): File {
        require(name.matches(Regex("bui_warning_pairs_[A-Za-z0-9_-]+"))) { "Invalid batch fixture name." }
        val base = context.filesDir.canonicalFile
        return File(base, name).canonicalFile.also {
            check(it.parentFile == base && it.isDirectory) { "Batch fixture is missing." }
        }
    }

    private fun replayUser(
        name: String, user: JSONObject, root: File, pipeline: WclPipeline, power: PowerManager,
    ): JSONObject {
        val start = SystemClock.elapsedRealtime()
        val thermalBefore = thermalStatus(power)
        val objectArray = user.getJSONArray("objects")
        val objects = List(objectArray.length()) { objectArray.getString(it) }
        check(objects.size >= 2 && objects.distinct().size == objects.size && objects.none { it.isBlank() })
        val images = File(root, "img").canonicalFile
        check(images.parentFile == root && images.isDirectory) { "Expected a fixture-owned img directory." }
        val encoded = HashMap<File, FloatArray>()
        val clipArray = user.getJSONArray("clips")
        val clips = List(clipArray.length()) { i ->
            val clip = clipArray.getJSONObject(i)
            val item = clip.getInt("item")
            check(item in objects.indices) { "Unknown item at $name clip $i." }
            val frames = clip.getJSONArray("frames")
            check(frames.length() > 0) { "Empty clip at $name clip $i." }
            val rows = List(frames.length()) { j ->
                val path = frames.getString(j)
                val file = File(root, path).canonicalFile
                check(file.toPath().startsWith(images.toPath()) && file.extension.equals("jpg", ignoreCase = true) &&
                    file.isFile) { "Missing or invalid JPEG: $path" }
                encoded.getOrPut(file) {
                    pipeline.embed(WclPipeline.imageOf(file)).also { row ->
                        check(row.size == WclPipeline.D && row.all { it.isFinite() }) { "Invalid embedding for $path" }
                    }
                }
            }
            Clip(item, rows)
        }
        val caseArray = user.getJSONArray("cases")
        check(caseArray.length() > 0) { "No paired cases at $name." }
        val cases = JSONArray()
        for (ordinal in 0 until caseArray.length()) {
            val pair = caseArray.getJSONArray(ordinal)
            check(pair.length() == 2)
            val a = pair.getInt(0)
            val b = pair.getInt(1)
            check(a in clips.indices && b in clips.indices && clips[a].item != clips[b].item) {
                "Invalid held pair at $name case $ordinal."
            }
            val gallery = gallery(objects, clips, a, b)
            val intended = objects[clips[b].item]
            cases.put(JSONObject()
                .put("ordinal", ordinal).put("a", a).put("b", b).put("intended", intended)
                .put("wrong", score(clips[a], intended, gallery))
                .put("normal", score(clips[b], intended, gallery)))
        }
        return JSONObject().put("user", name).put("encoded_images", encoded.size).put("cases", cases)
            .put("thermal_status_before", thermalBefore).put("thermal_status_after", thermalStatus(power))
            .put("elapsed_ms", SystemClock.elapsedRealtime() - start)
    }

    private fun gallery(objects: List<String>, clips: List<Clip>, a: Int, b: Int): WclTeachingStore.Library {
        val sums = List(objects.size) { FloatArray(WclPipeline.D) }
        val counts = IntArray(objects.size)
        for (i in clips.indices) {
            if (i == a || i == b) continue
            val clip = clips[i]
            val sum = sums[clip.item]
            for (row in clip.rows) for (d in sum.indices) sum[d] += row[d]
            counts[clip.item] += clip.rows.size
        }
        check(counts.all { it > 0 }) { "A held clip emptied a saved item." }
        return WclTeachingStore.Library(objects, WclPipeline.prototypesOf(sums), counts.toList())
    }

    private fun score(clip: Clip, intended: String, gallery: WclTeachingStore.Library): JSONObject {
        val match = checkNotNull(WclTeachingWarning.compare(clip.rows, intended, gallery))
        check(match.gap.isFinite() && match.retained == (clip.rows.size + 1) / 2)
        return JSONObject().put("gap", match.gap.toDouble()).put("retained", match.retained)
            .put("suggested", match.suggested).put("warn", match.gap > WclTeachingWarning.THRESHOLD)
    }

    private fun thermalStatus(power: PowerManager): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) power.currentThermalStatus else -1

    private fun writeReceipt(file: File, report: JSONObject) {
        val atomic = AtomicFile(file)
        val stream = atomic.startWrite()
        try {
            stream.write(report.toString().toByteArray(Charsets.UTF_8))
            atomic.finishWrite(stream)
        } catch (error: Exception) {
            atomic.failWrite(stream)
            throw error
        }
    }

    private data class Clip(val item: Int, val rows: List<FloatArray>)

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it) }
}
