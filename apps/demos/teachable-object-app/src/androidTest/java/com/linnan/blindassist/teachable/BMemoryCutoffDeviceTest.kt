package com.linnan.blindassist.teachable

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** P189 teaching-only benchmark for the production B memory path. */
@RunWith(AndroidJUnit4::class)
class BMemoryCutoffDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun pairedTeaching() {
        val input = loadP189()
        val teachFiles = input.teachFiles
        val output = File(context.filesDir, "b_memory_cutoff")
        check(output.mkdir()) { "Expected a fresh test output directory: $output" }
        val runs = JSONArray()
        val result = input.summary.put("pairs", runs)
        val json = File(output, "p189.json")
        withWarmPipeline(input) { b ->
            var firstFeatureHash: String? = null
            repeat(3) { pair ->
                val cutoffFirst = pair % 2 == 1
                fun measure(cutoff: Boolean) = measured(if (cutoff) "cutoff" else "original") {
                    b.buildTeachingMemory(teachFiles, cutoff = cutoff, captureEvidence = true)
                }
                val first = measure(cutoffFirst)
                val second = measure(!cutoffFirst)
                val original = if (cutoffFirst) second else first
                val cutoff = if (cutoffFirst) first else second

                val originalRetained = requireNotNull(original.value.retained)
                val cutoffRetained = requireNotNull(cutoff.value.retained)
                val sameRetained = originalRetained.contentEquals(cutoffRetained)
                val sameParts = rawEqual(original.value.built.parts, cutoff.value.built.parts)
                val sameMemory = rawEqual(original.value.memory.parts, cutoff.value.memory.parts) &&
                    rawEqual(original.value.memory.protos, cutoff.value.memory.protos) &&
                    rawEqual(original.value.memory.valid, cutoff.value.memory.valid)
                val originalHash = requireNotNull(original.value.featureSha256)
                val cutoffHash = requireNotNull(cutoff.value.featureSha256)
                if (pair == 0) firstFeatureHash = originalHash
                val sameFeatures = originalHash == cutoffHash && originalHash == firstFeatureHash
                val prunedRows = requireNotNull(cutoff.value.diagnostics)[2]
                val equal = sameRetained && sameParts && sameMemory && sameFeatures

                runs.put(JSONObject()
                    .put("pair", pair + 1)
                    .put("execution_order", JSONArray(listOf(first.name, second.name)))
                    .put("original", original.summary)
                    .put("cutoff", cutoff.summary)
                    .put("retained_bit_equal", sameRetained)
                    .put("parts_bit_equal", sameParts)
                    .put("memory_bit_equal", sameMemory)
                    .put("feature_sha256_equal_across_all_runs", sameFeatures))
                json.writeText(result.toString(2))
                if (pair == 0 || !equal) {
                    val folder = File(output, "pair_${pair + 1}").apply { check(mkdir()) }
                    writeRaw(folder, original, originalRetained)
                    writeRaw(folder, cutoff, cutoffRetained)
                }
                check(equal) { "Pair ${pair + 1} disagrees. See ${json.path}" }
                check(prunedRows > 0) { "Pair ${pair + 1} did not execute pruning. See ${json.path}" }
            }
        }
    }

    @Test
    fun pairedAppend() {
        val runId = InstrumentationRegistry.getArguments().getString("runId") ?: "p189"
        require(runId.matches(Regex("[a-zA-Z0-9_-]+")))
        val root = File(context.filesDir, "b_memory_incremental").apply { mkdir() }
        val output = File(root, runId)
        check(output.mkdir()) { "Expected a fresh test output directory: $output" }
        val json = File(output, "p189.json")
        val input = loadP189()
        val teachFiles = input.teachFiles
        val runs = JSONArray()
        val result = input.summary
            .put("run_id", runId)
            .put("initial_objects", 4)
            .put("appended_object_index", 4)
            .put("cache_scope", "process_resident")
            .put("pairs", runs)
            .put("completed", false)
        json.writeText(result.toString(2))
        withWarmPipeline(input) { b ->
            val seedStart = thermalSnapshot()
            val prepared = b.prepareTeachingAppend(teachFiles.take(4), captureEvidence = true)
            val seed = arm("initial", prepared.initial, seedStart)
            result.put("initial", seed.summary.put("encoded_jpegs", 120).put("cache_bytes", prepared.cacheBytes))
            json.writeText(result.toString(2))
            check(seed.value.memory.objects == 4 && prepared.cacheBytes > 0)
            val seedRetained = requireNotNull(seed.value.retained).filter { it >= 0 }.toSet()
            val oldPatchCounts = teachFiles.take(4).map { it.size * TeachablePipeline.GRID * TeachablePipeline.GRID }

            var firstFeatureHash: String? = null
            repeat(3) { pair ->
                val incrementalFirst = pair % 2 == 1
                fun measure(incremental: Boolean): Arm {
                    val mode = if (incremental) "incremental" else "full_cutoff"
                    val arm = measured(mode) {
                        if (incremental) b.appendTeachingMemory(prepared, teachFiles.last(), captureEvidence = true)
                        else b.buildTeachingMemory(teachFiles, captureEvidence = true)
                    }
                    arm.summary.put("encoded_jpegs", if (incremental) 30 else 150)
                    return arm
                }
                val pairResult = JSONObject().put("pair", pair + 1)
                    .put("execution_order", JSONArray(if (incrementalFirst) listOf("incremental", "full_cutoff")
                        else listOf("full_cutoff", "incremental")))
                runs.put(pairResult)
                val first = measure(incrementalFirst)
                pairResult.put(first.name, first.summary)
                json.writeText(result.toString(2))
                val second = measure(!incrementalFirst)
                pairResult.put(second.name, second.summary)
                val full = if (incrementalFirst) second else first
                val incremental = if (incrementalFirst) first else second

                val fullRetained = requireNotNull(full.value.retained)
                val incrementalRetained = requireNotNull(incremental.value.retained)
                val sameRetained = fullRetained.contentEquals(incrementalRetained)
                val sameParts = rawEqual(full.value.built.parts, incremental.value.built.parts)
                val sameCls = rawEqual(full.value.memory.protos, incremental.value.memory.protos)
                val sameMemory = full.value.memory.objects == 5 && incremental.value.memory.objects == 5 &&
                    rawEqual(full.value.memory.parts, incremental.value.memory.parts) && sameCls &&
                    rawEqual(full.value.memory.valid, incremental.value.memory.valid)
                val fullHash = requireNotNull(full.value.featureSha256)
                val appendHash = requireNotNull(incremental.value.featureSha256)
                if (pair == 0) firstFeatureHash = fullHash
                val sameFeatures = fullHash == appendHash && fullHash == firstFeatureHash
                val equal = sameRetained && sameParts && sameMemory && sameFeatures
                pairResult.put("saved_ms", full.value.totalMs - incremental.value.totalMs)
                    .put("retained_bit_equal", sameRetained)
                    .put("parts_bit_equal", sameParts)
                    .put("cls_bit_equal", sameCls)
                    .put("memory_bit_equal", sameMemory)
                    .put("old_retained_changes", oldRetainedChanges(seedRetained, incrementalRetained, oldPatchCounts))
                    .put("feature_sha256_equal_across_all_runs", sameFeatures)
                json.writeText(result.toString(2))
                if (pair == 0 || !equal) {
                    val folder = File(output, "pair_${pair + 1}").apply { check(mkdir()) }
                    writeRaw(folder, full, fullRetained)
                    writeRaw(folder, incremental, incrementalRetained)
                }
                check(equal) { "Pair ${pair + 1} disagrees. See ${json.path}" }
                val fullDiagnostics = requireNotNull(full.value.diagnostics)
                val appendDiagnostics = requireNotNull(incremental.value.diagnostics)
                check(fullDiagnostics[2] > 0 && appendDiagnostics[1] < fullDiagnostics[1]) {
                    "Pair ${pair + 1} did not exercise both cutoff and own-score reuse. See ${json.path}"
                }
                check(appendDiagnostics[3] > 0 && appendDiagnostics[4] > 0 && appendDiagnostics[5] > 0) {
                    "Pair ${pair + 1} did not exercise retained and evicted witnesses. See ${json.path}"
                }
            }
            result.put("completed", true)
            json.writeText(result.toString(2))
        }
    }

    private class Input(val teachFiles: List<List<File>>, val summary: JSONObject)

    private fun oldRetainedChanges(before: Set<Int>, after: IntArray, counts: List<Int>): JSONArray {
        val end = counts.sum()
        val newOldRows = after.filter { it in 0 until end }.toSet()
        val added = newOldRows - before
        val removed = before - newOldRows
        var offset = 0
        return JSONArray(counts.mapIndexed { objectIndex, count ->
            val rows = offset until offset + count
            offset += count
            JSONObject()
                .put("object_index", objectIndex)
                .put("added", added.count { it in rows })
                .put("removed", removed.count { it in rows })
        })
    }

    private fun loadP189(): Input {
        val input = File(context.filesDir, "m0/P189")
        val manifest = File(input, "meta.json").readBytes()
        val meta = JSONObject(manifest.toString(Charsets.UTF_8))
        val teach = meta.getJSONArray("teach")
        require(meta.getString("user") == "P189" && teach.length() == 5)
        val teachFiles = List(teach.length()) { objectIndex ->
            val images = teach.getJSONArray(objectIndex)
            require(images.length() == 30)
            List(images.length()) { File(input, "img/${images.getString(it)}") }
        }
        require(teachFiles.flatten().toSet().size == 150 && teachFiles.all { files -> files.all(File::isFile) })
        return Input(teachFiles, JSONObject()
            .put("user", "P189")
            .put("device", Build.MODEL)
            .put("fingerprint", Build.FINGERPRINT)
            .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
            .put("manifest_sha256", sha256(manifest))
            .put("teaching_jpeg_sha256", JSONObject(teachFiles.flatten().associate { it.name to sha256(it.readBytes()) }))
            .put("objects", meta.getJSONArray("objects"))
            .put("frames_per_object", JSONArray(teachFiles.map { it.size }))
            .put("patches_per_frame", TeachablePipeline.GRID * TeachablePipeline.GRID)
            .put("dim", B_DIM))
    }

    private fun withWarmPipeline(input: Input, run: (TeachablePipeline) -> Unit) {
        val models = File(context.filesDir, "bwcl")
        val initStart = System.nanoTime()
        TeachablePipeline(
            File(models, "backbone_b_cached.dlc"), File(models, "bhead_fp32.dlc"),
            context.applicationInfo.nativeLibraryDir, dim = B_DIM,
        ).use { b ->
            input.summary.put("init_ms", (System.nanoTime() - initStart) / 1e6)
            val warm = WclPipeline.imageOf(input.teachFiles.first().first())
            repeat(3) { b.tokens(warm) }
            run(b)
        }
    }

    private class Arm(val name: String, val value: TeachablePipeline.TeachingMemory, val summary: JSONObject)

    private fun measured(name: String, run: () -> TeachablePipeline.TeachingMemory): Arm {
        val start = thermalSnapshot()
        return arm(name, run(), start)
    }

    private fun arm(name: String, value: TeachablePipeline.TeachingMemory, start: JSONObject): Arm {
        val stop = thermalSnapshot()
        val diagnostics = requireNotNull(value.diagnostics)
        val retained = requireNotNull(value.retained).count { it >= 0 }
        val summary = JSONObject()
            .put("total_ms", value.totalMs)
            .put("decode_and_encode_ms", value.encodingMs)
            .put("native_ms", value.nativeMs)
            .put("negative_ms", value.built.otherMs)
            .put("own_ms", value.built.ownMs)
            .put("selection_and_kmeans_ms", value.built.kmeansMs)
            .put("negative_dots", diagnostics[0])
            .put("positive_dots", diagnostics[1])
            .put("pruned_rows", diagnostics[2])
            .put("retained", retained)
            .put("feature_sha256", value.featureSha256)
            .put("start", start)
            .put("stop", stop)
        if (diagnostics.size >= 6) {
            summary.put("reused_witness_rows", diagnostics[3])
                .put("evicted_or_missing_witness_rows", diagnostics[4])
                .put("cached_negative_dots_skipped", diagnostics[5])
        }
        return Arm(name, value, summary)
    }

    private fun thermalSnapshot(): JSONObject {
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val temperature = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        return JSONObject().apply {
            if (temperature != null && temperature != Int.MIN_VALUE) put("battery_c", temperature / 10.0)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) put("thermal_status", power.currentThermalStatus)
        }
    }

    private fun writeRaw(folder: File, arm: Arm, retained: IntArray) {
        val indexBytes = ByteBuffer.allocate(retained.count { it >= 0 } * Int.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        for (index in retained) if (index >= 0) indexBytes.putInt(index)
        File(folder, "${arm.name}_retained.i32").writeBytes(indexBytes.array())
        writeFloats(File(folder, "${arm.name}_parts.f32"), arm.value.built.parts)
        writeFloats(File(folder, "${arm.name}_cls.f32"), arm.value.memory.protos)
    }

    private fun writeFloats(file: File, values: FloatArray) {
        val bytes = ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        for (value in values) bytes.putInt(value.toRawBits())
        file.writeBytes(bytes.array())
    }

    private fun rawEqual(a: FloatArray, b: FloatArray): Boolean =
        a.size == b.size && a.indices.all { a[it].toRawBits() == b[it].toRawBits() }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }

    private companion object {
        const val B_DIM = 768
    }
}
