package com.linnan.blindassist.teachable

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
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

    @Test
    fun pairedAppendChain() {
        val runId = InstrumentationRegistry.getArguments().getString("runId") ?: "chain_p189"
        require(runId.matches(Regex("[a-zA-Z0-9_-]+")))
        val root = File(context.filesDir, "b_memory_incremental").apply { mkdir() }
        val output = File(root, runId)
        check(output.mkdir()) { "Expected a fresh test output directory: $output" }
        val json = File(output, "p189.json")
        val input = loadP189()
        val teachFiles = input.teachFiles
        val patchCounts = teachFiles.map { it.size * TeachablePipeline.GRID * TeachablePipeline.GRID }
        val runs = JSONArray()
        val result = input.summary
            .put("run_id", runId)
            .put("initial_objects", 3)
            .put("cache_scope", "process_resident")
            .put("incremental_timing_includes_successor_cache", true)
            .put("pairs", runs)
            .put("completed", false)
        json.writeText(result.toString(2))
        withWarmPipeline(input) { b ->
            val seedStart = thermalSnapshot()
            val seed = b.prepareTeachingAppend(teachFiles.take(3), captureEvidence = true)
            check(seed.initial.memory.objects == 3 && seed.cacheBytes > 0)
            result.put("initial", arm("initial", seed.initial, seedStart).summary
                .put("encoded_jpegs", teachFiles.take(3).sumOf { it.size })
                .put("cache_bytes", seed.cacheBytes))
            json.writeText(result.toString(2))

            val featureHashes = mutableMapOf<Int, String>()
            repeat(3) { pair ->
                var current = seed
                val steps = JSONArray()
                runs.put(JSONObject().put("pair", pair + 1).put("steps", steps))
                val incrementalFirst = pair % 2 == 1
                for (objects in 4..5) {
                    val fromObjects = objects - 1
                    check(current.initial.memory.objects == fromObjects)
                    val previousRetained = requireNotNull(current.initial.retained).filter { it >= 0 }.toSet()
                    val incomingHash = requireNotNull(current.initial.featureSha256)
                    val incomingBytes = current.cacheBytes
                    val newFiles = teachFiles[fromObjects]
                    val allFiles = teachFiles.take(objects)
                    fun measure(incremental: Boolean): Arm {
                        if (!incremental) {
                            val full = measured("full_cutoff") { b.buildTeachingMemory(allFiles, captureEvidence = true) }
                            full.summary.put("encoded_jpegs", allFiles.sumOf { it.size })
                            return full
                        }
                        val start = thermalSnapshot()
                        current = b.appendTeachingAndCache(current, newFiles, captureEvidence = true)
                        val append = arm("incremental", current.initial, start)
                        append.summary.put("encoded_jpegs", newFiles.size)
                            .put("cache_source", if (fromObjects == 3) "initial" else "3_to_4_incremental")
                            .put("input_feature_sha256", incomingHash)
                            .put("input_cache_bytes", incomingBytes)
                            .put("cache_bytes", current.cacheBytes)
                        return append
                    }
                    val step = JSONObject().put("from_objects", fromObjects).put("to_objects", objects)
                        .put("execution_order", JSONArray(if (incrementalFirst) listOf("incremental", "full_cutoff")
                            else listOf("full_cutoff", "incremental")))
                    steps.put(step)
                    val first = measure(incrementalFirst)
                    step.put(first.name, first.summary)
                    json.writeText(result.toString(2))
                    val second = measure(!incrementalFirst)
                    step.put(second.name, second.summary)
                    val full = if (incrementalFirst) second else first
                    val append = if (incrementalFirst) first else second

                    val fullRetained = requireNotNull(full.value.retained)
                    val appendRetained = requireNotNull(append.value.retained)
                    val sameRetained = fullRetained.contentEquals(appendRetained)
                    val sameParts = rawEqual(full.value.built.parts, append.value.built.parts)
                    val sameCls = rawEqual(full.value.memory.protos, append.value.memory.protos)
                    val sameMemory = full.value.memory.objects == objects && append.value.memory.objects == objects &&
                        rawEqual(full.value.memory.parts, append.value.memory.parts) && sameCls &&
                        rawEqual(full.value.memory.valid, append.value.memory.valid)
                    val fullHash = requireNotNull(full.value.featureSha256)
                    val firstHash = featureHashes.getOrPut(objects) { fullHash }
                    val sameFeatures = fullHash == append.value.featureSha256 && fullHash == firstHash
                    val equal = sameRetained && sameParts && sameMemory && sameFeatures
                    step.put("saved_ms", full.value.totalMs - append.value.totalMs)
                        .put("retained_bit_equal", sameRetained)
                        .put("parts_bit_equal", sameParts)
                        .put("cls_bit_equal", sameCls)
                        .put("memory_bit_equal", sameMemory)
                        .put("old_retained_changes", oldRetainedChanges(previousRetained, appendRetained, patchCounts.take(fromObjects)))
                        .put("feature_sha256_equal_across_all_runs", sameFeatures)
                    json.writeText(result.toString(2))
                    if (pair == 0 || !equal) {
                        val folder = File(output, "pair_${pair + 1}/${fromObjects}_to_$objects").apply { check(mkdirs()) }
                        writeRaw(folder, full, fullRetained)
                        writeRaw(folder, append, appendRetained)
                    }
                    check(equal) { "Pair ${pair + 1}, $fromObjects to $objects disagrees. See ${json.path}" }
                    val fullDiagnostics = requireNotNull(full.value.diagnostics)
                    val appendDiagnostics = requireNotNull(append.value.diagnostics)
                    check(fullDiagnostics[2] > 0 && appendDiagnostics[1] < fullDiagnostics[1] &&
                        appendDiagnostics[3] > 0 && appendDiagnostics[5] > 0 && current.cacheBytes > incomingBytes) {
                        "Pair ${pair + 1}, $fromObjects to $objects did not exercise reuse and successor capture. See ${json.path}"
                    }
                    val oldRows = patchCounts.take(fromObjects).sum().toLong()
                    check(appendDiagnostics[3] + appendDiagnostics[4] == oldRows &&
                        if (objects == 4) appendDiagnostics[4] == 0L else appendDiagnostics[4] > 0) {
                        "Pair ${pair + 1}, $fromObjects to $objects has unexpected witness coverage. See ${json.path}"
                    }
                }
            }
            result.put("completed", true)
            json.writeText(result.toString(2))
        }
    }

    @Test
    fun prepareRestart() {
        val output = restartDirectory()
        check(output.mkdir()) { "Expected a fresh test output directory: $output" }
        val assetsStart = System.nanoTime()
        val input = loadP189()
        val receipt = File(output, "prepare.json")
        val identity = restartIdentity(input)
        val result = input.summary
            .put("run_id", output.name)
            .put("process", processIdentity())
            .put("cache_identity_sha256", identity)
            .put("assets_check_ms", (System.nanoTime() - assetsStart) / 1e6)
            .put("completed", false)
        receipt.writeText(result.toString(2))
        withWarmPipeline(input) { b ->
            val start = thermalSnapshot()
            val three = b.prepareTeachingAppend(input.teachFiles.take(3), captureEvidence = true)
            result.put("initial", arm("initial", three.initial, start).summary
                .put("encoded_jpegs", 90).put("cache_bytes", three.cacheBytes))
            val appendStart = thermalSnapshot()
            val four = b.appendTeachingAndCache(three, input.teachFiles[3], captureEvidence = true)
            check(four.initial.memory.objects == 4)
            result.put("append", arm("append", four.initial, appendStart).summary
                .put("encoded_jpegs", 30).put("cache_bytes", four.cacheBytes))

            val cache = File(output, "four_objects.cache")
            val saveStart = System.nanoTime()
            b.saveTeachingAppend(four, cache, identity)
            result.put("save_ms", (System.nanoTime() - saveStart) / 1e6)
                .put("cache_payload_bytes", four.cacheBytes)
                .put("cache_file_bytes", cache.length())
                .put("completed", true)
            check(cache.length() >= four.cacheBytes)
            receipt.writeText(result.toString(2))
        }
    }

    @Test
    fun measureRestart() {
        val args = InstrumentationRegistry.getArguments()
        val pair = requireNotNull(args.getString("pair")).toInt()
        val mode = requireNotNull(args.getString("arm"))
        require(pair in 1..3 && mode in setOf("full", "restored"))
        val root = restartDirectory()
        val prepared = JSONObject(File(root, "prepare.json").readText())
        check(prepared.getBoolean("completed"))
        val folder = File(root, "pair_$pair").apply { mkdir() }
        val receipt = File(folder, "$mode.json")
        check(receipt.createNewFile()) { "Expected a fresh arm receipt: $receipt" }
        val process = processIdentity()
        val oldProcess = prepared.getJSONObject("process")
        check(process.getInt("pid") != oldProcess.getInt("pid") ||
            process.getLong("stat_start_ticks") != oldProcess.getLong("stat_start_ticks")) {
            "The prepare process is still running"
        }
        val assetsStart = System.nanoTime()
        val input = loadP189()
        val identity = restartIdentity(input)
        check(identity == prepared.getString("cache_identity_sha256")) { "Teaching assets changed after saving the cache" }
        val result = input.summary.put("run_id", root.name).put("pair", pair).put("arm", mode)
            .put("process", process).put("prepare_process", oldProcess)
            .put("cache_identity_sha256", identity)
            .put("assets_check_ms", (System.nanoTime() - assetsStart) / 1e6)
            .put("completed", false)
        receipt.writeText(result.toString(2))

        withWarmPipeline(input, input.teachFiles.last().first()) { b ->
            val thermalBefore = thermalSnapshot()
            val before = memorySnapshot()
            val build: TeachablePipeline.TeachingMemory
            val restoreMs: Double
            result.put("memory_before", before)
            if (mode == "restored") {
                val cache = File(root, "four_objects.cache")
                val restoreStart = System.nanoTime()
                val restored = b.restoreTeachingAppend(cache, identity)
                restoreMs = (System.nanoTime() - restoreStart) / 1e6
                result.put("memory_after_restore", memorySnapshot())
                    .put("restore_ms", restoreMs)
                    .put("restored_cache_bytes", restored.cacheBytes)
                check(restored.cacheBytes == prepared.getLong("cache_payload_bytes"))
                val successor = b.appendTeachingAndCache(restored, input.teachFiles.last(), captureEvidence = true,
                    captureFeatureHash = false)
                build = successor.initial
                result.put("successor_cache_bytes", successor.cacheBytes)
            } else {
                restoreMs = 0.0
                build = b.buildTeachingMemory(input.teachFiles, captureEvidence = true, captureFeatureHash = false)
            }
            result.put("memory_after_build", memorySnapshot())
            val measured = arm(mode, build, thermalBefore)
            val retained = requireNotNull(build.retained)
            val diagnostics = requireNotNull(build.diagnostics)
            check(build.memory.objects == 5 && retained.count { it >= 0 } == 5760 && build.built.parts.size == 160 * B_DIM)
            check(if (mode == "restored") diagnostics[3] > 0 && diagnostics[4] > 0 && diagnostics[5] > 0
                else diagnostics[2] > 0)
            result.put("restore_ms", restoreMs)
                .put("restore_plus_teaching_ms", restoreMs + build.totalMs)
                .put("result", measured.summary.put("encoded_jpegs", if (mode == "restored") 30 else 150))
            receipt.writeText(result.toString(2))
            writeRaw(folder, measured, retained)
            writeFloats(File(folder, "${mode}_padded_parts.f32"), build.memory.parts)
            writeFloats(File(folder, "${mode}_valid.f32"), build.memory.valid)
            receipt.writeText(result.put("completed", true).toString(2))
        }
    }

    private fun restartDirectory(): File {
        val runId = InstrumentationRegistry.getArguments().getString("runId") ?: "restart_p189"
        require(runId.matches(Regex("[a-zA-Z0-9_-]+")))
        return File(File(context.filesDir, "b_memory_restart").apply { mkdir() }, runId)
    }

    private fun restartIdentity(input: Input): String {
        val images = input.summary.getJSONObject("teaching_jpeg_sha256")
        val models = File(context.filesDir, "bwcl")
        val identities = buildList {
            add(input.summary.getString("manifest_sha256"))
            add(sha256File(File(models, "backbone_b_cached.dlc")))
            add(sha256File(File(models, "bhead_fp32.dlc")))
            for (objectFiles in input.teachFiles.take(4)) {
                add(objectFiles.joinToString(",") { images.getString(it.name) })
            }
        }
        return sha256(identities.joinToString("\n").toByteArray(Charsets.UTF_8))
    }

    private fun processIdentity(): JSONObject {
        // The comm field in /proc/self/stat can contain spaces. Field 22 is the process start in kernel ticks.
        val startTicks = File("/proc/self/stat").readText().substringAfterLast(") ").split(' ')[19].toLong()
        return JSONObject().put("pid", Process.myPid())
            .put("start_elapsed_realtime_ms", Process.getStartElapsedRealtime())
            .put("stat_start_ticks", startTicks)
            .put("sample_elapsed_realtime_ms", SystemClock.elapsedRealtime())
    }

    private fun memorySnapshot(): JSONObject {
        val status = File("/proc/self/status").readLines()
        fun read(key: String) = status.single { it.startsWith("$key:") }.substringAfter(':').trim()
            .substringBefore(' ').toLong()
        return JSONObject().put("vm_hwm_kib", read("VmHWM")).put("vm_rss_kib", read("VmRSS"))
    }

    private fun sha256File(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val length = stream.read(buffer)
                if (length < 0) break
                digest.update(buffer, 0, length)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
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

    private fun withWarmPipeline(
        input: Input, warmFile: File = input.teachFiles.first().first(), run: (TeachablePipeline) -> Unit,
    ) {
        val models = File(context.filesDir, "bwcl")
        val initStart = System.nanoTime()
        TeachablePipeline(
            File(models, "backbone_b_cached.dlc"), File(models, "bhead_fp32.dlc"),
            context.applicationInfo.nativeLibraryDir, dim = B_DIM,
        ).use { b ->
            input.summary.put("init_ms", (System.nanoTime() - initStart) / 1e6)
            val warmStart = System.nanoTime()
            val warm = WclPipeline.imageOf(warmFile)
            repeat(3) { b.tokens(warm) }
            input.summary.put("warmup_ms", (System.nanoTime() - warmStart) / 1e6)
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
