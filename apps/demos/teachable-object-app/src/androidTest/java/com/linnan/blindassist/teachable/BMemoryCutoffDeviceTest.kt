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

        val output = File(context.filesDir, "b_memory_cutoff")
        check(output.mkdir()) { "Expected a fresh test output directory: $output" }
        val runs = JSONArray()
        val result = JSONObject()
            .put("user", "P189")
            .put("device", Build.MODEL)
            .put("fingerprint", Build.FINGERPRINT)
            .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
            .put("manifest_sha256", sha256(manifest))
            .put("teaching_jpeg_sha256", JSONObject(teachFiles.flatten().associate { it.name to sha256(it.readBytes()) }))
            .put("objects", meta.getJSONArray("objects"))
            .put("frames_per_object", JSONArray(teachFiles.map { it.size }))
            .put("patches_per_frame", TeachablePipeline.GRID * TeachablePipeline.GRID)
            .put("dim", B_DIM)
            .put("pairs", runs)
        val json = File(output, "p189.json")
        val models = File(context.filesDir, "bwcl")
        val initStart = System.nanoTime()
        TeachablePipeline(
            File(models, "backbone_b_cached.dlc"), File(models, "bhead_fp32.dlc"),
            context.applicationInfo.nativeLibraryDir, dim = B_DIM,
        ).use { b ->
            result.put("init_ms", (System.nanoTime() - initStart) / 1e6)
            val warm = WclPipeline.imageOf(teachFiles.first().first())
            repeat(3) { b.tokens(warm) }
            var firstFeatureHash: String? = null
            repeat(3) { pair ->
                val cutoffFirst = pair % 2 == 1
                fun measure(cutoff: Boolean) = measured(cutoff) {
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

    private class Arm(val name: String, val value: TeachablePipeline.TeachingMemory, val summary: JSONObject)

    private fun measured(cutoff: Boolean, run: () -> TeachablePipeline.TeachingMemory): Arm {
        val start = thermalSnapshot()
        val value = run()
        val stop = thermalSnapshot()
        val diagnostics = requireNotNull(value.diagnostics)
        val retained = requireNotNull(value.retained).count { it >= 0 }
        return Arm(if (cutoff) "cutoff" else "original", value, JSONObject()
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
            .put("stop", stop))
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
