package com.linnan.blindassist.benchmark

import android.content.Intent
import android.content.IntentFilter
import android.graphics.BitmapFactory
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.vision.ImagePreprocessor
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import java.util.zip.GZIPOutputStream

/**
 * R2-L1E sequence-sharded Android Canvas/TFLite raw exporter.
 *
 * This exporter has no selection, scoring, Android shadow, H2, human-outcome, or
 * production authority. One invocation accepts exactly one frozen masked ledger.
 */
@RunWith(AndroidJUnit4::class)
class UstrfR2L1ExploratoryProfileDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun exportOneFrozenMaskedLedger() {
        if (arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() != true) {
            assumeTrue("R2-L1E export was not requested", false)
            return
        }
        val inputFile = privateFile(requireNotNull(arguments.getString(ARG_INPUT)), "R2-L1E manifest")
        val input = JSONObject(inputFile.readText(Charsets.UTF_8))
        check(input.getString("schema") == INPUT_SCHEMA)
        check(input.getString("stage") == "R2-L1E")
        check(input.getJSONArray("input_shape").toIntArray().contentEquals(intArrayOf(1, 320, 320, 3)))
        check(input.getJSONArray("output_shape").toIntArray().contentEquals(intArrayOf(1, 84, 2100)))
        check(input.getInt("person_class_index") == 0)
        check(input.getDouble("confidence_threshold") == 0.35)
        check(input.getDouble("nms_iou_threshold") == 0.45)
        check(sha256Asset(MODEL_ASSET) == input.getString("model_sha256"))
        check(sha256Asset(LABELS_ASSET) == input.getString("labels_sha256"))
        val frames = input.getJSONArray("frames")
        check(frames.length() == input.getInt("frame_count"))
        check(frames.length() > 0)

        val rawFile = privateNewOutputFile(requireNotNull(arguments.getString(ARG_RAW_OUTPUT)))
        val receiptFile = privateOutputFile(requireNotNull(arguments.getString(ARG_RECEIPT_OUTPUT)))
        val rawDigest = MessageDigest.getInstance("SHA-256")
        val rows = JSONArray()
        val power = targetContext.getSystemService(PowerManager::class.java)
        val startBatteryC = batteryTemperatureC()
        var maxBatteryC = startBatteryC
        var maxThermalStatus = power.currentThermalStatus
        var completed = 0
        var error: String? = null
        var rawCompressedSha: String? = null
        var rawUncompressedSha: String? = null
        var rawCompressedBytes = 0L

        val interpreter = Interpreter(loadMappedAsset(MODEL_ASSET), Interpreter.Options().apply { setNumThreads(4) })
        try {
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            check(inputTensor.dataType() == DataType.FLOAT32)
            check(inputTensor.shape().contentEquals(intArrayOf(1, 320, 320, 3)))
            check(outputTensor.dataType() == DataType.FLOAT32)
            check(outputTensor.shape().contentEquals(intArrayOf(1, 84, 2100)))
            val preprocessor = ImagePreprocessor(320)
            val outputBuffer = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())
            GZIPOutputStream(BufferedOutputStream(FileOutputStream(rawFile), STREAM_BUFFER_BYTES)).use { rawStream ->
                for (index in 0 until frames.length()) {
                    val expected = frames.getJSONObject(index)
                    val thermalStatus = power.currentThermalStatus
                    val batteryC = batteryTemperatureC()
                    maxThermalStatus = maxOf(maxThermalStatus, thermalStatus)
                    maxBatteryC = maxOf(maxBatteryC, batteryC)
                    check(thermalStatus <= MAX_THERMAL_STATUS) { "thermal_status_guard:$thermalStatus" }
                    check(batteryC <= MAX_BATTERY_C) { "battery_temperature_guard:$batteryC" }
                    check(batteryC - startBatteryC <= MAX_BATTERY_RISE_C) {
                        "battery_temperature_rise_guard:start=$startBatteryC current=$batteryC"
                    }
                    val imageFile = privateChild(inputFile.parentFile, expected.getString("image_path"), "masked frame")
                    check(sha256File(imageFile) == expected.getString("image_sha256")) {
                        "image_hash_mismatch:${expected.getString("frame_id")}"
                    }
                    val bitmap = requireNotNull(BitmapFactory.decodeFile(imageFile.absolutePath)) {
                        "image_decode_failure:${expected.getString("frame_id")}"
                    }
                    val startedNs = SystemClock.elapsedRealtimeNanos()
                    try {
                        val prepared = preprocessor.prepare(bitmap)
                        outputBuffer.rewind()
                        interpreter.run(prepared.buffer, outputBuffer)
                    } finally {
                        bitmap.recycle()
                    }
                    val consumedNs = SystemClock.elapsedRealtimeNanos()
                    val rawBytes = bufferBytes(outputBuffer)
                    val rawSha = sha256Bytes(rawBytes)
                    rawStream.write(rawBytes)
                    rawDigest.update(rawBytes)
                    rows.put(
                        JSONObject()
                            .put("source_id", expected.getString("source_id"))
                            .put("sequence_id", expected.getString("sequence_id"))
                            .put("frame_id", expected.getString("frame_id"))
                            .put("source_capture_timestamp_ns", expected.getLong("source_capture_timestamp_ns"))
                            .put("image_sha256", expected.getString("image_sha256"))
                            .put("android_raw_output_sha256", rawSha)
                            .put("detector_processing_latency_ns", consumedNs - startedNs)
                            .put("thermal_status", thermalStatus)
                            .put("battery_temperature_c", batteryC)
                    )
                    completed++
                }
            }
            rawCompressedSha = sha256File(rawFile)
            rawUncompressedSha = rawDigest.digest().toHex()
            rawCompressedBytes = rawFile.length()
        } catch (failure: Throwable) {
            error = failure.javaClass.simpleName + ":" + (failure.message ?: "unknown")
        } finally {
            interpreter.close()
            val finalBatteryC = batteryTemperatureC()
            maxBatteryC = maxOf(maxBatteryC, finalBatteryC)
            maxThermalStatus = maxOf(maxThermalStatus, power.currentThermalStatus)
            val receipt = JSONObject()
                .put("schema", OUTPUT_SCHEMA)
                .put("stage", "R2-L1E")
                .put("status", if (error == null && completed == frames.length()) "DEVICE_RAW_SHARD_COMPLETE" else "DEVICE_RAW_SHARD_INCOMPLETE")
                .put("input_manifest_sha256", sha256File(inputFile))
                .put("source_id", input.getString("source_id"))
                .put("sequence_id", input.getString("sequence_id"))
                .put("frame_mask_sha256", input.getString("frame_mask_sha256"))
                .put("expected_frame_count", frames.length())
                .put("completed_frame_count", completed)
                .put("error", error)
                .put("device", JSONObject().put("model", Build.MODEL).put("sdk", Build.VERSION.SDK_INT))
                .put("runtime", "production_same_android_canvas_tflite_cpu_4_threads")
                .put("canonical_raw_stream", JSONObject()
                    .put("encoding", "gzip_concatenated_float32_native_little_endian")
                    .put("bytes_per_frame_uncompressed", CANONICAL_RAW_BYTES_PER_FRAME)
                    .put("uncompressed_sha256", rawUncompressedSha)
                    .put("compressed_sha256", rawCompressedSha)
                    .put("compressed_bytes", rawCompressedBytes))
                .put("guards", JSONObject()
                    .put("maximum_thermal_status", MAX_THERMAL_STATUS)
                    .put("maximum_battery_temperature_c", MAX_BATTERY_C)
                    .put("maximum_battery_temperature_rise_c", MAX_BATTERY_RISE_C)
                    .put("start_battery_temperature_c", startBatteryC)
                    .put("final_battery_temperature_c", finalBatteryC)
                    .put("observed_maximum_battery_temperature_c", maxBatteryC)
                    .put("observed_maximum_thermal_status", maxThermalStatus))
                .put("frames", rows)
                .put("authority", JSONObject()
                    .put("exploratory_profile_input_only", true)
                    .put("selection", false)
                    .put("android_shadow", false)
                    .put("h2", false)
                    .put("human_outcome", false)
                    .put("production", false))
            receiptFile.writeText(receipt.toString(), Charsets.UTF_8)
        }
        check(error == null && completed == frames.length()) { error ?: "incomplete R2-L1E device shard" }
    }

    private fun batteryTemperatureC(): Double {
        val intent: Intent = requireNotNull(
            targetContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        )
        return intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) / 10.0
    }

    private fun JSONArray.toIntArray() = IntArray(length()) { getInt(it) }

    private fun privateFile(relative: String, label: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(root, relative).canonicalFile.also {
            check(it.path.startsWith(root.path + File.separator) && it.isFile) {
                "$label unavailable: $relative"
            }
        }
    }

    private fun privateChild(parent: File?, relative: String, label: String): File {
        check(parent != null && relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(parent, relative).canonicalFile.also {
            check(it.path.startsWith(root.path + File.separator) && it.isFile) {
                "$label unavailable: $relative"
            }
        }
    }

    private fun privateOutputFile(relative: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(root, relative).canonicalFile.also {
            check(it.path.startsWith(root.path + File.separator))
            it.parentFile?.mkdirs()
        }
    }

    private fun privateNewOutputFile(relative: String): File = privateOutputFile(relative).also {
        check(!it.exists()) { "refusing to overwrite R2-L1E raw shard: $relative" }
    }

    private fun bufferBytes(buffer: ByteBuffer): ByteArray {
        val duplicate = buffer.duplicate().order(ByteOrder.nativeOrder())
        duplicate.rewind()
        return ByteArray(duplicate.remaining()).also(duplicate::get)
    }

    private fun sha256Bytes(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).toHex()

    private fun ByteArray.toHex() = joinToString("") { "%02x".format(it) }
    private fun sha256File(file: File) = file.inputStream().use(::digest)
    private fun sha256Asset(path: String) = testContext.assets.open(path).use(::digest)

    private fun digest(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        return digest.digest().toHex()
    }

    private fun loadMappedAsset(assetName: String): MappedByteBuffer {
        val descriptor = testContext.assets.openFd(assetName)
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength
            )
        }
    }

    private companion object {
        const val INPUT_SCHEMA = "blindassist_ustrf_route_target_l1e_device_shard_manifest_r1"
        const val OUTPUT_SCHEMA = "blindassist_ustrf_route_target_l1e_device_raw_receipt_r1"
        const val MODEL_ASSET = "yolo11n_fp16_320.tflite"
        const val LABELS_ASSET = "coco_labels.txt"
        const val CANONICAL_RAW_BYTES_PER_FRAME = 84 * 2100 * 4
        const val STREAM_BUFFER_BYTES = 1024 * 1024
        const val MAX_THERMAL_STATUS = 2
        const val MAX_BATTERY_C = 45.0
        const val MAX_BATTERY_RISE_C = 8.0
        const val ARG_REQUIRED = "ustrfR2L1eRequired"
        const val ARG_INPUT = "ustrfR2L1eInput"
        const val ARG_RAW_OUTPUT = "ustrfR2L1eRawOutput"
        const val ARG_RECEIPT_OUTPUT = "ustrfR2L1eReceiptOutput"
    }
}
