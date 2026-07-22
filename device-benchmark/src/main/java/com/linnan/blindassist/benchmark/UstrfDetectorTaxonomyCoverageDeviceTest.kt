package com.linnan.blindassist.benchmark

import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
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
import java.io.File
import java.io.BufferedOutputStream
import java.io.FileOutputStream
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import java.util.zip.GZIPOutputStream
import kotlin.math.max
import kotlin.math.roundToInt

/** Full frozen-frame receipt for Android Canvas input and raw detector output parity. */
@RunWith(AndroidJUnit4::class)
class UstrfDetectorTaxonomyCoverageDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun exportFrozenCanvasAndRawTensorReceipts() {
        if (arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() != true) {
            assumeTrue("detector taxonomy parity was not requested", false)
            return
        }
        val inputFile = privateFile(requireNotNull(arguments.getString(ARG_INPUT)), "detector taxonomy manifest")
        val input = JSONObject(inputFile.readText(Charsets.UTF_8))
        check(input.getString("schema") == INPUT_SCHEMA)
        check(input.getInt("frame_count") == REQUIRED_FRAME_COUNT)
        check(input.getJSONArray("input_shape").toIntArray().contentEquals(intArrayOf(1, 320, 320, 3)))
        check(input.getJSONArray("output_shape").toIntArray().contentEquals(intArrayOf(1, 84, 2100)))
        check(input.getInt("person_class_index") == 0)
        check(input.getDouble("confidence_threshold") == 0.35)
        check(input.getDouble("nms_iou_threshold") == 0.45)
        check(sha256Asset(MODEL_ASSET) == input.getString("model_sha256"))
        check(sha256Asset(LABELS_ASSET) == input.getString("labels_sha256"))
        val labels = testContext.assets.open(LABELS_ASSET).bufferedReader().useLines { lines ->
            lines.map(String::trim).filter(String::isNotEmpty).toList()
        }
        check(labels.size == 80 && labels[0] == "person")
        val interpreter = Interpreter(loadMappedAsset(MODEL_ASSET), Interpreter.Options().apply { setNumThreads(4) })
        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        check(inputTensor.dataType() == DataType.FLOAT32 && inputTensor.shape().contentEquals(intArrayOf(1, 320, 320, 3)))
        check(outputTensor.dataType() == DataType.FLOAT32 && outputTensor.shape().contentEquals(intArrayOf(1, 84, 2100)))
        val preprocessor = ImagePreprocessor(320)
        val outputBuffer = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())
        val canonicalInputFile = privateNewOutputFile(requireNotNull(arguments.getString(ARG_CANONICAL_INPUT)))
        val canonicalRawFile = privateNewOutputFile(requireNotNull(arguments.getString(ARG_CANONICAL_RAW)))
        val canonicalInputDigest = MessageDigest.getInstance("SHA-256")
        val canonicalRawDigest = MessageDigest.getInstance("SHA-256")
        val rows = JSONArray()
        var failures = 0
        var inputHashMatches = 0
        var rawHashMatches = 0
        var personFrames = 0
        GZIPOutputStream(BufferedOutputStream(FileOutputStream(canonicalInputFile), STREAM_BUFFER_BYTES)).use { canonicalInput ->
        GZIPOutputStream(BufferedOutputStream(FileOutputStream(canonicalRawFile), STREAM_BUFFER_BYTES)).use { canonicalRaw ->
        try {
            val frames = input.getJSONArray("frames")
            for (index in 0 until frames.length()) {
                val expected = frames.getJSONObject(index)
                val imageFile = privateChild(inputFile.parentFile, expected.getString("image_path"), "frozen frame")
                val imageHash = sha256File(imageFile)
                var rowFailure: String? = null
                var androidInputHash: String? = null
                var androidOutputHash: String? = null
                var personMax: Float? = null
                var allClassMax: Float? = null
                val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
                if (bitmap == null || imageHash != expected.getString("image_sha256")) {
                    rowFailure = "image_decode_or_hash_failure"
                } else {
                    try {
                        val prepared = preprocessor.prepare(bitmap)
                        androidInputHash = sha256Buffer(prepared.buffer)
                        val canonicalRgb = canonicalRgbBytes(prepared.buffer)
                        canonicalInput.write(canonicalRgb)
                        canonicalInputDigest.update(canonicalRgb)
                        outputBuffer.rewind()
                        interpreter.run(prepared.buffer, outputBuffer)
                        androidOutputHash = sha256Buffer(outputBuffer)
                        val canonicalRawBytes = bufferBytes(outputBuffer)
                        canonicalRaw.write(canonicalRawBytes)
                        canonicalRawDigest.update(canonicalRawBytes)
                        val values = FloatArray(outputTensor.numElements())
                        outputBuffer.rewind()
                        outputBuffer.asFloatBuffer().get(values)
                        var localPersonMax = 0f
                        var localAllMax = 0f
                        val predictions = 2100
                        for (prediction in 0 until predictions) {
                            for (classId in labels.indices) {
                                val value = values[(4 + classId) * predictions + prediction]
                                check(value.isFinite())
                                localAllMax = max(localAllMax, value)
                                if (classId == 0) localPersonMax = max(localPersonMax, value)
                            }
                        }
                        personMax = localPersonMax
                        allClassMax = localAllMax
                        if (localPersonMax >= 0.35f) personFrames++
                    } catch (error: Throwable) {
                        rowFailure = error.javaClass.simpleName + ":" + (error.message ?: "unknown")
                    } finally {
                        bitmap.recycle()
                    }
                }
                val inputMatch = androidInputHash == expected.getString("host_input_tensor_sha256")
                val rawMatch = androidOutputHash == expected.getString("host_raw_output_sha256")
                if (inputMatch) inputHashMatches++
                if (rawMatch) rawHashMatches++
                if (rowFailure != null) failures++
                rows.put(JSONObject()
                    .put("source_name", expected.getString("source_name"))
                    .put("frame_id", expected.getString("frame_id"))
                    .put("image_sha256", imageHash)
                    .put("android_input_tensor_sha256", androidInputHash)
                    .put("host_input_tensor_sha256", expected.getString("host_input_tensor_sha256"))
                    .put("input_tensor_exact_match", inputMatch)
                    .put("android_raw_output_sha256", androidOutputHash)
                    .put("host_raw_output_sha256", expected.getString("host_raw_output_sha256"))
                    .put("raw_output_exact_match", rawMatch)
                    .put("android_raw_person_max_confidence", personMax)
                    .put("host_raw_person_max_confidence", expected.getDouble("host_raw_person_max_confidence"))
                    .put("android_raw_all_class_max_confidence", allClassMax)
                    .put("failure", rowFailure))
                if ((index + 1) % 100 == 0) Log.i(TAG, "frames=${index + 1}/${frames.length()}")
            }
        } finally {
            interpreter.close()
        }
        }
        }
        val output = JSONObject()
            .put("schema", OUTPUT_SCHEMA)
            .put("input_manifest_sha256", sha256File(inputFile))
            .put("device", JSONObject().put("model", Build.MODEL).put("sdk", Build.VERSION.SDK_INT))
            .put("runtime", "tflite_cpu_4_threads")
            .put("frame_count", REQUIRED_FRAME_COUNT)
            .put("failure_count", failures)
            .put("input_tensor_exact_match_count", inputHashMatches)
            .put("raw_output_exact_match_count", rawHashMatches)
            .put("person_frame_count", personFrames)
            .put("canonical_input_stream", JSONObject()
                .put("encoding", "gzip_concatenated_rgb_u8_nhwc")
                .put("bytes_per_frame_uncompressed", CANONICAL_RGB_BYTES_PER_FRAME)
                .put("uncompressed_sha256", canonicalInputDigest.digest().toHex())
                .put("compressed_sha256", sha256File(canonicalInputFile))
                .put("compressed_bytes", canonicalInputFile.length()))
            .put("canonical_raw_stream", JSONObject()
                .put("encoding", "gzip_concatenated_float32_native_little_endian")
                .put("bytes_per_frame_uncompressed", CANONICAL_RAW_BYTES_PER_FRAME)
                .put("uncompressed_sha256", canonicalRawDigest.digest().toHex())
                .put("compressed_sha256", sha256File(canonicalRawFile))
                .put("compressed_bytes", canonicalRawFile.length()))
            .put("full_exact_parity_passed", failures == 0 && inputHashMatches == REQUIRED_FRAME_COUNT && rawHashMatches == REQUIRED_FRAME_COUNT)
            .put("frames", rows)
            .put("authority", JSONObject().put("benchmark_only", true).put("tracker_reopened", false)
                .put("h2_reopened", false).put("app_change_authorized", false).put("production_authorized", false))
        privateOutputFile(requireNotNull(arguments.getString(ARG_OUTPUT))).writeText(output.toString(), Charsets.UTF_8)
        Log.i(TAG, "complete input=$inputHashMatches raw=$rawHashMatches failures=$failures person=$personFrames")
        check(failures == 0) { "device parity execution had $failures frame failures" }
    }

    private fun JSONArray.toIntArray() = IntArray(length()) { getInt(it) }
    private fun privateFile(relative: String, label: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(root, relative).canonicalFile.also {
            check(it.path.startsWith(root.path + File.separator) && it.isFile) {
                "$label unavailable: $relative root=${root.path} resolved=${it.path} exists=${it.exists()} file=${it.isFile} readable=${it.canRead()}"
            }
        }
    }
    private fun privateChild(parent: File?, relative: String, label: String): File {
        check(parent != null && relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(parent, relative).canonicalFile.also {
            check(it.path.startsWith(root.path + File.separator) && it.isFile) { "$label unavailable: $relative" }
        }
    }
    private fun privateOutputFile(relative: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(root, relative).canonicalFile.also {
            check(it.path.startsWith(root.path + File.separator)); it.parentFile?.mkdirs()
        }
    }
    private fun privateNewOutputFile(relative: String): File = privateOutputFile(relative).also {
        check(!it.exists()) { "refusing to overwrite canonical stream: $relative" }
    }
    private fun sha256Buffer(buffer: ByteBuffer): String {
        val duplicate = buffer.duplicate().order(ByteOrder.nativeOrder())
        duplicate.rewind()
        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = ByteArray(64 * 1024)
        while (duplicate.hasRemaining()) {
            val count = minOf(bytes.size, duplicate.remaining())
            duplicate.get(bytes, 0, count)
            digest.update(bytes, 0, count)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
    private fun bufferBytes(buffer: ByteBuffer): ByteArray {
        val duplicate = buffer.duplicate().order(ByteOrder.nativeOrder())
        duplicate.rewind()
        return ByteArray(duplicate.remaining()).also(duplicate::get)
    }
    private fun canonicalRgbBytes(buffer: ByteBuffer): ByteArray {
        val duplicate = buffer.duplicate().order(ByteOrder.nativeOrder())
        duplicate.rewind()
        val result = ByteArray(CANONICAL_RGB_BYTES_PER_FRAME)
        for (index in result.indices) {
            val value = duplicate.float
            check(value.isFinite() && value in 0f..1f)
            result[index] = (value * 255f).roundToInt().coerceIn(0, 255).toByte()
        }
        check(!duplicate.hasRemaining())
        return result
    }
    private fun ByteArray.toHex() = joinToString("") { "%02x".format(it) }
    private fun sha256File(file: File) = file.inputStream().use(::digest)
    private fun sha256Asset(path: String) = testContext.assets.open(path).use(::digest)
    private fun digest(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer); if (count < 0) break
            digest.update(buffer, 0, count)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
    private fun loadMappedAsset(assetName: String): MappedByteBuffer {
        val descriptor = testContext.assets.openFd(assetName)
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            return stream.channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
        }
    }

    private companion object {
        const val INPUT_SCHEMA = "blindassist_ustrf_detector_taxonomy_device_input_v1"
        const val OUTPUT_SCHEMA = "blindassist_ustrf_detector_taxonomy_device_output_v1"
        const val MODEL_ASSET = "yolo11n_fp16_320.tflite"
        const val LABELS_ASSET = "coco_labels.txt"
        const val REQUIRED_FRAME_COUNT = 4594
        const val CANONICAL_RGB_BYTES_PER_FRAME = 320 * 320 * 3
        const val CANONICAL_RAW_BYTES_PER_FRAME = 84 * 2100 * 4
        const val STREAM_BUFFER_BYTES = 1024 * 1024
        const val ARG_REQUIRED = "ustrfDetectorTaxonomyRequired"
        const val ARG_INPUT = "ustrfDetectorTaxonomyInput"
        const val ARG_OUTPUT = "ustrfDetectorTaxonomyOutput"
        const val ARG_CANONICAL_INPUT = "ustrfDetectorTaxonomyCanonicalInput"
        const val ARG_CANONICAL_RAW = "ustrfDetectorTaxonomyCanonicalRaw"
        const val TAG = "UstrfDetectorTaxonomy"
    }
}
