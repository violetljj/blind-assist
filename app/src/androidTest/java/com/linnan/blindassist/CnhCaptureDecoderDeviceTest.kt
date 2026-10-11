package com.linnan.blindassist

import android.os.Build
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.capture.CnhCaptureDecoder
import com.linnan.blindassist.capture.CnhCaptureRecord
import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest

/** Explicit actual-raw parser replay on Android's org.json; this does not acquire sensor data. */
@RunWith(AndroidJUnit4::class)
class CnhCaptureDecoderDeviceTest {
    @Test
    fun actualH3RawPassesAndroidDecoder() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("Push actual captured bytes to app cache before this opt-in diagnostic",
            args.getString("cnhDecoderActualRaw") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.cacheDir, "cnh-decoder-device-test").canonicalFile
        val source = File(root, args.getString("cnhDecoderRawFile") ?: "actual-h3.raw").canonicalFile
        require(source.parentFile == root && source.isFile) { "Actual raw payload missing from test cache" }
        val raw = source.readBytes()
        assertTrue("Actual raw payload is empty", raw.isNotEmpty())

        val decoder = CnhCaptureDecoder()
        val records = mutableListOf<CnhCaptureRecord>()
        var offset = 0
        while (offset < raw.size) {
            val next = (offset + 8192).coerceAtMost(raw.size)
            records += decoder.feed(raw.copyOfRange(offset, next), SystemClock.elapsedRealtimeNanos())
            offset = next
        }
        decoder.finish()?.let(records::add)

        val tokens = JSONArray()
        val sensorErrors = JSONArray()
        var frameDiagnostics = 0
        raw.toString(Charsets.UTF_8).lineSequence().filter { it.isNotBlank() }.forEach { line ->
            val diagnostic = JSONObject()
            try {
                val tokener = JSONTokener(line)
                val parsed = tokener.nextValue()
                diagnostic.put("next_value_class", parsed.javaClass.name)
                diagnostic.put("next_clean_code", tokener.nextClean().code)
                if (parsed is JSONObject) {
                    val kind = parsed.optString("type")
                    diagnostic.put("type", kind)
                    if (kind == "error") sensorErrors.put(parsed)
                    if (kind == "cnh_frame") {
                        frameDiagnostics++
                        val numericTypes = JSONObject()
                        listOf("seq", "ms", "read_us", "stream", "rows", "cols", "bins").forEach { key ->
                            numericTypes.put(key, parsed.opt(key)?.javaClass?.name ?: "null")
                        }
                        numericTypes.put("hist_raw_0_0", parsed.optJSONArray("hist_raw")
                            ?.optJSONArray(0)?.opt(0)?.javaClass?.name ?: "null")
                        numericTypes.put("hist_scaler_0_0", parsed.optJSONArray("hist_scaler")
                            ?.optJSONArray(0)?.opt(0)?.javaClass?.name ?: "null")
                        numericTypes.put("ambient_raw_0", parsed.optJSONArray("ambient_raw")
                            ?.opt(0)?.javaClass?.name ?: "null")
                        numericTypes.put("cnh_header_4", parsed.optJSONArray("cnh_header")
                            ?.opt(4)?.javaClass?.name ?: "null")
                        diagnostic.put("numeric_types", numericTypes)
                        diagnostic.put("sequence", parsed.opt("seq"))
                    }
                    if (kind != "cnh_frame" || frameDiagnostics <= 3) tokens.put(diagnostic)
                } else tokens.put(diagnostic)
            } catch (error: Exception) {
                diagnostic.put("json_exception", error.message ?: error.javaClass.name)
                diagnostic.put("line_bytes", line.toByteArray(Charsets.UTF_8).size)
                tokens.put(diagnostic)
            }
        }

        val validFrames = records.count { it.frame != null }
        val issues = JSONArray()
        records.filter { it.issues.isNotEmpty() }.forEach { record ->
            issues.put(JSONObject().put("line", record.lineNumber).put("kind", record.kind)
                .put("bytes", record.rawLine.size).put("issues", JSONArray(record.issues)))
        }
        val receipt = JSONObject()
            .put("schema", "blindassist-cnh-android-decoder-diagnostic-v1")
            .put("device_model", Build.MODEL).put("android_api", Build.VERSION.SDK_INT)
            .put("source_file", source.absolutePath).put("source_bytes", raw.size)
            .put("source_sha256", MessageDigest.getInstance("SHA-256").digest(raw)
                .joinToString("") { "%02x".format(it.toInt() and 255) })
            .put("record_count", records.size).put("valid_frame_count", validFrames)
            .put("issues", issues).put("sensor_errors", sensorErrors)
            .put("android_json_token_diagnostics", tokens)
            .put("boundary", "Actual captured bytes replayed through Android parser; host receipt timestamps generated for parser replay, not actual sensor synchronization")
        val output = File(root, "actual-h3-android-decoder.json")
        output.writeText(receipt.toString(2), Charsets.UTF_8)
        assertTrue("Expected at least 3 actual valid CNH frames; got $validFrames; " +
            "sensor_errors=$sensorErrors; issues=$issues; diagnostic=${output.absolutePath}", validFrames >= 3)
    }
}
