package com.linnan.blindassist.device.glasses

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import com.linnan.blindassist.risk.DemoTofCell
import com.linnan.blindassist.risk.DemoTofDecision
import com.linnan.blindassist.risk.TofCorridorDemo
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL

data class HardwareDemoSnapshot(
    val mode: String,
    val runId: String,
    val cameraSequence: Long?,
    val tofSequence: Long?,
    val image: Bitmap?,
    val cells: List<DemoTofCell>,
    val rows: Int,
    val cols: Int,
    val decision: DemoTofDecision,
    val liveUsable: Boolean,
    val status: String,
    val cameraUsable: Boolean = liveUsable,
    val tofUsable: Boolean = liveUsable,
    val cameraAgeMs: Long? = null,
    val tofAgeMs: Long? = null
)

/** Read-only USB/localhost bridge. No capture controls or device-clock comparisons. */
class HardwareDemoClient(
    private val fetch: (String) -> ByteArray = ::readDemoBytes,
    private val clockMs: () -> Long = SystemClock::elapsedRealtime
) {
    private var run = ""
    private var lastMode = ""
    private var replayStart = 0L
    private var replayPosition = 0.0
    private var cameraSeq: Long? = null
    private var tofSeq: Long? = null
    private var cameraChanged = 0L
    private var tofChanged = 0L
    private var imageUrl: String? = null
    private var image: Bitmap? = null

    fun poll(): HardwareDemoSnapshot {
        val requested = clockMs()
        val state = JSONObject(fetch("/api/state?at_ms=$replayPosition").toString(Charsets.UTF_8))
        val mode = state.getString("mode")
        val runId = state.optString("run_id", "")
        val now = clockMs()
        if (run != runId || lastMode != mode) {
            run = runId
            lastMode = mode
            replayStart = now
            cameraSeq = null
            tofSeq = null
            imageUrl = null
            image = null
        }
        val duration = state.optDouble("duration_ms", 0.0)
        replayPosition = if (mode == "replay" && duration.isFinite() && duration > 0.0)
            (now - replayStart).toDouble() % duration else 0.0
        val camera = state.optJSONObject("camera")
        val tof = state.optJSONObject("tof")
        val nextCamera = camera?.getLong("seq")
        val nextTof = tof?.getLong("seq")
        if (nextCamera != cameraSeq) { cameraSeq = nextCamera; cameraChanged = now }
        if (nextTof != tofSeq) { tofSeq = nextTof; tofChanged = now }
        val rows = tof?.optInt("rows", 0) ?: 0
        val cols = tof?.optInt("cols", 0) ?: 0
        val array = tof?.optJSONArray("cells")
        require(array == null || array.length() <= 64) { "Unexpected ToF cell count" }
        val cells = (0 until (array?.length() ?: 0)).map { index ->
            val cell = requireNotNull(array).getJSONObject(index)
            DemoTofCell(cell.getInt("zone"), cell.getDouble("raw_mm"),
                cell.getInt("status"), cell.getInt("targets"), cell.getString("quality"))
        }
        val nextUrl = camera?.optString("url")
        if (camera != null && fresh(camera) && nextUrl != null && nextUrl.startsWith("/api/frame?")) {
            if (imageUrl != nextUrl || image == null) {
                val jpeg = fetch(nextUrl)
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, bounds)
                require(bounds.outWidth in 1..4096 && bounds.outHeight in 1..4096) { "Invalid camera image" }
                image = requireNotNull(BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size))
                imageUrl = nextUrl
            }
        } else { image = null; imageUrl = null }
        val done = clockMs()
        val prompt = done - requested <= FRESH_MS
        val freshPair = camera != null && tof != null && fresh(camera) && fresh(tof) && image != null &&
            camera.optDouble("age_ms", Double.POSITIVE_INFINITY) + done - requested <= FRESH_MS &&
            tof.optDouble("age_ms", Double.POSITIVE_INFINITY) + done - requested <= FRESH_MS && prompt
        val live = mode == "live" && state.optJSONObject("recording")?.optBoolean("active", false) == true
        val progressing = done - cameraChanged <= FRESH_MS && done - tofChanged <= FRESH_MS
        val usable = live && freshPair && progressing
        val replay = mode == "replay"
        val decision = TofCorridorDemo.evaluate(rows, cols, cells, usable || (replay && freshPair))
        return HardwareDemoSnapshot(mode, runId, nextCamera, nextTof,
            if (freshPair) image else null, cells, rows, cols, decision, usable,
            when {
                replay -> "实物记录回放 · 静音 · 非实时"
                usable -> "实时 · 电脑 USB 中转 · 接收时间配对"
                else -> "UNKNOWN · 采集停止、输入缺失或数据过期"
            }, cameraUsable = usable || (replay && freshPair), tofUsable = usable || (replay && freshPair))
    }

    private fun fresh(value: JSONObject): Boolean {
        val age = value.optDouble("age_ms", Double.NaN)
        return value.has("stale") && !value.getBoolean("stale") && age.isFinite() && age in 0.0..FRESH_MS.toDouble()
    }

    private companion object { const val FRESH_MS = 1500L }
}

private fun readDemoBytes(path: String): ByteArray {
    require(path.startsWith("/api/state?") || path.startsWith("/api/frame?"))
    val connection = URL("http://127.0.0.1:8766$path").openConnection() as HttpURLConnection
    return try {
        connection.connectTimeout = 1000
        connection.readTimeout = 1000
        connection.instanceFollowRedirects = false
        connection.useCaches = false
        check(connection.responseCode == 200) { "USB 中转未连接：HTTP ${connection.responseCode}" }
        connection.inputStream.use { stream ->
            val output = ByteArrayOutputStream()
            val buffer = ByteArray(8192)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                require(output.size() + count <= 2_000_000) { "Bridge response too large" }
                output.write(buffer, 0, count)
            }
            output.toByteArray()
        }
    } finally { connection.disconnect() }
}
