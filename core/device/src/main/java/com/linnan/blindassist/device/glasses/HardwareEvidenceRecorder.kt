package com.linnan.blindassist.device.glasses

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import com.linnan.blindassist.risk.DemoTofCell
import com.linnan.blindassist.risk.DemoTofDecision
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class HardwareEvidenceFrame(
    val atMs: Long, val runId: String, val cameraSequence: Long?, val tofSequence: Long?,
    val cameraUsable: Boolean, val tofUsable: Boolean, val cameraAgeMs: Long?, val tofAgeMs: Long?,
    val rows: Int, val cols: Int, val cells: List<DemoTofCell>,
    val decision: DemoTofDecision, val status: String, val thumbnail: ByteArray? = null,
)

/** Output requests are logged; queued TTS is not proof of audible speech. */
data class HardwareEvidenceEvent(
    val atMs: Long, val kind: String, val text: String,
    val speechQueued: Boolean, val vibrationRequested: Boolean,
)

data class HardwareEvidenceClip(
    val createdAtMs: Long, val appVersion: String,
    val frames: List<HardwareEvidenceFrame>, val events: List<HardwareEvidenceEvent>,
) {
    val durationMs: Long get() = (frames.lastOrNull()?.atMs ?: 0) - (frames.firstOrNull()?.atMs ?: 0)
    fun frameIndex(positionMs: Long): Int {
        val target = frames.first().atMs + positionMs.coerceIn(0, durationMs)
        return frames.indexOfLast { it.atMs <= target }.coerceAtLeast(0)
    }

    /** Uses saved decisions, never reevaluates history with today's algorithm. */
    fun snapshotAt(positionMs: Long): HardwareDemoSnapshot {
        val index = frameIndex(positionMs)
        val frame = frames[index]
        val imageFrame = if (frame.cameraUsable) (index downTo 0).firstOrNull { candidate ->
            val value = frames[candidate]
            value.runId == frame.runId && value.cameraUsable && value.thumbnail != null &&
                (value.cameraAgeMs ?: 750L) + frame.atMs - value.atMs <= 750 &&
                (candidate..index).all { frames[it].cameraUsable }
        }?.let(frames::get) else null
        val bitmap = imageFrame?.thumbnail?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
        return HardwareDemoSnapshot("replay", frame.runId, frame.cameraSequence, frame.tofSequence,
            bitmap, frame.cells, frame.rows, frame.cols, frame.decision, false,
            "历史记录 · 静音 · ${if (bitmap == null) "此时未记录画面" else "2 Hz 抽帧，非同步"} · ${frame.status}",
            cameraUsable = bitmap != null, tofUsable = frame.tofUsable,
            cameraAgeMs = frame.cameraAgeMs, tofAgeMs = frame.tofAgeMs)
    }
}

data class HardwareEvidenceSummary(val id: String, val createdAtMs: Long, val bytes: Long)

/** Foreground-only, bounded scalar ring. Optional thumbnails run on one worker,
 * with at most one pending image; no compression or file I/O on the polling thread.
 */
class HardwareEvidenceRecorder(private val appVersion: String) : AutoCloseable {
    private val frames = ArrayDeque<HardwareEvidenceFrame>()
    private val events = ArrayDeque<HardwareEvidenceEvent>()
    private val worker = Executors.newSingleThreadExecutor()
    private val imageBusy = AtomicBoolean(false)
    private var lastImageAt = Long.MIN_VALUE
    private var imageGeneration = 0
    private var lastState: Triple<String, Boolean, Boolean>? = null
    private var closed = false
    var captureImages: Boolean = false
        @Synchronized set(value) {
            field = value
            imageGeneration++
            lastImageAt = Long.MIN_VALUE
            if (!value) {
                val scalarOnly = frames.map { it.copy(thumbnail = null) }
                frames.clear(); frames.addAll(scalarOnly)
            }
        }

    @Synchronized fun offer(snapshot: HardwareDemoSnapshot, nowMs: Long) {
        if (closed || snapshot.mode != "live") return
        if (frames.lastOrNull()?.atMs?.let { nowMs < it } == true) clear()
        val state = Triple(snapshot.runId, snapshot.cameraUsable, snapshot.tofUsable)
        val previous = frames.lastOrNull()
        if (previous != null && nowMs - previous.atMs < 100 && state == lastState &&
            snapshot.tofSequence == previous.tofSequence && snapshot.decision == previous.decision) return
        lastState = state
        val frame = HardwareEvidenceFrame(nowMs, snapshot.runId, snapshot.cameraSequence, snapshot.tofSequence,
            snapshot.cameraUsable, snapshot.tofUsable, snapshot.cameraAgeMs, snapshot.tofAgeMs,
            snapshot.rows, snapshot.cols, snapshot.cells.toList(), snapshot.decision.copy(
                triggeringZones = snapshot.decision.triggeringZones.toList()), snapshot.status)
        frames.addLast(frame)
        while (frames.size > 256 || frames.first().atMs < nowMs - 20_000) frames.removeFirst()
        while (events.isNotEmpty() && events.first().atMs < frames.first().atMs) events.removeFirst()
        val bitmap = snapshot.image
        if (captureImages && snapshot.cameraUsable && bitmap != null &&
            (lastImageAt == Long.MIN_VALUE || nowMs - lastImageAt >= 500) && imageBusy.compareAndSet(false, true)) {
            lastImageAt = nowMs
            val generation = imageGeneration
            worker.execute {
                try {
                    val scale = minOf(1f, 320f / bitmap.width, 240f / bitmap.height)
                    val small = Bitmap.createScaledBitmap(bitmap,
                        (bitmap.width * scale).toInt().coerceAtLeast(1),
                        (bitmap.height * scale).toInt().coerceAtLeast(1), true)
                    val bytes = try { ByteArrayOutputStream().use { output ->
                        check(small.compress(Bitmap.CompressFormat.JPEG, 65, output))
                        output.toByteArray().also { require(it.size <= 100_000) }
                    } } finally { if (small !== bitmap) small.recycle() }
                    synchronized(this) {
                        if (!closed && captureImages && generation == imageGeneration) {
                            val updated = frames.map { if (it.atMs == frame.atMs) it.copy(thumbnail = bytes) else it }
                            frames.clear(); frames.addAll(updated)
                        }
                    }
                } catch (_: Exception) {
                    // Missing thumbnail is explicit; scalar evidence remains intact.
                } finally { imageBusy.set(false) }
            }
        }
    }

    @Synchronized fun recordEvent(event: HardwareEvidenceEvent) {
        if (closed || frames.isEmpty()) return
        events.addLast(event)
        while (events.size > 128) events.removeFirst()
    }

    @Synchronized fun freeze(createdAtMs: Long = System.currentTimeMillis()): HardwareEvidenceClip {
        check(frames.isNotEmpty()) { "还没有实时记录，请先连接硬件" }
        return HardwareEvidenceClip(createdAtMs, appVersion, frames.toList(), events.toList())
    }

    @Synchronized fun clear() {
        frames.clear(); events.clear(); lastState = null; imageGeneration++
        lastImageAt = Long.MIN_VALUE
    }

    @Synchronized override fun close() {
        closed = true
        clear()
        worker.shutdownNow()
    }
}

/** Private no-backup storage. Export is an explicit user-selected document write.
 * Never evicts saved evidence automatically; quota failure asks the user to delete.
 */
class HardwareEvidenceStore(private val directory: File) {
    companion object { const val MAX_FILE_BYTES = 8_000_000L }

    fun list(): List<HardwareEvidenceSummary> = directory.listFiles().orEmpty()
        .filter { it.isFile && it.name.matches(Regex("clip-[0-9a-f-]+\\.json")) }
        .sortedByDescending { it.lastModified() }
        .map { HardwareEvidenceSummary(it.name, it.lastModified(), it.length()) }

    private fun file(id: String): File {
        require(id.matches(Regex("clip-[0-9a-f-]+\\.json"))) { "无效记录名称" }
        return File(directory, id)
    }

    fun save(clip: HardwareEvidenceClip): HardwareEvidenceSummary {
        check(directory.isDirectory || directory.mkdirs()) { "无法创建记录目录" }
        val existing = list()
        check(existing.size < 50 && existing.sumOf { it.bytes } < 50_000_000L) { "记录空间已满，请删除不需要的记录" }
        val bytes = encode(clip).toString().toByteArray(Charsets.UTF_8)
        require(bytes.size <= MAX_FILE_BYTES) { "记录过大" }
        check(existing.sumOf { it.bytes } + bytes.size <= 50_000_000L) { "记录空间已满，请删除不需要的记录" }
        val destination = file("clip-${UUID.randomUUID()}.json")
        val pending = File(directory, destination.name + ".pending")
        try {
            pending.outputStream().use { it.write(bytes); it.fd.sync() }
            check(pending.renameTo(destination)) { "记录保存失败" }
        } finally { if (pending.exists()) pending.delete() }
        return HardwareEvidenceSummary(destination.name, destination.lastModified(), destination.length())
    }

    fun readBytes(id: String): ByteArray {
        val source = file(id)
        require(source.length() in 1..MAX_FILE_BYTES) { "记录大小无效" }
        return source.readBytes()
    }
    fun load(id: String): HardwareEvidenceClip = decode(JSONObject(readBytes(id).toString(Charsets.UTF_8)))
    fun delete(id: String) { check(file(id).delete()) { "记录删除失败" } }

    private fun encode(clip: HardwareEvidenceClip) = JSONObject().apply {
        put("schema", "hardware-evidence-v1"); put("created_at_ms", clip.createdAtMs)
        put("app_version", clip.appVersion); put("algorithm", "tof-corridor-demo-v1")
        put("clock", "android_elapsed_realtime_ms"); put("window_ms", 20_000)
        put("boundary", "Saved decisions; independent streams; output requests do not prove audible speech or felt vibration; not ground truth")
        put("frames", JSONArray().apply { clip.frames.forEach { f -> put(JSONObject().apply {
            put("at_ms", f.atMs); put("run_id", f.runId)
            put("camera_seq", f.cameraSequence ?: JSONObject.NULL); put("tof_seq", f.tofSequence ?: JSONObject.NULL)
            put("camera_usable", f.cameraUsable); put("tof_usable", f.tofUsable)
            put("camera_age_ms", f.cameraAgeMs ?: JSONObject.NULL); put("tof_age_ms", f.tofAgeMs ?: JSONObject.NULL)
            put("rows", f.rows); put("cols", f.cols); put("status", f.status)
            put("cells", JSONArray().apply { f.cells.forEach { c -> put(JSONObject().apply {
                put("zone", c.zone); put("mm", if (c.rangeMm.isFinite()) c.rangeMm else JSONObject.NULL)
                put("status", c.status); put("targets", c.targets); put("quality", c.quality)
            }) } })
            put("decision", JSONObject().apply {
                put("alert", f.decision.alert); put("state", f.decision.state)
                put("nearest_mm", f.decision.nearestMm ?: JSONObject.NULL)
                put("triggering", JSONArray(f.decision.triggeringZones)); put("valid_zones", f.decision.validZones)
            })
            put("jpeg", f.thumbnail?.let { Base64.encodeToString(it, Base64.NO_WRAP) } ?: JSONObject.NULL)
        }) } })
        put("events", JSONArray().apply { clip.events.forEach { event -> put(JSONObject().apply {
            put("at_ms", event.atMs); put("kind", event.kind); put("text", event.text)
            put("speech_queued", event.speechQueued); put("vibration_requested", event.vibrationRequested)
        }) } })
    }

    private fun decode(json: JSONObject): HardwareEvidenceClip {
        require(json.getString("schema") == "hardware-evidence-v1") { "不支持的记录版本" }
        val array = json.getJSONArray("frames")
        require(array.length() in 1..256)
        fun JSONObject.nullableLong(key: String) = if (isNull(key)) null else getLong(key)
        val frames = (0 until array.length()).map { i ->
            val f = array.getJSONObject(i)
            val raw = f.getJSONArray("cells")
            require(raw.length() <= 64)
            val cells = (0 until raw.length()).map { j -> raw.getJSONObject(j).let { c ->
                DemoTofCell(c.getInt("zone"), if (c.isNull("mm")) Double.NaN else c.getDouble("mm"),
                    c.getInt("status"), c.getInt("targets"), c.getString("quality"))
            } }
            val d = f.getJSONObject("decision")
            val zones = d.getJSONArray("triggering")
            require(zones.length() <= 64)
            val tofUsable = f.getBoolean("tof_usable")
            val alert = d.getBoolean("alert")
            require(!alert || tofUsable) { "记录包含过期判决" }
            val thumbnail = if (f.isNull("jpeg")) null else Base64.decode(f.getString("jpeg"), Base64.NO_WRAP).also {
                require(it.size <= 100_000)
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeByteArray(it, 0, it.size, bounds)
                require(bounds.outWidth in 1..320 && bounds.outHeight in 1..240)
            }
            HardwareEvidenceFrame(f.getLong("at_ms"), f.getString("run_id"), f.nullableLong("camera_seq"),
                f.nullableLong("tof_seq"), f.getBoolean("camera_usable"), tofUsable,
                f.nullableLong("camera_age_ms"), f.nullableLong("tof_age_ms"), f.getInt("rows"), f.getInt("cols"),
                cells, DemoTofDecision(alert, d.getString("state"), d.nullableLong("nearest_mm")?.toInt(),
                    (0 until zones.length()).map(zones::getInt), d.getInt("valid_zones")), f.getString("status"), thumbnail)
        }
        require(frames.zipWithNext().all { (a, b) -> b.atMs >= a.atMs })
        require(frames.last().atMs - frames.first().atMs in 0..20_000)
        val events = json.getJSONArray("events")
        require(events.length() <= 128)
        return HardwareEvidenceClip(json.getLong("created_at_ms"), json.getString("app_version"), frames,
            (0 until events.length()).map { i -> events.getJSONObject(i).let { e ->
                HardwareEvidenceEvent(e.getLong("at_ms"), e.getString("kind"), e.getString("text"),
                    e.getBoolean("speech_queued"), e.getBoolean("vibration_requested"))
            } })
    }
}
