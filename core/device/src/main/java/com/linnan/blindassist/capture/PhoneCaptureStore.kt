package com.linnan.blindassist.capture

import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.FilterOutputStream
import java.io.IOException
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

data class PhoneCaptureSummary(
    val id: String,
    val directory: File,
    val createdAtMs: Long,
    val status: String,
    val rgbCount: Long,
    val tofCount: Long,
    val bytes: Long,
    val tofRecordCount: Long = tofCount,
    val sceneNote: String = "",
    val imuCount: Long = 0,
    val imuRecordCount: Long = 0,
) {
    /** A process death leaves RECORDING on disk; it does not erase the session. */
    val isInterrupted: Boolean get() = status == "RECORDING" || status == "RECOVERABLE"
}

/**
 * App-private, incremental RGB + raw ToF recording. Call on a background I/O worker.
 * A row is acknowledged only after its payload and index have reached the filesystem.
 * The two streams share receipt timestamps, not a fabricated hardware synchronization.
 * Saved captures are never automatically evicted, uploaded, or annotated as ground truth.
 */
class PhoneCaptureStore private constructor(
    val directory: File,
    private val sceneNote: String,
    private val sourceMetadata: Map<String, String>,
    private val createdAtMs: Long,
    private val startedElapsedNs: Long?,
) : AutoCloseable {
    val id: String get() = directory.name
    @Volatile var rgbCount: Long = 0
        private set
    @Volatile var tofCount: Long = 0
        private set
    @Volatile var tofRecordCount: Long = 0
        private set
    @Volatile var imuCount: Long = 0
        private set
    @Volatile var imuRecordCount: Long = 0
        private set
    private var eventCount = 0L
    private var finalStatus: String? = null
    private var finishReason: String? = null
    private var finishedAtMs: Long? = null
    private var failureMessage: String? = null
    // Keep only the six append-only stream descriptors open. Each acknowledged row
    // still syncs its payload and index; reuse removes open/close work at IMU rate.
    private val appendStreams = mutableMapOf<File, FileOutputStream>()
    private val appendOffsets = mutableMapOf<File, Long>()
    private val rgbDigest = MessageDigest.getInstance("SHA-256")

    @Synchronized
    fun recordRgb(
        jpeg: ByteArray,
        cameraTimestampNs: Long?,
        receivedElapsedNs: Long,
        width: Int,
        height: Int,
        rotationDegrees: Int,
        cameraClockDomain: String = "CAMERA_HARDWARE_UNMAPPED",
        sourceFrameId: Long? = null,
        sourceId: String? = null,
        intrinsics: Map<String, Double> = emptyMap(),
        sourceHeaders: Map<String, String> = emptyMap(),
        displayMirrorHorizontal: Boolean = false,
    ): String {
        ensureRecording()
        require(jpeg.isNotEmpty()) { "RGB JPEG 不能为空" }
        require(width > 0 && height > 0) { "RGB 尺寸无效" }
        require((cameraTimestampNs == null || cameraTimestampNs >= 0) && receivedElapsedNs >= 0) { "RGB 时间戳无效" }
        require(rotationDegrees in listOf(0, 90, 180, 270)) { "RGB 旋转角度无效" }
        require(intrinsics.values.all { it.isFinite() }) { "相机内参包含无效数值" }
        val relativePath = "rgb/frame-${(rgbCount + 1).toString().padStart(6, '0')}.jpg"
        val row = JSONObject().apply {
            put("index", rgbCount + 1)
            put("file", relativePath)
            put("bytes", jpeg.size)
            put("sha256", hex(rgbDigest.digest(jpeg)))
            put("camera_timestamp_ns", cameraTimestampNs ?: JSONObject.NULL)
            put("camera_clock_domain", cameraClockDomain)
            put("received_elapsed_ns", receivedElapsedNs)
            put("width", width)
            put("height", height)
            put("rotation_degrees", rotationDegrees)
            put("source_frame_id", sourceFrameId ?: JSONObject.NULL)
            put("source_id", sourceId ?: JSONObject.NULL)
            put("camera_intrinsics", JSONObject(intrinsics))
            put("source_headers", JSONObject(sourceHeaders))
            put("boot_id", sourceHeaders["x-sequence-id"] ?: JSONObject.NULL)
            put("device_capture_timestamp_us", sourceHeaders["x-capture-timestamp-us"]
                ?.toLongOrNull()?.takeIf { it >= 0 } ?: JSONObject.NULL)
            put("device_jpeg_ready_timestamp_us", sourceHeaders["x-jpeg-ready-timestamp-us"]
                ?.toLongOrNull()?.takeIf { it >= 0 } ?: JSONObject.NULL)
            put("display_mirror_horizontal", displayMirrorHorizontal)
            put("camera_timestamp_semantics", if (sourceHeaders.isNotEmpty())
                "device_reported_camera_frame_timestamp; exposure_timing_unverified; unmapped"
                else "source_capture_timestamp_in_declared_clock_domain")
        }
        return writeRecord {
            writeAtomic(File(directory, relativePath), jpeg, replaceExisting = false)
            appendJson(File(directory, "rgb.jsonl"), row)
            rgbCount++
            relativePath
        }
    }

    /** rawLine retains the received text, including an existing CR/LF terminator. */
    @Synchronized
    fun recordTof(
        rawLine: String,
        receivedElapsedNs: Long,
        sequence: Long? = null,
        deviceTimestampMs: Long? = null,
        validFrame: Boolean = true,
        imuAlignment: Map<String, Any?> = emptyMap(),
    ): Long {
        require(rawLine.isNotEmpty()) { "ToF 原始数据不能为空" }
        return recordTof(
            (if (rawLine.endsWith('\n')) rawLine else "$rawLine\n").toByteArray(Charsets.UTF_8),
            receivedElapsedNs, sequence, deviceTimestampMs, validFrame, imuAlignment,
        )
    }

    /** Exact transport bytes, including malformed text, CR/LF, and incomplete line chunks. */
    @Synchronized
    fun recordTof(
        rawLine: ByteArray,
        receivedElapsedNs: Long,
        sequence: Long? = null,
        deviceTimestampMs: Long? = null,
        validFrame: Boolean = true,
        imuAlignment: Map<String, Any?> = emptyMap(),
    ): Long {
        ensureRecording()
        require(rawLine.isNotEmpty()) { "ToF 原始数据不能为空" }
        require(receivedElapsedNs >= 0) { "ToF 接收时间戳无效" }
        val rawFile = File(directory, "tof.raw.jsonl")
        val row = JSONObject().apply {
            put("index", tofRecordCount + 1)
            put("valid_frame", validFrame)
            put("received_elapsed_ns", receivedElapsedNs)
            put("sequence", sequence ?: JSONObject.NULL)
            put("device_timestamp_ms", deviceTimestampMs ?: JSONObject.NULL)
            put("raw_byte_offset", appendPosition(rawFile))
            put("raw_byte_length", rawLine.size)
            put("imu_alignment", JSONObject().apply {
                put("status", "UNAVAILABLE")
                put("rule", "closest_causal_android_receipt_within_100ms")
                put("receipt_clock", "android_elapsed_realtime_nanos")
                put("device_clock_mapping", "UNMAPPED")
                put("pose_frame", "BMI270_NATIVE_NOT_TOF_OPTICAL")
                put("raw_reference", JSONObject.NULL)
                put("imu_minus_tof_ns", JSONObject.NULL)
                imuAlignment.forEach { (key, value) -> put(key, JSONObject.wrap(value)) }
            })
        }
        return writeRecord {
            appendBytes(rawFile, rawLine)
            appendJson(File(directory, "tof.index.jsonl"), row)
            if (validFrame) tofCount++
            ++tofRecordCount
        }
    }

    /** IMU bytes and indices remain independent of RGB/ToF exposure/device clocks. */
    @Synchronized
    fun recordImu(
        rawLine: ByteArray,
        receivedElapsedNs: Long,
        sequence: Long? = null,
        deviceTimestampUs: Long? = null,
        validFrame: Boolean = true,
        frame: ImuCaptureFrame? = null,
    ): Long {
        ensureRecording()
        require(rawLine.isNotEmpty()) { "IMU 原始数据不能为空" }
        require(receivedElapsedNs >= 0) { "IMU 接收时间戳无效" }
        val sample = if (validFrame) frame ?: ImuCaptureDecoder().feed(rawLine, receivedElapsedNs)
            .singleOrNull()?.frame else null
        val indexedValid = validFrame && sample != null
        if (sample != null) {
            require(sequence == null || sequence == sample.sequence) { "IMU 序号与解析帧不一致" }
            require(deviceTimestampUs == null || deviceTimestampUs == sample.deviceUs) { "IMU 设备时间与解析帧不一致" }
        }
        val rawFile = File(directory, "imu.raw.jsonl")
        val row = JSONObject().apply {
            put("index", imuRecordCount + 1)
            put("valid_frame", indexedValid)
            put("received_elapsed_ns", receivedElapsedNs)
            put("sequence", sample?.sequence ?: sequence ?: JSONObject.NULL)
            put("device_timestamp_us", sample?.deviceUs ?: deviceTimestampUs ?: JSONObject.NULL)
            put("device_id", sample?.deviceId ?: JSONObject.NULL)
            put("boot_id", sample?.bootId ?: JSONObject.NULL)
            put("raw_byte_offset", appendPosition(rawFile))
            put("raw_byte_length", rawLine.size)
            put("accel_raw", sample?.accelRaw?.let { org.json.JSONArray(it) } ?: JSONObject.NULL)
            put("gyro_raw", sample?.gyroRaw?.let { org.json.JSONArray(it) } ?: JSONObject.NULL)
            put("accel_g", sample?.accelG?.let { org.json.JSONArray(it) } ?: JSONObject.NULL)
            put("gyro_dps", sample?.gyroDps?.let { org.json.JSONArray(it) } ?: JSONObject.NULL)
            put("gravity_unit", sample?.gravityUnit?.let { org.json.JSONArray(it) } ?: JSONObject.NULL)
            put("pitch_deg", sample?.pitchDeg ?: JSONObject.NULL)
            put("roll_deg", sample?.rollDeg ?: JSONObject.NULL)
            put("accel_correction", sample?.accelCorrection ?: JSONObject.NULL)
            put("dt_us", sample?.dtUs ?: JSONObject.NULL)
            put("read_us", sample?.readUs ?: JSONObject.NULL)
            put("sample_gap", sample?.sampleGap ?: JSONObject.NULL)
            put("device_clock", "atom_device_microseconds_unmapped")
            put("pose_frame", "BMI270_NATIVE_NOT_TOF_OPTICAL")
            put("mount_calibration", "REQUIRED_FOR_TOF_OPTICAL_POSE")
        }
        return writeRecord {
            appendBytes(rawFile, rawLine)
            appendJson(File(directory, "imu.index.jsonl"), row)
            if (indexedValid) imuCount++
            ++imuRecordCount
        }
    }

    @Synchronized
    fun recordEvent(kind: String, message: String, receivedElapsedNs: Long) {
        ensureRecording()
        require(kind.isNotBlank()) { "事件类型不能为空" }
        require(receivedElapsedNs >= 0) { "事件时间戳无效" }
        val row = JSONObject().apply {
            put("index", eventCount + 1)
            put("kind", kind)
            put("message", message)
            put("received_elapsed_ns", receivedElapsedNs)
        }
        writeRecord {
            appendJson(File(directory, "events.jsonl"), row)
            eventCount++
        }
    }

    /** COMPLETED means recording stopped and saved; it makes no completeness/label claim. */
    @Synchronized
    fun finish(reason: String = "user_stop", status: String = "COMPLETED"): PhoneCaptureSummary {
        require(status in FINAL_STATUSES) { "无效采集结束状态" }
        if (finalStatus == null) {
            finalStatus = status
            finishReason = reason
            finishedAtMs = System.currentTimeMillis()
        }
        // Retry writing the same final state after an I/O failure, without reopening recording.
        closeAppendStreams()
        writeManifest()
        return summary(directory)
    }

    @Synchronized
    override fun close() {
        if (finalStatus == null) finish("store_closed", "INTERRUPTED")
    }

    private fun ensureRecording() {
        check(finalStatus == null) { "采集已经结束：${finalStatus ?: "unknown"}" }
    }

    private inline fun <T> writeRecord(block: () -> T): T = try {
        block()
    } catch (error: Exception) {
        finalStatus = "FAILED"
        finishReason = "storage_error"
        finishedAtMs = System.currentTimeMillis()
        failureMessage = error.message ?: error.javaClass.simpleName
        try {
            closeAppendStreams()
        } catch (closeError: Exception) {
            error.addSuppressed(closeError)
        }
        try {
            writeManifest()
        } catch (manifestError: Exception) {
            error.addSuppressed(manifestError)
        }
        throw error
    }

    private fun appendJson(file: File, row: JSONObject) =
        appendBytes(file, (row.toString() + "\n").toByteArray(Charsets.UTF_8))

    private fun appendBytes(file: File, bytes: ByteArray) {
        val offset = appendPosition(file)
        val output = appendStreams.getOrPut(file) { FileOutputStream(file, true) }
        output.write(bytes)
        output.fd.sync()
        appendOffsets[file] = offset + bytes.size
    }

    // This store exclusively appends these files. Retain the position after each
    // synced write rather than stat-ing the raw file for every incoming packet.
    private fun appendPosition(file: File): Long = appendOffsets.getOrPut(file) { file.length() }

    private fun closeAppendStreams() {
        var failure: Exception? = null
        for (output in appendStreams.values) {
            try { output.close() } catch (error: Exception) {
                if (failure == null) failure = error else failure.addSuppressed(error)
            }
        }
        appendStreams.clear()
        appendOffsets.clear()
        failure?.let { throw it }
    }

    private fun writeManifest() {
        val json = JSONObject().apply {
            put("schema", "blindassist-phone-capture-v1")
            put("session_id", id)
            put("status", finalStatus ?: "RECORDING")
            put("created_at_unix_ms", createdAtMs)
            put("started_elapsed_ns", startedElapsedNs ?: JSONObject.NULL)
            put("finished_at_unix_ms", finishedAtMs ?: JSONObject.NULL)
            put("finish_reason", finishReason ?: JSONObject.NULL)
            put("failure_message", failureMessage ?: JSONObject.NULL)
            put("scene_note", sceneNote)
            put("sources", JSONObject(sourceMetadata))
            put("rgb_count", rgbCount)
            put("tof_count", tofCount)
            put("tof_record_count", tofRecordCount)
            put("imu_count", imuCount)
            put("imu_record_count", imuRecordCount)
            put("event_count", eventCount)
            put("receipt_clock", "android_elapsed_realtime_nanos")
            put("device_clock", "external_device_milliseconds_unmapped")
            put("rgb_encoding", "jpeg_unrotated_with_rotation_metadata")
            put("tof_encoding", "exact_received_transport_bytes; use index byte offsets for malformed/incomplete records")
            put("imu_encoding", "exact_received_atom_imu_bytes; six-axis/pitch/roll in independent index")
            put("imu_device_clock", "atom_device_microseconds_unmapped")
            put("imu_pose_frame", "BMI270_NATIVE_NOT_TOF_OPTICAL")
            put("storage", "app_private_no_backup; manual_user_selected_zip_export")
            put("labels", JSONObject.NULL)
            put("synchronization", "independent_streams; receipt timestamps only; no hardware clock mapping")
            put("count_rule", "Acknowledged indexed records. For interrupted sessions, inspect JSONL indexes and preserved payloads.")
            put("boundary", "Unlabeled captured sensor data; scene notes are user descriptions, not ground truth or clearance evidence.")
        }
        writeAtomic(File(directory, "manifest.json"), json.toString(2).toByteArray(Charsets.UTF_8), true)
    }

    companion object {
        private val FINAL_STATUSES = setOf("COMPLETED", "INTERRUPTED", "FAILED")
        private val SESSION_NAME = Regex("capture-[0-9]{8}T[0-9]{9}Z-[0-9a-f]{8}")
        private const val EXPORT_BUFFER_BYTES = 256 * 1024
        private const val EXPORT_PROGRESS_INTERVAL_NS = 100_000_000L

        fun create(
            context: Context,
            sceneNote: String = "",
            sourceMetadata: Map<String, String> = emptyMap(),
        ): PhoneCaptureStore = create(
            root = captureRoot(context),
            sceneNote = sceneNote,
            sourceMetadata = sourceMetadata,
            startedElapsedNs = SystemClock.elapsedRealtimeNanos(),
        )

        /** Filesystem overload lets focused JVM tests cover payload and export integrity. */
        fun create(
            root: File,
            sceneNote: String = "",
            sourceMetadata: Map<String, String> = emptyMap(),
            createdAtMs: Long = System.currentTimeMillis(),
            startedElapsedNs: Long? = null,
        ): PhoneCaptureStore {
            check(root.isDirectory || root.mkdirs()) { "无法创建采集目录：${root.absolutePath}" }
            val timestamp = SimpleDateFormat("yyyyMMdd'T'HHmmssSSS'Z'", Locale.US).apply {
                timeZone = TimeZone.getTimeZone("UTC")
            }.format(Date(createdAtMs))
            val directory = File(root, "capture-$timestamp-${UUID.randomUUID().toString().take(8)}")
            check(directory.mkdir()) { "无法创建采集会话目录" }
            check(File(directory, "rgb").mkdir()) { "无法创建 RGB 目录" }
            val store = PhoneCaptureStore(directory, sceneNote, sourceMetadata.toMap(), createdAtMs, startedElapsedNs)
            try {
                store.writeManifest()
                // Empty files explicitly distinguish zero records from an absent stream.
                listOf("rgb.jsonl", "tof.raw.jsonl", "tof.index.jsonl", "imu.raw.jsonl", "imu.index.jsonl", "events.jsonl").forEach {
                    check(File(directory, it).createNewFile()) { "无法创建采集索引：$it" }
                }
                return store
            } catch (error: Exception) {
                try { store.closeAppendStreams() } catch (closeError: Exception) { error.addSuppressed(closeError) }
                throw error
            }
        }

        fun listSessions(context: Context): List<PhoneCaptureSummary> = listSessions(captureRoot(context))

        fun listSessions(root: File): List<PhoneCaptureSummary> = root.listFiles().orEmpty()
            .filter { it.isDirectory && SESSION_NAME.matches(it.name) }
            .map(::summary)
            .sortedByDescending { it.createdAtMs }

        /** Caller owns output; finishing ZIP leaves the Storage Access Framework stream open. */
        fun exportZip(
            session: PhoneCaptureSummary,
            output: OutputStream,
            onProgress: (writtenSourceBytes: Long, totalSourceBytes: Long) -> Unit = { _, _ -> },
        ) {
            val directory = session.directory.canonicalFile
            require(directory.isDirectory) { "采集会话不存在" }
            val prefix = directory.path + File.separator
            // Reuse this ordered inventory for both sizing and writing; no extra tree walk.
            val files = directory.walkTopDown().filter { file -> file.isFile }
                .sortedBy { file -> file.relativeTo(directory).path }.toList()
            val totalSourceBytes = files.sumOf { it.length() }
            var writtenSourceBytes = 0L
            var lastProgressNs = System.nanoTime()
            onProgress(0L, totalSourceBytes)
            val callerOwned = object : FilterOutputStream(output) {
                // FilterOutputStream's default bulk write loops over write(int).
                // SAF/document providers must receive blocks, never a call per byte.
                override fun write(bytes: ByteArray, offset: Int, length: Int) {
                    output.write(bytes, offset, length)
                }
                override fun close() { flush() }
            }
            val zip = ZipOutputStream(BufferedOutputStream(callerOwned, EXPORT_BUFFER_BYTES))
            val transferBuffer = ByteArray(EXPORT_BUFFER_BYTES)
            zip.use { archive ->
                files.forEach { file ->
                    require(file.canonicalPath.startsWith(prefix)) { "采集文件路径越界" }
                    val name = "${session.id}/${file.relativeTo(directory).invariantSeparatorsPath}"
                    // JPEG is already compressed. Keep it streaming in DEFLATED
                    // format at level 0, avoiding both recompression and a CRC prepass.
                    archive.setLevel(if (file.extension.equals("jpg", ignoreCase = true) ||
                        file.extension.equals("jpeg", ignoreCase = true)) Deflater.NO_COMPRESSION
                        else Deflater.BEST_SPEED)
                    archive.putNextEntry(ZipEntry(name).apply { time = file.lastModified() })
                    file.inputStream().use { input ->
                        while (true) {
                            val count = input.read(transferBuffer)
                            if (count < 0) break
                            archive.write(transferBuffer, 0, count)
                            writtenSourceBytes += count
                            val nowNs = System.nanoTime()
                            if (nowNs - lastProgressNs >= EXPORT_PROGRESS_INTERVAL_NS) {
                                onProgress(writtenSourceBytes, totalSourceBytes)
                                lastProgressNs = nowNs
                            }
                        }
                    }
                    archive.closeEntry()
                }
            }
            // Completion follows the central directory and final buffered provider write.
            onProgress(writtenSourceBytes, totalSourceBytes)
        }

        private fun captureRoot(context: Context): File = File(context.noBackupFilesDir, "phone-capture")

        private fun summary(directory: File): PhoneCaptureSummary {
            val manifest = try {
                JSONObject(File(directory, "manifest.json").readText(Charsets.UTF_8))
            } catch (_: Exception) {
                null // Keep even a damaged/incomplete manifest visible and exportable.
            }
            val status = manifest?.optString("status")?.takeIf { it in FINAL_STATUSES || it == "RECORDING" }
                ?: "RECOVERABLE"
            // A finalized atomic manifest contains counters acknowledged after fd.sync.
            // Reconstruct unfinished sessions from complete index rows after process loss.
            val finalized = manifest?.takeIf { status in FINAL_STATUSES }
            fun records(index: String, countKey: String, validKey: String = countKey): IndexCounts {
                if (finalized != null && finalized.has(countKey) && finalized.has(validKey)) {
                    val rows = runCatching { finalized.getLong(countKey) }.getOrNull()
                    val valid = runCatching { finalized.getLong(validKey) }.getOrNull()
                    if (rows != null && valid != null && rows >= 0 && valid in 0L..rows) {
                        return IndexCounts(rows, valid)
                    }
                }
                return countIndexedRecords(File(directory, index))
            }
            val rgbRecords = records("rgb.jsonl", "rgb_count")
            val tofRecords = records("tof.index.jsonl", "tof_record_count", "tof_count")
            val imuRecords = records("imu.index.jsonl", "imu_record_count", "imu_count")
            return PhoneCaptureSummary(
                id = directory.name,
                directory = directory,
                createdAtMs = manifest?.optLong("created_at_unix_ms", directory.lastModified()) ?: directory.lastModified(),
                status = status,
                rgbCount = rgbRecords.rows,
                tofCount = tofRecords.validFrames,
                bytes = directory.walkTopDown().filter { it.isFile }.sumOf { it.length() },
                tofRecordCount = tofRecords.rows,
                sceneNote = manifest?.optString("scene_note", "") ?: "",
                imuCount = imuRecords.validFrames,
                imuRecordCount = imuRecords.rows,
            )
        }

        private data class IndexCounts(val rows: Long, val validFrames: Long)

        /** A final partial row after process/power loss must not count as an acknowledged row. */
        private fun countIndexedRecords(file: File): IndexCounts {
            if (!file.isFile) return IndexCounts(0, 0)
            return file.inputStream().buffered().use { input ->
                var count = 0L
                var validFrames = 0L
                val line = java.io.ByteArrayOutputStream()
                while (true) {
                    val value = input.read()
                    if (value < 0) break
                    if (value == '\n'.code) {
                        val complete = line.toString(Charsets.UTF_8.name())
                        line.reset()
                        val row = runCatching { JSONObject(complete) }.getOrNull()
                        if (row != null && row.has("index")) {
                            count++
                            if (row.optBoolean("valid_frame", true)) validFrames++
                        }
                    } else {
                        line.write(value)
                    }
                }
                IndexCounts(count, validFrames)
            }
        }

        private fun hex(bytes: ByteArray): String {
            val alphabet = "0123456789abcdef"
            val chars = CharArray(bytes.size * 2)
            bytes.forEachIndexed { index, byte ->
                val value = byte.toInt() and 255
                chars[index * 2] = alphabet[value ushr 4]
                chars[index * 2 + 1] = alphabet[value and 15]
            }
            return String(chars)
        }

        private fun writeAtomic(destination: File, bytes: ByteArray, replaceExisting: Boolean) {
            check(replaceExisting || !destination.exists()) { "拒绝覆盖已采集文件：${destination.name}" }
            val pending = File(destination.parentFile, destination.name + ".pending")
            FileOutputStream(pending, false).use { output ->
                output.write(bytes)
                output.fd.sync()
            }
            // Both files are in the same app-private directory. Atomic replacement also
            // works in host JVM tests on Windows; failure preserves the .pending payload.
            try {
                Files.move(pending.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING)
            } catch (error: IOException) {
                throw IOException("无法完成文件写入：${destination.name}", error)
            }
        }
    }
}
