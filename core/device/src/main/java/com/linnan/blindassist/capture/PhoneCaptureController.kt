package com.linnan.blindassist.capture

import android.content.Context
import android.os.SystemClock
import androidx.lifecycle.LifecycleOwner
import org.json.JSONObject

data class PhoneCaptureState(
    val cameraReady: Boolean = false,
    val tofReady: Boolean = false,
    val connected: Boolean = false,
    val recording: Boolean = false,
    val rgbCount: Long = 0,
    val tofCount: Long = 0,
    val elapsedMs: Long = 0,
    val message: String = "连接 ToF 后开始采集",
    val sessionId: String? = null,
    val imuEnabled: Boolean = true,
    val imuReady: Boolean = false,
    val imuConnected: Boolean = false,
    val imuCount: Long = 0,
    val imuMessage: String = "连接 Atom 后读取 IMU",
    val cameraFrame: AtomCameraCaptureFrame? = null,
    val tofFrame: CnhCaptureFrame? = null,
    val imuFrame: ImuCaptureFrame? = null,
    val cameraAgeMs: Long? = null,
    val tofAgeMs: Long? = null,
    val imuAgeMs: Long? = null,
    val cameraConnected: Boolean = false,
    val cameraMessage: String = "连接 Atom 后读取画面",
    val displayRotationDegrees: Int = 90,
    val displayMirrorHorizontal: Boolean = true,
)

/** Owns only the foreground acquisition page; saved buffers never enter the assist algorithm. */
class PhoneCaptureController(
    context: Context,
    @Suppress("UNUSED_PARAMETER") lifecycleOwner: LifecycleOwner? = null,
    private val monotonicClock: () -> Long = SystemClock::elapsedRealtimeNanos,
) {
    private val context = context.applicationContext
    private val lock = Any()
    private var cameraSource: AtomCameraCaptureSource? = null
    private var cameraGeneration = 0L
    @Volatile private var cameraConnected = false
    private var cameraMessage = "连接 Atom 后读取画面"
    @Volatile private var latestCamera: AtomCameraCaptureFrame? = null
    private var latestTof: CnhCaptureFrame? = null
    private var ending = false
    private var source: CnhCaptureSource? = null
    private var sourceGeneration = 0L
    private var imuSource: ImuCaptureSource? = null
    private var imuGeneration = 0L
    private var imuEnabled = true
    private var imuConnected = false
    private var imuMessage = "连接 Atom 后读取 IMU"
    private var imuSeenNs = 0L
    private var latestImu: ImuCaptureRecord? = null
    private var latestImuIndex: Long? = null
    private val imuRecent = ArrayDeque<Pair<ImuCaptureRecord, Long?>>()
    private val imuPreamble = ArrayDeque<ImuCaptureRecord>()
    private var store: PhoneCaptureStore? = null
    private val preamble = ArrayDeque<CnhCaptureRecord>()
    private var cameraSeenNs = 0L
    private var tofSeenNs = 0L
    private var startedNs = 0L
    private var connected = false
    private var message = "连接 ToF 后开始采集"
    private var lastSummary: PhoneCaptureSummary? = null
    // First failure wins: closing the other sources emits STOPPED callbacks too.
    private var terminalCause: String? = null
    private val stalledStreams = mutableMapOf<String, Long>()
    private var stallWarning: String? = null

    fun connectCamera(next: AtomCameraCaptureSource) {
        whenConnectable(next) { }
        disconnectCamera()
        val generation = whenConnectable(next) {
            terminalCause = null
            cameraSource = next
            ++cameraGeneration
        }
        next.start({ frame ->
            var failed = false
            synchronized(lock) {
                if (generation != cameraGeneration) return@start
                cameraSeenNs = frame.receivedElapsedNs
                latestCamera = frame
                cameraMessage = "Atom 画面已就绪"
                try {
                    store?.recordRgb(frame.jpeg,
                        frame.deviceCaptureTimestampUs?.takeIf { it <= Long.MAX_VALUE / 1000L }?.times(1000L),
                        frame.receivedElapsedNs, frame.width, frame.height, 90,
                        "ATOM_DEVICE_MONOTONIC_UNMAPPED", frame.sequence, frame.sourceId,
                        sourceHeaders = frame.headers, displayMirrorHorizontal = true)
                    recordRecovery("camera", "相机", frame.receivedElapsedNs)
                } catch (error: Exception) {
                    message = "Atom RGB 保存失败：${error.message}"
                    if (terminalCause == null) terminalCause = message
                    failed = true
                }
            }
            if (failed) endRecording("rgb_write_error", "INTERRUPTED")
        }, { status ->
            var interrupted = false
            synchronized(lock) {
                if (generation != cameraGeneration) return@start
                cameraConnected = status.state == CnhConnectionState.CONNECTED
                cameraMessage = status.message
                runCatching { store?.recordEvent("camera_${status.state.name.lowercase()}", status.message, nowNs()) }
                if (status.state in setOf(CnhConnectionState.ERROR, CnhConnectionState.DISCONNECTED)) {
                    cameraSeenNs = 0L
                    message = "相机连接中断：${status.message}"
                    if (store != null && terminalCause == null) terminalCause = message
                    interrupted = store != null
                }
            }
            if (interrupted) endRecording("camera_disconnected", "INTERRUPTED")
        })
    }

    fun connect(next: CnhCaptureSource) {
        whenConnectable(next) { }
        disconnectTof()
        val generation = whenConnectable(next) {
            terminalCause = null
            source = next
            sourceGeneration += 1
            preamble.clear()
            sourceGeneration
        }
        next.start({ record ->
            var failed = false
            synchronized(lock) {
                if (generation != sourceGeneration) return@start
                if (record.frame != null) {
                    tofSeenNs = record.hostReceivedMonotonicNs
                    latestTof = record.frame
                    if (store == null) message = "ToF 数据已就绪"
                } else if (record.issues.isNotEmpty()) {
                    message = "ToF 数据未通过校验：${record.issues.joinToString("; ")}"
                } else if (record.kind == "error") {
                    val error = JSONObject(record.rawLine.toString(Charsets.UTF_8))
                    message = "ToF 固件异常：${error.optString("message").take(128)} " +
                        error.optString("op").take(64)
                }
                if (record.frame == null && record.issues.isEmpty() && record.kind != null) {
                    if (preamble.size == 16) preamble.removeFirst()
                    preamble.addLast(record)
                }
                val active = store ?: return@start
                try {
                    saveTof(active, record)
                    if (record.frame != null) recordRecovery("tof", "ToF", record.hostReceivedMonotonicNs)
                } catch (error: Exception) {
                    message = "ToF 保存失败：${error.message}"
                    if (terminalCause == null) terminalCause = message
                    failed = true
                }
            }
            if (failed) endRecording("tof_write_error", "INTERRUPTED")
        }, { status ->
            var interrupted = false
            synchronized(lock) {
                if (generation != sourceGeneration) return@start
                connected = status.state == CnhConnectionState.CONNECTED
                message = status.message
                try {
                    store?.recordEvent("tof_${status.state.name.lowercase()}", status.message, nowNs())
                } catch (error: Exception) { message = "状态记录失败：${error.message}" }
                if (status.state in setOf(CnhConnectionState.ERROR, CnhConnectionState.DISCONNECTED)) {
                    tofSeenNs = 0L
                    if (store != null && terminalCause == null) terminalCause = "ToF 连接中断：${status.message}"
                    interrupted = store != null
                }
            }
            if (interrupted) endRecording("tof_disconnected", "INTERRUPTED")
        })
    }

    fun disconnect() {
        endRecording("disconnect", "INTERRUPTED")
    }

    fun setImuEnabled(enabled: Boolean) {
        synchronized(lock) { check(store == null); imuEnabled = enabled }
        if (!enabled) disconnectImu()
    }

    fun connectImu(next: ImuCaptureSource) {
        whenConnectable(next) { }
        disconnectImu()
        val generation = whenConnectable(next) {
            terminalCause = null
            imuSource = next
            imuGeneration += 1
            imuPreamble.clear()
            imuGeneration
        }
        next.start({ record ->
            var failed = false
            synchronized(lock) {
                if (generation != imuGeneration) return@start
                if (record.frame != null) {
                    imuSeenNs = record.hostReceivedMonotonicNs
                    imuMessage = "IMU 数据已就绪"
                    latestImu = record
                    if (imuRecent.size == 256) imuRecent.removeFirst()
                    imuRecent.addLast(record to null)
                } else if (record.issues.isNotEmpty()) {
                    imuMessage = "IMU 数据不可用：${record.issues.joinToString("; ")}"
                }
                if (record.frame == null && record.issues.isEmpty() && record.kind != null) {
                    if (imuPreamble.size == 8) imuPreamble.removeFirst()
                    imuPreamble.addLast(record)
                }
                val active = store ?: return@start
                try {
                    val index = saveImu(active, record)
                    if (record.frame != null) {
                        recordRecovery("imu", "IMU", record.hostReceivedMonotonicNs)
                        latestImuIndex = index
                        imuRecent.removeLast()
                        imuRecent.addLast(record to index)
                    }
                } catch (error: Exception) {
                    message = "IMU 保存失败：${error.message}"
                    if (terminalCause == null) terminalCause = message
                    failed = true
                }
            }
            if (failed) endRecording("imu_write_error", "INTERRUPTED")
        }, { status ->
            var interrupted = false
            synchronized(lock) {
                if (generation != imuGeneration) return@start
                imuConnected = status.state == CnhConnectionState.CONNECTED
                imuMessage = status.message
                if (status.state in setOf(CnhConnectionState.ERROR, CnhConnectionState.DISCONNECTED)) {
                    message = "IMU ${status.message}"
                    imuSeenNs = 0L
                    if (store != null && terminalCause == null) terminalCause = "IMU 连接中断：${status.message}"
                    interrupted = store != null
                }
                runCatching { store?.recordEvent("imu_${status.state.name.lowercase()}", status.message, nowNs()) }
            }
            if (interrupted) endRecording("imu_disconnected", "INTERRUPTED")
        })
    }

    /** Reject a reconnect during finalization rather than waiting on source callbacks. */
    private fun <T> whenConnectable(next: AutoCloseable, assign: () -> T): T {
        try {
            return synchronized(lock) {
                check(!ending) { "正在保存，请稍后连接" }
                check(store == null) { "请先停止采集再连接" }
                assign()
            }
        } catch (error: Exception) {
            // A rejected new source has never been started; release its executor too.
            runCatching { next.close() }.onFailure { error.addSuppressed(it) }
            throw error
        }
    }

    private fun disconnectImu() {
        val old = synchronized(lock) {
            imuConnected = false
            imuSeenNs = 0L
            imuSource.also { imuSource = null }
        }
        old?.close()
        synchronized(lock) {
            imuGeneration++
            latestImu = null
            latestImuIndex = null
            imuRecent.clear()
        }
    }

    private fun disconnectCamera() {
        val old = synchronized(lock) {
            cameraConnected = false
            cameraSeenNs = 0L
            cameraSource.also { cameraSource = null }
        }
        old?.close()
        synchronized(lock) { cameraGeneration++; latestCamera = null }
    }

    private fun disconnectTof() {
        val old = synchronized(lock) {
            connected = false
            tofSeenNs = 0L
            source.also { source = null }
        }
        old?.close()
        synchronized(lock) { sourceGeneration++; latestTof = null }
    }

    private fun endRecording(reason: String, status: String) {
        if (!synchronized(lock) {
            if (ending) false else {
                ending = true
                if (store != null && status == "INTERRUPTED" && terminalCause == null) {
                    terminalCause = when (reason) {
                        "page_backgrounded" -> "采集页面进入后台，已停止采集"
                        "disconnect" -> "已断开采集连接"
                        else -> message
                    }
                }
                true
            }
        }) return
        try {
            disconnectCamera()
            disconnectTof()
            disconnectImu()
            synchronized(lock) { finishLocked(reason, status) }
        } finally {
            synchronized(lock) { ending = false }
        }
    }

    fun startRecording(sceneNote: String) = synchronized(lock) {
        val now = nowNs()
        check(!ending && cameraConnected && fresh(cameraSeenNs, now) && fresh(tofSeenNs, now) && connected) {
            "请先连接 Atom 相机与 ToF，等待两路数据更新"
        }
        check(!imuEnabled || imuConnected && fresh(imuSeenNs, now)) { "请连接 Atom IMU，或关闭同时记录 IMU" }
        check(store == null) { "正在采集" }
        val next = PhoneCaptureStore.create(context, sceneNote, mapOf(
            "rgb_source" to "atom_mjpeg_camera",
            "rgb_transport" to (cameraSource?.transport ?: "UNKNOWN"),
            "tof_transport" to (source?.transport ?: "UNKNOWN"),
            "imu_enabled" to imuEnabled.toString(),
            "imu_transport" to (imuSource?.transport ?: "DISABLED"),
            "imu_pose_frame" to "BMI270 native axes; NOT_TOF_OPTICAL; mounting calibration required",
            "imu_alignment" to "closest causal Android receipt; signed imu_minus_tof_ns; max 100ms; not exposure synchronization",
            "rgb_encoding" to "Original Atom JPEG bytes; no recompression or pixel transform",
            "rgb_clock" to "Atom reported frame timestamp; unmapped; exposure timing unverified",
            "rgb_display" to "90 degrees clockwise; horizontal mirror; original JPEG saved",
            "tof_clock" to "device millis at read/extract completion; unmapped",
            "synchronization" to "independent capture clocks; Android receipt timestamps only",
            "geometry" to "Atom RGB / external ToF extrinsics NOT_CALIBRATED",
            "labels" to "user note only; no ground truth"
        ))
        store = next
        startedNs = now
        lastSummary = null
        terminalCause = null
        stalledStreams.clear()
        stallWarning = null
        latestImuIndex = null
        imuRecent.clear()
        try {
            if (imuEnabled) {
                for (record in imuPreamble) saveImu(next, record)
                latestImu?.let {
                    latestImuIndex = saveImu(next, it)
                    imuRecent.addLast(it to latestImuIndex)
                }
            }
            for (record in preamble) saveTof(next, record)
            next.recordEvent("start", "User started foreground Atom capture", now)
            message = "正在采集；保持此页面打开"
        } catch (error: Exception) {
            message = "创建采集失败：${error.message}"
            terminalCause = message
            finishLocked("start_write_error", "INTERRUPTED")
            throw error
        }
    }

    fun stopRecording(): PhoneCaptureSummary? {
        val status = synchronized(lock) {
            val active = store
            if (active != null && active.rgbCount > 0 && active.tofCount > 0 &&
                (!imuEnabled || active.imuCount > 0)) "COMPLETED" else "INTERRUPTED"
        }
        endRecording("user_stop", status)
        return synchronized(lock) {
            if (lastSummary != null) message = "已保存，可以导出 ZIP 或开始下一段"
            lastSummary
        }
    }

    fun snapshot(): PhoneCaptureState {
        val timeout = synchronized(lock) {
            val active = store
            if (active == null || ending) null else {
                val now = nowNs()
                val streams = buildList {
                    add(Triple("camera", "相机", cameraSeenNs))
                    add(Triple("tof", "ToF", tofSeenNs))
                    if (imuEnabled) add(Triple("imu", "IMU", imuSeenNs))
                }
                try {
                    streams.forEach { (key, label, seen) ->
                        val age = ageMs(seen, now)
                        if (!fresh(seen, now) && key !in stalledStreams) {
                            active.recordEvent("${key}_stalled", "$label 暂无新数据；age_ms=$age", now)
                            stalledStreams[key] = seen
                        }
                    }
                    val delayed = streams.filter { it.first in stalledStreams }
                    stallWarning = delayed.takeIf { it.isNotEmpty() }?.joinToString("；") { (_, label, seen) ->
                        "$label 暂无新数据（${ageMs(seen, now)} 毫秒）"
                    }?.plus("；继续等待，缺失区间会保留记录")
                    val expired = streams.filter { (_, _, seen) ->
                        seen <= 0L || now - seen > RECORDING_STALL_NS
                    }
                    if (expired.isEmpty()) null else {
                        val details = expired.joinToString("；") { (_, label, seen) ->
                            "$label 超过 8 秒无新数据（${ageMs(seen, now)} 毫秒）"
                        }
                        if (terminalCause == null) terminalCause = details
                        active.recordEvent("input_timeout", details, now)
                        "input_stale_" + expired.joinToString("_") { it.first }
                    }
                } catch (error: Exception) {
                    val writeFailure = "状态记录失败：${error.message}"
                    terminalCause = terminalCause?.let { "$it；$writeFailure" } ?: writeFailure
                    "status_write_error"
                }
            }
        }
        if (timeout != null) endRecording(timeout, "INTERRUPTED")
        return synchronized(lock) {
            val now = nowNs()
            val active = store
            val cameraReady = cameraConnected && fresh(cameraSeenNs, now)
            val tofReady = connected && fresh(tofSeenNs, now)
            val imuReady = imuEnabled && imuConnected && fresh(imuSeenNs, now)
            PhoneCaptureState(
                cameraReady = cameraReady,
                tofReady = tofReady,
                connected = connected,
                recording = active != null,
                rgbCount = active?.rgbCount ?: lastSummary?.rgbCount ?: 0L,
                tofCount = active?.tofCount ?: lastSummary?.tofCount ?: 0L,
                elapsedMs = if (active != null) (now - startedNs) / 1_000_000L else 0L,
                message = terminalCause?.let { "$it；已有内容已保留，请重新连接后开始新一段" }
                    ?: stallWarning.takeIf { active != null } ?: message,
                sessionId = active?.id ?: lastSummary?.id,
                imuEnabled = imuEnabled,
                imuReady = imuReady,
                imuConnected = imuConnected,
                imuCount = active?.imuCount ?: lastSummary?.imuCount ?: 0L,
                imuMessage = imuMessage,
                cameraFrame = latestCamera.takeIf { cameraReady },
                tofFrame = latestTof.takeIf { tofReady },
                imuFrame = latestImu?.frame.takeIf { imuReady },
                cameraAgeMs = ageMs(cameraSeenNs, now),
                tofAgeMs = ageMs(tofSeenNs, now),
                imuAgeMs = ageMs(imuSeenNs, now),
                cameraConnected = cameraConnected,
                cameraMessage = cameraMessage,
            )
        }
    }

    /** Preview-only lookup: never waits for storage/fsync or performs lifecycle work. */
    fun peekCameraFrame(): AtomCameraCaptureFrame? {
        if (!cameraConnected) return null
        val frame = latestCamera ?: return null
        val now = nowNs()
        return frame.takeIf { cameraConnected && fresh(it.receivedElapsedNs, now) }
    }

    fun stopForeground() = endRecording("page_backgrounded", "INTERRUPTED")

    fun shutdown() = stopForeground()

    private fun saveTof(active: PhoneCaptureStore, record: CnhCaptureRecord) {
        val match = imuRecent.lastOrNull {
            it.second != null && it.first.hostReceivedMonotonicNs <= record.hostReceivedMonotonicNs
        }
        val imu = match?.first
        val residual = imu?.hostReceivedMonotonicNs?.minus(record.hostReceivedMonotonicNs)
        val aligned = record.frame != null && imuEnabled && imu?.frame != null && match?.second != null &&
            residual != null && residual in -100_000_000L..0L
        val alignment = mapOf<String, Any?>(
            "status" to if (!imuEnabled) "DISABLED" else if (aligned) "ALIGNED_RECEIPT_CAUSAL" else "UNAVAILABLE",
            "method" to "closest_causal_android_receipt",
            "imu_minus_tof_ns" to if (aligned) residual else null,
            "imu_record_index" to if (aligned) match?.second else null,
            "raw_reference" to if (aligned) mapOf("index_file" to "imu.index.jsonl",
                "record_index" to match?.second, "raw_file" to "imu.raw.jsonl") else null,
            "imu_sequence" to if (aligned) imu?.frame?.sequence else null,
            "imu_boot_id" to if (aligned) imu?.frame?.bootId else null,
            "imu_device_us" to if (aligned) imu?.frame?.deviceUs else null,
            "imu_pitch_deg" to if (aligned) imu?.frame?.pitchDeg else null,
            "imu_roll_deg" to if (aligned) imu?.frame?.rollDeg else null,
            "pose_frame" to "BMI270_NATIVE_NOT_TOF_OPTICAL"
        )
        active.recordTof(record.rawLine, record.hostReceivedMonotonicNs,
            record.frame?.sequence, record.frame?.deviceMs, record.frame != null, alignment)
        if (record.issues.isNotEmpty()) active.recordEvent("tof_record_issue",
            record.issues.joinToString("; "), record.hostReceivedMonotonicNs)
    }

    private fun saveImu(active: PhoneCaptureStore, record: ImuCaptureRecord): Long {
        val index = active.recordImu(record.rawLine, record.hostReceivedMonotonicNs,
            record.frame?.sequence, record.frame?.deviceUs, record.frame != null, record.frame)
        if (record.issues.isNotEmpty()) active.recordEvent("imu_record_issue",
            record.issues.joinToString("; "), record.hostReceivedMonotonicNs)
        return index
    }

    /** Called under lock on the first saved valid frame after a reported stall. */
    private fun recordRecovery(key: String, label: String, receiptNs: Long) {
        val previous = stalledStreams[key] ?: return
        val active = store ?: return
        active.recordEvent("${key}_recovered",
            "$label 数据恢复；receipt_gap_ms=${(receiptNs - previous) / 1_000_000L}；缺失区间未补采", receiptNs)
        stalledStreams.remove(key)
    }

    private fun finishLocked(reason: String, status: String) {
        val active = store ?: return
        store = null
        try { lastSummary = active.finish(reason, status) }
        catch (error: Exception) {
            message = "保存中断，目录已保留：${error.message}"
            terminalCause = terminalCause?.let { "$it；$message" } ?: message
        }
    }

    private fun nowNs() = monotonicClock()
    private companion object {
        const val RECORDING_STALL_NS = 8_000_000_000L
    }
    private fun fresh(time: Long, now: Long) = time > 0L && now - time in 0L..2_000_000_000L
    private fun ageMs(time: Long, now: Long): Long? = if (time > 0L && now >= time) (now - time) / 1_000_000L else null
}
