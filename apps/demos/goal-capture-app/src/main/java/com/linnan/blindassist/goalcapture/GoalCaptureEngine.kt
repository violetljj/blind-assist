package com.linnan.blindassist.goalcapture

import android.content.Context
import android.media.MediaMetadataRetriever
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.security.MessageDigest
import java.time.Instant
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

internal sealed interface RecorderState {
    data object LoadingPlan : RecorderState
    data class Preparing(val message: String) : RecorderState
    data class Ready(val episodeIndex: Int, val episodeCount: Int, val episode: CapturePlanEpisode) : RecorderState
    data class RecordingEpisode(val episodeIndex: Int, val episodeCount: Int, val episode: CapturePlanEpisode) : RecorderState
    data class Finalizing(val episodeIndex: Int, val episodeCount: Int) : RecorderState
    data class Complete(val sessionDirectory: File, val receipt: File) : RecorderState
    data class Hold(val reason: String) : RecorderState
}

internal class GoalCaptureEngine(
    context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val plan: CapturePlan,
    private val sessionRoot: File,
    private val onState: (RecorderState) -> Unit,
) {
    private val appContext = context.applicationContext
    private val mainExecutor = ContextCompat.getMainExecutor(appContext)
    private val worker: ExecutorService = Executors.newSingleThreadExecutor()
    private val terminal = AtomicBoolean(false)
    private var provider: ProcessCameraProvider? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    private var activeFile: File? = null
    private var activeStartedAt: Instant? = null
    private var cancelRequested = false
    private var currentIndex = 0
    private val completed = mutableListOf<CompletedCapture>()

    fun start() {
        if (terminal.get()) return
        emit(RecorderState.Preparing("正在验证采集目录并启动后置相机…"))
        try {
            require(!sessionRoot.exists() || sessionRoot.listFiles().isNullOrEmpty()) { "session directory already contains data" }
            sessionRoot.mkdirs()
            sessionRoot.resolve("device_captures").mkdirs()
            sessionRoot.resolve("capture_plan.json").writeText(plan.originalJson)
        } catch (error: Throwable) {
            finishHold("采集目录初始化失败：${error.message ?: error.javaClass.simpleName}")
            return
        }
        val future = ProcessCameraProvider.getInstance(appContext)
        future.addListener({
            try {
                val cameraProvider = future.get(10, TimeUnit.SECONDS)
                if (terminal.get()) return@addListener
                val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                val recorder = Recorder.Builder()
                    .setQualitySelector(
                        QualitySelector.from(
                            Quality.HD,
                            FallbackStrategy.higherQualityOrLowerThan(Quality.HD),
                        ),
                    )
                    .build()
                val capture = VideoCapture.withOutput(recorder)
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
                provider = cameraProvider
                videoCapture = capture
                emitReady()
            } catch (error: Throwable) {
                finishHold("CameraX 启动失败：${error.message ?: error.javaClass.simpleName}")
            }
        }, mainExecutor)
    }

    fun startEpisode() {
        if (terminal.get() || activeRecording != null || currentIndex !in plan.episodes.indices) return
        val episode = plan.episodes[currentIndex]
        val capture = videoCapture ?: return finishHold("录像 use case 尚未准备")
        val output = sessionRoot.resolve("device_captures").resolve(episode.mediaRelativePath)
        if (output.exists()) return finishHold("冻结媒体文件已存在，拒绝覆盖：${output.name}")
        cancelRequested = false
        activeStartedAt = null
        activeFile = output
        try {
            val pending = capture.output.prepareRecording(appContext, FileOutputOptions.Builder(output).build())
            activeRecording = pending.start(mainExecutor) { event -> onVideoEvent(event) }
        } catch (error: Throwable) {
            activeFile = null
            finishHold("录像启动失败：${error.message ?: error.javaClass.simpleName}")
        }
    }

    fun stopEpisode() {
        activeRecording?.stop()
    }

    fun cancelSession() {
        cancelRequested = true
        val recording = activeRecording
        if (recording != null) recording.stop() else finishHold("用户取消；未完成 roster 不得导出 receipt")
    }

    fun shutdown() {
        cancelRequested = true
        runCatching { activeRecording?.close() }
        provider?.unbindAll()
        worker.shutdownNow()
    }

    private fun onVideoEvent(event: VideoRecordEvent) {
        when (event) {
            is VideoRecordEvent.Start -> {
                activeStartedAt = Instant.now()
                emit(RecorderState.RecordingEpisode(currentIndex, plan.episodes.size, plan.episodes[currentIndex]))
            }
            is VideoRecordEvent.Finalize -> finalizeEpisode(event)
        }
    }

    private fun finalizeEpisode(event: VideoRecordEvent.Finalize) {
        val output = activeFile
        val startedAt = activeStartedAt
        activeRecording = null
        activeFile = null
        activeStartedAt = null
        if (output == null || startedAt == null || cancelRequested || event.hasError()) {
            output?.delete()
            val reason = if (cancelRequested) "用户取消；部分录像已删除" else "录像 finalize 失败：${event.error}"
            finishHold(reason)
            return
        }
        val completedAt = Instant.now()
        emit(RecorderState.Finalizing(currentIndex, plan.episodes.size))
        worker.execute {
            try {
                val metadata = mediaMetadata(output)
                val elapsed = java.time.Duration.between(startedAt, completedAt).toMillis() / 1000.0
                require(kotlin.math.abs(elapsed - metadata.durationSeconds) <= 1.0) { "recorder timeline and media duration disagree" }
                val episode = plan.episodes[currentIndex]
                completed += CompletedCapture(
                    episodeId = episode.episodeId,
                    captureStartedAt = startedAt,
                    captureCompletedAt = completedAt,
                    mediaPath = episode.mediaRelativePath,
                    mediaSha256 = sha256(output),
                    width = metadata.width,
                    height = metadata.height,
                    durationSeconds = metadata.durationSeconds,
                )
                currentIndex += 1
                mainExecutor.execute {
                    if (currentIndex == plan.episodes.size) finishComplete() else emitReady()
                }
            } catch (error: Throwable) {
                output.delete()
                finishHold("媒体校验失败：${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    private fun finishComplete() {
        if (!terminal.compareAndSet(false, true)) return
        try {
            val receiptMap = CaptureReceiptBuilder.build(plan, completed.toList(), Instant.now())
            val receipt = sessionRoot.resolve("physical_capture_receipt.json")
            receipt.writeText(CanonicalJson.encode(receiptMap))
            provider?.unbindAll()
            worker.shutdown()
            onState(RecorderState.Complete(sessionRoot, receipt))
        } catch (error: Throwable) {
            terminal.set(false)
            finishHold("receipt 生成失败：${error.message ?: error.javaClass.simpleName}")
        }
    }

    private fun finishHold(reason: String) {
        if (!terminal.compareAndSet(false, true)) return
        runCatching {
            sessionRoot.mkdirs()
            val hold = linkedMapOf<String, Any?>(
                "schema_version" to "blindassist_p1_pa3_device_capture_hold_v1",
                "status" to "HOLD",
                "reason" to reason,
                "completed_episode_count" to completed.size,
                "evaluation_authorized" to false,
                "provider_model_calls" to 0,
            )
            sessionRoot.resolve("capture_hold.json").writeText(CanonicalJson.encode(hold))
        }
        mainExecutor.execute {
            provider?.unbindAll()
            worker.shutdownNow()
            onState(RecorderState.Hold(reason))
        }
    }

    private fun emitReady() = emit(RecorderState.Ready(currentIndex, plan.episodes.size, plan.episodes[currentIndex]))
    private fun emit(state: RecorderState) = mainExecutor.execute { onState(state) }

    private data class MediaMetadata(val width: Int, val height: Int, val durationSeconds: Double)

    private fun mediaMetadata(file: File): MediaMetadata {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(file.absolutePath)
            val width = requireNotNull(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)).toInt()
            val height = requireNotNull(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)).toInt()
            val duration = requireNotNull(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)).toLong() / 1000.0
            require(duration in 3.0..45.0) { "video duration must be within 3–45 seconds" }
            MediaMetadata(width, height, duration)
        } finally {
            retriever.release()
        }
    }

    private fun sha256(file: File): String = file.inputStream().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        digest.digest().joinToString("") { "%02x".format(it) }
    }
}
