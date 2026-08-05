package com.linnan.blindassist.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.camera.view.PreviewView
import com.linnan.blindassist.model.ReplayScenario
import com.linnan.blindassist.util.FatalThrowables
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.linnan.blindassist.vision.VisionFrame
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

/** Replays one bundled scene through the same frame callback used by CameraX. */
class ReplayFrameSource internal constructor(
    internal val scenario: ReplayScenario,
    private val frameDecoder: () -> RgbaVisionFrame,
    private val scheduler: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor(),
    private val framePeriodMillis: Long = DEFAULT_FRAME_PERIOD_MILLIS
) : FrameSource {
    constructor(
        context: Context,
        scenario: ReplayScenario
    ) : this(
        scenario = scenario,
        frameDecoder = assetFrameDecoder(context.applicationContext, scenario)
    )

    private val lifecycleLock = Any()
    private var scheduledTask: ScheduledFuture<*>? = null
    private var starting = false
    private var started = false
    private var shutdownRequested = false
    private var sessionGeneration = 0L
    private var errorReported = false

    init {
        require(framePeriodMillis > 0L) { "framePeriodMillis must be positive" }
    }

    override fun start(
        previewView: PreviewView?,
        onFrame: (VisionFrame) -> Unit,
        onStarted: () -> Unit,
        onError: (Throwable) -> Unit,
        onPreviewBitmap: ((Bitmap) -> Unit)?
    ) {
        val generation = synchronized(lifecycleLock) {
            if (shutdownRequested || starting || started) return
            starting = true
            errorReported = false
            sessionGeneration += 1L
            sessionGeneration
        }

        try {
            val future = scheduler.scheduleAtFixedRate(
                {
                    emitFrame(generation, onFrame, onStarted, onError)
                },
                0L,
                framePeriodMillis,
                TimeUnit.MILLISECONDS
            )
            synchronized(lifecycleLock) {
                if (isCurrentSessionLocked(generation)) {
                    scheduledTask = future
                } else {
                    future.cancel(false)
                }
            }
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            synchronized(lifecycleLock) {
                if (isCurrentSessionLocked(generation)) starting = false
            }
            reportError(generation, error, onError)
        }
    }

    override fun stop() {
        val task = synchronized(lifecycleLock) {
            sessionGeneration += 1L
            starting = false
            started = false
            errorReported = false
            scheduledTask.also { scheduledTask = null }
        }
        task?.cancel(false)
    }

    override fun shutdown() {
        val shouldShutdown = synchronized(lifecycleLock) {
            if (shutdownRequested) {
                false
            } else {
                shutdownRequested = true
                true
            }
        }
        stop()
        if (shouldShutdown) {
            shutdownExecutor(scheduler)
        }
    }

    private fun emitFrame(
        generation: Long,
        onFrame: (VisionFrame) -> Unit,
        onStarted: () -> Unit,
        onError: (Throwable) -> Unit
    ) {
        if (!isCurrentSession(generation)) return
        val frame = try {
            frameDecoder()
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            reportError(generation, error, onError)
            return
        }

        val shouldNotifyStarted = synchronized(lifecycleLock) {
            if (!isCurrentSessionLocked(generation)) {
                frame.close()
                return
            }
            if (!started) {
                started = true
                starting = false
                true
            } else {
                false
            }
        }

        try {
            if (shouldNotifyStarted) onStarted()
            if (!isCurrentSession(generation)) {
                frame.close()
                return
            }
            onFrame(frame)
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            frame.close()
            reportError(generation, error, onError)
        }
    }

    private fun reportError(generation: Long, error: Throwable, onError: (Throwable) -> Unit) {
        val shouldReport = synchronized(lifecycleLock) {
            if (isCurrentSessionLocked(generation) && !errorReported) {
                errorReported = true
                true
            } else {
                false
            }
        }
        if (shouldReport) {
            try {
                onError(error)
            } catch (callbackError: Throwable) {
                FatalThrowables.rethrowIfFatal(callbackError)
            }
        }
    }

    private fun isCurrentSession(generation: Long): Boolean {
        return synchronized(lifecycleLock) { isCurrentSessionLocked(generation) }
    }

    private fun isCurrentSessionLocked(generation: Long): Boolean {
        return !shutdownRequested && sessionGeneration == generation
    }

    companion object {
        internal const val DEFAULT_FRAME_PERIOD_MILLIS = 500L

        private fun assetFrameDecoder(
            context: Context,
            scenario: ReplayScenario
        ): () -> RgbaVisionFrame = {
            val bitmap = context.assets.open(scenario.assetPath).use { input ->
                requireNotNull(BitmapFactory.decodeStream(input)) {
                    "Unable to decode replay asset ${scenario.assetPath}"
                }
            }
            try {
                BitmapRgbaVisionFrame.from(bitmap)
            } finally {
                bitmap.recycle()
            }
        }

        private fun shutdownExecutor(executor: ExecutorService) {
            executor.shutdown()
            try {
                if (!executor.awaitTermination(500L, TimeUnit.MILLISECONDS)) {
                    executor.shutdownNow()
                }
            } catch (error: InterruptedException) {
                executor.shutdownNow()
                Thread.currentThread().interrupt()
            }
        }
    }
}
