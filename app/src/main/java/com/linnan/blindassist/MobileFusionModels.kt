package com.linnan.blindassist

import android.app.Application
import com.linnan.blindassist.vision.CnhStrongEngine
import com.linnan.blindassist.vision.Dav2DepthRuntime
import com.linnan.blindassist.vision.Dav2DepthRuntimeFactory
import com.linnan.blindassist.vision.MetricDepthRuntime
import com.linnan.blindassist.vision.MobileFusionCamera
import com.linnan.blindassist.vision.UniDepthRuntimeFactory
import java.io.File

/** Process-wide fusion model cache: loaded once when the demo page opens, reused by every session.
 * Sessions reset the ToF history themselves; DAV2 and the calibration are stateless. Only one
 * session runs at a time (MobileFusionLease), so the cached engines are never used concurrently. */
internal object MobileFusionModels {
    /** Exactly one depth runtime is loaded: DAV2-S (current method) or, with [metric], UniDepth-V2-S (direction A). */
    class Loaded(val camera: MobileFusionCamera, val tof: CnhStrongEngine, val depth: Dav2DepthRuntime?,
                 val metric: MetricDepthRuntime?) {
        fun closeDepth() { runCatching { depth?.close() }; runCatching { metric?.close() } }
    }

    private var loaded: Loaded? = null
    private var loadedDir: String? = null
    private var loadedMetric = false
    @Volatile var ready = false
        private set

    @Synchronized
    fun acquire(application: Application, directory: File, metric: Boolean = false): Loaded {
        loaded?.takeIf { loadedDir == directory.absolutePath && loadedMetric == metric }?.let { return it }
        // Switching the depth model frees the other HTP graph first; camera and ToF engine are reused.
        val previous = loaded?.takeIf { loadedDir == directory.absolutePath }
        previous?.closeDepth(); if (previous == null) loaded?.let { runCatching { it.tof.close() }; it.closeDepth() }
        loaded = null; ready = false
        val camera = previous?.camera ?: MobileFusionCamera(directory)
        val tof = previous?.tof ?: CnhStrongEngine(File(directory, "cnh_strong_m3.onnx"), File(directory, "cnh_strong_calibration.bin"))
        val result = try {
            if (metric) Loaded(camera, tof, null, UniDepthRuntimeFactory.create(application, directory))
            else Loaded(camera, tof, Dav2DepthRuntimeFactory.create(application, File(directory, "dav2s_686x518_fp32.dlc")), null)
        } catch (error: Exception) { runCatching { tof.close() }; throw error }
        return result.also { loaded = it; loadedDir = directory.absolutePath; loadedMetric = metric; ready = true }
    }

    fun preload(application: Application, directory: File, metric: Boolean = false) {
        if (ready && loadedMetric == metric) return
        Thread({ runCatching { acquire(application, directory, metric) } }, "fusion-preload").start()
    }

    /** Frees the accelerator once no session holds the lease; a later page reloads on demand. */
    fun releaseWhenIdle() {
        Thread({
            val deadline = System.currentTimeMillis() + 60_000
            while (MobileFusionLease.busy && System.currentTimeMillis() < deadline) Thread.sleep(200)
            synchronized(this) {
                if (MobileFusionLease.busy) return@Thread
                loaded?.let { runCatching { it.tof.close() }; it.closeDepth() }
                loaded = null; loadedDir = null; ready = false
            }
        }, "fusion-release").start()
    }
}
