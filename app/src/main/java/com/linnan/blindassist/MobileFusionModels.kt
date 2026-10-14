package com.linnan.blindassist

import android.app.Application
import com.linnan.blindassist.vision.CnhStrongEngine
import com.linnan.blindassist.vision.Dav2DepthRuntime
import com.linnan.blindassist.vision.Dav2DepthRuntimeFactory
import com.linnan.blindassist.vision.MobileFusionCamera
import java.io.File

/** Process-wide fusion model cache: loaded once when the demo page opens, reused by every session.
 * Sessions reset the ToF history themselves; DAV2 and the calibration are stateless. Only one
 * session runs at a time (MobileFusionLease), so the cached engines are never used concurrently. */
internal object MobileFusionModels {
    class Loaded(val camera: MobileFusionCamera, val tof: CnhStrongEngine, val depth: Dav2DepthRuntime)

    private var loaded: Loaded? = null
    private var loadedDir: String? = null
    @Volatile var ready = false
        private set

    @Synchronized
    fun acquire(application: Application, directory: File): Loaded {
        loaded?.takeIf { loadedDir == directory.absolutePath }?.let { return it }
        val camera = MobileFusionCamera(directory)
        val tof = CnhStrongEngine(File(directory, "cnh_strong_m3.onnx"), File(directory, "cnh_strong_calibration.bin"))
        val depth = try { Dav2DepthRuntimeFactory.create(application, File(directory, "dav2s_686x518_fp32.dlc")) }
                    catch (error: Exception) { runCatching { tof.close() }; throw error }
        return Loaded(camera, tof, depth).also { loaded = it; loadedDir = directory.absolutePath; ready = true }
    }

    fun preload(application: Application, directory: File) {
        if (ready) return
        Thread({ runCatching { acquire(application, directory) } }, "fusion-preload").start()
    }

    /** Frees the accelerator once no session holds the lease; a later page reloads on demand. */
    fun releaseWhenIdle() {
        Thread({
            val deadline = System.currentTimeMillis() + 60_000
            while (MobileFusionLease.busy && System.currentTimeMillis() < deadline) Thread.sleep(200)
            synchronized(this) {
                if (MobileFusionLease.busy) return@Thread
                loaded?.let { runCatching { it.tof.close() }; runCatching { it.depth.close() } }
                loaded = null; loadedDir = null; ready = false
            }
        }, "fusion-release").start()
    }
}
