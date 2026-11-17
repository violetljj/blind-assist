package com.linnan.blindassist.vision

import android.app.Application
import android.graphics.Bitmap
import java.io.File

/** Metric optical depth (metres) for the calibrated Atom upright camera; direction A, 2026-11-18.
 * The model is camera-specific (its ray embedding is baked for one K), so the caller must pass frames of that camera. */
interface MetricDepthRuntime : AutoCloseable {
    /** Input: undistorted upright 480x640 RGB. Output: row-major 480x640 optical z in metres. */
    fun estimate(upright: Bitmap): FloatArray
}

object UniDepthRuntimeFactory {
    const val MODEL = "unidepth_s_atom_658x490_fp32.dlc"
    const val RAYS = "unidepth_s_atom_rays_658x490.raw"

    fun available(directory: File) = File(directory, MODEL).isFile && File(directory, RAYS).isFile

    fun create(application: Application, directory: File): MetricDepthRuntime {
        val type = try {
            Class.forName("com.linnan.blindassist.vision.UniDepthSnpeRuntime")
        } catch (error: ClassNotFoundException) {
            throw IllegalStateException("This build has no SNPE runtime. Build with a local QAIRT SDK.", error)
        }
        return type.getConstructor(Application::class.java, File::class.java, File::class.java)
            .newInstance(application, File(directory, MODEL), File(directory, RAYS)) as MetricDepthRuntime
    }
}
