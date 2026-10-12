package com.linnan.blindassist.vision

import android.app.Application
import android.graphics.Bitmap
import java.io.File

/** Relative disparity only. Caller owns fisheye correction, metric anchoring and reminders. */
interface Dav2DepthRuntime : AutoCloseable {
    /** Input must be undistorted upright 480x640 RGB; output is row-major at that resolution. */
    fun estimate(upright: Bitmap): FloatArray
}

object Dav2DepthRuntimeFactory {
    fun create(application: Application, modelFile: File): Dav2DepthRuntime {
        val type = try {
            Class.forName("com.linnan.blindassist.vision.Dav2SnpeDepthEstimator")
        } catch (error: ClassNotFoundException) {
            throw IllegalStateException("This build has no DAV2 SNPE runtime. Build with a local QAIRT SDK and enableDav2Snpe=true.", error)
        }
        return type.getConstructor(Application::class.java, File::class.java)
            .newInstance(application, modelFile) as Dav2DepthRuntime
    }
}
