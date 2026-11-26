package com.linnan.blindassist.vision

import android.app.Application
import android.os.SystemClock
import com.linnan.blindassist.vision.egogeom.EgoGeomContracts
import com.linnan.blindassist.vision.egogeom.EgoGeomOutput
import com.linnan.blindassist.vision.egogeom.EgoGeomRuntime
import com.linnan.blindassist.vision.egogeom.GlassCorridorRiskEvaluator
import java.io.File

/**
 * Direct QNN HTP v75 context-binary runtime for EgoGeom-Omni v4.1 (+ optional G2 Glass Head).
 * Uses the APK's bundled QNN HTP libraries + `QnnHtpSkelInstaller` and executes via `egogeom_qnn_jni`.
 */
class EgoGeomQnnRuntime(
    application: Application,
    modelFile: File,
) : EgoGeomRuntime {
    private var handle: Long
    private var closed = false
    override val supportsTof: Boolean
    override val supportsGlass: Boolean

    init {
        require(modelFile.isFile) { "EgoGeom QNN context binary missing: $modelFile" }
        val skelDir = QnnHtpSkelInstaller.install(application)
        System.loadLibrary("QnnHtpV75Stub")
        System.loadLibrary("QnnHtp")
        System.loadLibrary("QnnSystem")
        System.loadLibrary("egogeom_qnn_jni")
        handle = nativeCreate(modelFile.absolutePath, skelDir.absolutePath)
        check(handle != 0L) { "Failed to create EgoGeom QNN HTP engine for $modelFile" }
        supportsTof = nativeSupportsTof(handle)
        supportsGlass = nativeSupportsGlass(handle)
    }

    @Synchronized
    override fun infer(
        rgbNchw: FloatArray,
        gravRaysNchw: FloatArray,
        hostFocal: Float,
        tofGridS4: FloatArray?,
        hasTof: Boolean,
    ): EgoGeomOutput {
        check(!closed) { "EgoGeomQnnRuntime is closed" }
        require(rgbNchw.size == EgoGeomContracts.RGB_FLOATS) { "Expected ${EgoGeomContracts.RGB_FLOATS} RGB floats" }
        require(gravRaysNchw.size == EgoGeomContracts.RAYS_FLOATS) { "Expected ${EgoGeomContracts.RAYS_FLOATS} ray floats" }
        if (hasTof && supportsTof) {
            require(tofGridS4 != null && tofGridS4.size == EgoGeomContracts.TOF_S4_FLOATS) {
                "Expected ${EgoGeomContracts.TOF_S4_FLOATS} ToF floats when hasTof=true"
            }
        }
        val depthM = FloatArray(EgoGeomContracts.DEPTH_FLOATS)
        val deltaHM = FloatArray(EgoGeomContracts.DEPTH_FLOATS)
        val normalGrav = FloatArray(EgoGeomContracts.NORMAL_FLOATS)
        val logSigma = FloatArray(EgoGeomContracts.DEPTH_FLOATS)
        val glassProbS16 = if (supportsGlass) FloatArray(GlassCorridorRiskEvaluator.S16_FLOATS) else null

        val t0 = SystemClock.elapsedRealtimeNanos()
        nativeExecute(
            handle,
            rgbNchw,
            gravRaysNchw,
            hostFocal.coerceIn(0.20f, 4.0f),
            if (hasTof && supportsTof) tofGridS4 else null,
            hasTof && supportsTof,
            depthM,
            deltaHM,
            normalGrav,
            logSigma,
            glassProbS16,
        )
        val execMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
        return EgoGeomOutput(depthM, deltaHM, normalGrav, logSigma, execMs, glassProbS16)
    }

    @Synchronized
    override fun close() {
        if (!closed) {
            closed = true
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private external fun nativeCreate(binPath: String, dspDir: String): Long
    private external fun nativeSupportsTof(handle: Long): Boolean
    private external fun nativeSupportsGlass(handle: Long): Boolean
    private external fun nativeExecute(
        handle: Long,
        rgb: FloatArray,
        gravRays: FloatArray,
        hostFocal: Float,
        tofGridS4: FloatArray?,
        hasTof: Boolean,
        outDepth: FloatArray,
        outDeltaH: FloatArray,
        outNormal: FloatArray,
        outLogSigma: FloatArray,
        outGlassProbS16: FloatArray?,
    )

    private external fun nativeDestroy(handle: Long)
}
