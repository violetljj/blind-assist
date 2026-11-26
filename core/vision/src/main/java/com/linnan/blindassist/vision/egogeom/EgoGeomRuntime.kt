package com.linnan.blindassist.vision.egogeom

import android.app.Application
import java.io.File

/**
 * Full 4-head metric geometry output (+ optional frozen-DINOv3 G2 `18x24` glass probability map)
 * of EgoGeom-Omni v4.1 at native 288x384 NCHW resolution.
 */
class EgoGeomOutput(
    val depthM: FloatArray,
    val deltaHM: FloatArray,
    val normalGrav: FloatArray,
    val logSigma: FloatArray,
    val executeMs: Double,
    val glassProbS16: FloatArray? = null,
)

/**
 * Unified Pure-RGB / Omni+Signed-H3 (+ optional G2 Glass Head) Hexagon v75 runtime interface for EgoGeom-Omni v4.1
 * (`pc_fp16dec_a16`, exact 0.000000 m zero-interference).
 */
interface EgoGeomRuntime : AutoCloseable {
    val supportsTof: Boolean
    val supportsGlass: Boolean
        get() = false

    /**
     * Execute one frame on Hexagon v75 HTP.
     * - `rgbNchw`: `[3, 288, 384]` ImageNet-normalized float32
     * - `gravRaysNchw`: `[3, 288, 384]` gravity-aligned unit ray map
     * - `hostFocal`: analytical 4-row chord focal factor (`[0.20, 4.0]`)
     * - `tofGridS4`: optional `[20, 72, 96]` packed signed-H3 CNH tensor (ignored if `hasTof == false`)
     */
    fun infer(
        rgbNchw: FloatArray,
        gravRaysNchw: FloatArray,
        hostFocal: Float,
        tofGridS4: FloatArray? = null,
        hasTof: Boolean = false,
    ): EgoGeomOutput
}

object EgoGeomQnnRuntimeFactory {
    const val OMNI_BIN = "egogeom_omni_v41_omni_v75_pc_fp16dec_a16.bin"
    const val PURE_RGB_BIN = "egogeom_omni_v41_pure_rgb_v75_pc_fp16dec_a16.bin"
    const val OMNI_G2_BIN = "egogeom_omni_v41_omni_g2_v75_pc_fp16dec_a16.bin"
    const val PURE_RGB_G2_BIN = "egogeom_omni_v41_pure_rgb_g2_v75_pc_fp16dec_a16.bin"

    fun available(directory: File): Boolean =
        File(directory, OMNI_G2_BIN).isFile ||
            File(directory, PURE_RGB_G2_BIN).isFile ||
            File(directory, OMNI_BIN).isFile ||
            File(directory, PURE_RGB_BIN).isFile

    fun create(application: Application, modelFile: File): EgoGeomRuntime {
        val type = try {
            Class.forName("com.linnan.blindassist.vision.EgoGeomQnnRuntime")
        } catch (error: ClassNotFoundException) {
            throw IllegalStateException(
                "This build has no EgoGeom QNN HTP runtime. Build with a local QAIRT SDK and enableDav2Snpe=true.",
                error,
            )
        }
        return type.getConstructor(Application::class.java, File::class.java)
            .newInstance(application, modelFile) as EgoGeomRuntime
    }
}
