package com.linnan.blindassist.egoclear

import android.content.Context
import android.os.SystemClock
import java.io.File
import java.nio.ByteBuffer
import java.util.zip.CRC32
import java.util.zip.ZipFile

internal object EgoClearSnpeNativeLoader {
    @Volatile private var loaded = false

    @Synchronized
    fun ensureLoaded(): Boolean {
        if (loaded) return true
        return try {
            System.loadLibrary("SnpeHtpV75Stub")
            System.loadLibrary("SnpeHtpV75CalculatorStub")
            System.loadLibrary("SnpeHtpPrepare")
            System.loadLibrary("egoclear_snpe_jni")
            loaded = true
            true
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }

    @Synchronized
    fun installSkeleton(apkContext: Context, cacheContext: Context): File {
        val dir = File(cacheContext.codeCacheDir, "egoclear-snpe-2.47-v75").apply {
            check(isDirectory || mkdirs()) { "Cannot create skeleton dir: $absolutePath" }
        }
        val target = File(dir, "libSnpeHtpV75Skel.so")
        ZipFile(apkContext.applicationInfo.sourceDir).use { apk ->
            val entry = requireNotNull(apk.getEntry("lib/arm64-v8a/${target.name}")) {
                "Missing libSnpeHtpV75Skel.so in ${apkContext.applicationInfo.sourceDir}"
            }
            fun matches(file: File): Boolean {
                if (!file.isFile || file.length() != entry.size) return false
                val crc = CRC32()
                file.inputStream().buffered().use { inp ->
                    val buf = ByteArray(65536)
                    while (true) {
                        val n = inp.read(buf)
                        if (n < 0) break
                        crc.update(buf, 0, n)
                    }
                }
                return crc.value == entry.crc
            }
            if (!matches(target)) {
                val tmp = File(dir, "${target.name}.tmp")
                try {
                    apk.getInputStream(entry).use { inp -> tmp.outputStream().use(inp::copyTo) }
                    check(matches(tmp))
                    check(tmp.renameTo(target)) { "Rename skeleton failed" }
                } finally {
                    tmp.delete()
                }
            }
        }
        check(target.setReadable(true, false) && target.setExecutable(true, false))
        return dir
    }
}

/**
 * Fast native C++ YUV_420_888 -> gravity-upright 16:9 crop -> 518x294 ImageNet NCHW preprocessor (~1 ms on SM8650).
 */
object EgoClearNativePreproc {
    fun preprocessYuv420ToNchw(
        yBuf: ByteBuffer,
        yRowStride: Int,
        yPixStride: Int,
        uBuf: ByteBuffer,
        uRowStride: Int,
        uPixStride: Int,
        vBuf: ByteBuffer,
        vRowStride: Int,
        vPixStride: Int,
        srcW: Int,
        srcH: Int,
        rotDeg: Int,
        cropX0: Float,
        cropY0: Float,
        cropW: Float,
        cropH: Float,
        outW: Int,
        outH: Int,
        outNchw: FloatArray,
    ): Float {
        if (!yBuf.isDirect || !uBuf.isDirect || !vBuf.isDirect) return -1f
        if (!EgoClearSnpeNativeLoader.ensureLoaded()) return -1f
        return nativePreprocessYuv420ToNchw(
            yBuf, yRowStride, yPixStride,
            uBuf, uRowStride, uPixStride,
            vBuf, vRowStride, vPixStride,
            srcW, srcH, rotDeg,
            cropX0, cropY0, cropW, cropH,
            outW, outH, outNchw,
        )
    }

    private external fun nativePreprocessYuv420ToNchw(
        yBuf: ByteBuffer, yRowStride: Int, yPixStride: Int,
        uBuf: ByteBuffer, uRowStride: Int, uPixStride: Int,
        vBuf: ByteBuffer, vRowStride: Int, vPixStride: Int,
        srcW: Int, srcH: Int, rotDeg: Int,
        cropX0: Float, cropY0: Float, cropW: Float, cropH: Float,
        outW: Int, outH: Int, outNchw: FloatArray,
    ): Float
}

/**
 * Qualcomm Snapdragon 8 Gen 3 (SM8650) Hexagon v75 HTP NPU FP16 encoder for EgoClear-v6.6 / v10.x / v11.
 * Runs the entire encoder backbone as 1 fused HTP subnet outputting (f2d, disp_cues|geom_cues).
 */
class EgoClearSnpeEncoder(
    apkContext: Context,
    cacheContext: Context,
    dlcFile: File,
    val auxOutputName: String = AUX_DISP_CUES,
    val f2dSize: Int = F2D_SIZE,
    val auxSize: Int = DC_SIZE,
) : AutoCloseable {

    private var handle: Long = 0L
    private var closed = false
    val loadMs: Double

    init {
        require(dlcFile.isFile) { "Missing EgoClear HTP DLC: ${dlcFile.absolutePath}" }
        val t0 = SystemClock.elapsedRealtimeNanos()
        check(EgoClearSnpeNativeLoader.ensureLoaded()) { "Failed to load SNPE HTP v75 native libs" }
        val dspDir = EgoClearSnpeNativeLoader.installSkeleton(apkContext, cacheContext)
        handle = nativeCreateWithAux(dlcFile.absolutePath, dspDir.absolutePath, auxOutputName)
        check(handle != 0L) { "Failed to create EgoClear HTP NPU engine ($auxOutputName)" }
        loadMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
    }

    @Synchronized
    fun encode(rgb: FloatArray, kScaled: FloatArray, rC2w: FloatArray): Pair<FloatArray, FloatArray> {
        check(!closed) { "EgoClearSnpeEncoder closed" }
        val f2d = FloatArray(f2dSize)
        val aux = FloatArray(auxSize)
        nativeExecute(handle, rgb, kScaled, rC2w, f2d, aux)
        return f2d to aux
    }

    @Synchronized
    override fun close() {
        if (!closed) {
            closed = true
            if (handle != 0L) {
                nativeDestroy(handle)
                handle = 0L
            }
        }
    }

    private external fun nativeCreate(dlcPath: String, dspDir: String): Long
    private external fun nativeCreateWithAux(dlcPath: String, dspDir: String, auxOutputName: String): Long
    private external fun nativeExecute(
        handle: Long,
        rgb: FloatArray,
        kScaled: FloatArray,
        rC2w: FloatArray,
        outF2d: FloatArray,
        outDc: FloatArray,
    )
    private external fun nativeDestroy(handle: Long)

    companion object {
        const val AUX_DISP_CUES = "disp_cues"
        const val AUX_GEOM_CUES = "geom_cues"
        const val F2D_SIZE = 64 * 42 * 74
        const val DC_SIZE = 4 * 42 * 74
        const val V10_F2D_SIZE = 64 * 84 * 148
        const val V10_GEOM_SIZE = 10 * 84 * 148
    }
}

/**
 * Qualcomm SM8650 Hexagon v75 HTP NPU streaming `step` engine (~4.9 ms, 100% VTCM SRAM, 0 DDR spill!).
 * Supports both v10.x (6 outputs) and v11 Omni (7 outputs including omni_probs [1, 6, 24]).
 */
class EgoClearSnpeStep(
    apkContext: Context,
    cacheContext: Context,
    dlcFile: File,
    val auxPrefix: String = "geom",
    val hasOmni: Boolean = false,
) : AutoCloseable {

    data class RawStepOutputs(
        val staticLogits: FloatArray,
        val dynamicLogits: FloatArray,
        val corrProbs: FloatArray,
        val uncLogits: FloatArray,
        val omniProbs: FloatArray?, // 6 * 24 when hasOmni == true
        val memNew: FloatArray,
        val ageNew: FloatArray,
    )

    private var handle: Long = 0L
    private var closed = false
    val loadMs: Double

    init {
        require(dlcFile.isFile) { "Missing EgoClear Step HTP DLC: ${dlcFile.absolutePath}" }
        val t0 = SystemClock.elapsedRealtimeNanos()
        check(EgoClearSnpeNativeLoader.ensureLoaded()) { "Failed to load SNPE HTP v75 native libs" }
        val dspDir = EgoClearSnpeNativeLoader.installSkeleton(apkContext, cacheContext)
        handle = nativeCreateStep(dlcFile.absolutePath, dspDir.absolutePath, auxPrefix, hasOmni)
        check(handle != 0L) { "Failed to create EgoClear HTP Step NPU engine (hasOmni=$hasOmni)" }
        loadMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
    }

    @Synchronized
    fun execute(inputs: Array<FloatArray>): RawStepOutputs {
        check(!closed) { "EgoClearSnpeStep closed" }
        val outStatic = FloatArray(3 * 24 * 15)
        val outDynamic = FloatArray(3 * 24 * 15)
        val outCorr = FloatArray(24)
        val outUnc = FloatArray(24)
        val outOmni = if (hasOmni) FloatArray(6 * 24) else null
        val outMem = FloatArray(128 * 24 * 15)
        val outAge = FloatArray(1 * 24 * 15)
        nativeExecuteStep(handle, inputs, outStatic, outDynamic, outCorr, outUnc, outOmni, outMem, outAge)
        return RawStepOutputs(outStatic, outDynamic, outCorr, outUnc, outOmni, outMem, outAge)
    }

    @Synchronized
    override fun close() {
        if (!closed) {
            closed = true
            if (handle != 0L) {
                nativeDestroyStep(handle)
                handle = 0L
            }
        }
    }

    private external fun nativeCreateStep(dlcPath: String, dspDir: String, auxPrefix: String, hasOmni: Boolean): Long
    private external fun nativeExecuteStep(
        handle: Long,
        inArrays: Array<FloatArray>,
        outStatic: FloatArray,
        outDynamic: FloatArray,
        outCorr: FloatArray,
        outUnc: FloatArray,
        outOmni: FloatArray?,
        outMem: FloatArray,
        outAge: FloatArray,
    )
    private external fun nativeDestroyStep(handle: Long)
}
