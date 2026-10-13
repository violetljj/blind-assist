package com.linnan.blindassist.vision

import android.app.Application
import android.graphics.Bitmap
import java.io.File
import java.util.zip.CRC32
import java.util.zip.ZipFile
import kotlin.math.floor

/** Explicit experimental DAV2-Small relative-depth runtime. Never downloads or chooses a model. */
class Dav2SnpeDepthEstimator(
    application: Application,
    modelFile: File,
) : Dav2DepthRuntime {
    private var handle: Long
    val inputShape: IntArray
    enum class Runtime { DSP }
    val runtime: Runtime = Runtime.DSP // Native builder has one DSP runtime and no CPU fallback.
    private var closed = false

    init {
        require(modelFile.isFile) { "DAV2 DLC missing: $modelFile" }
        System.loadLibrary("SnpeHtpV75Stub")
        System.loadLibrary("SnpeHtpV75CalculatorStub")
        System.loadLibrary("SnpeHtpPrepare")
        System.loadLibrary("dav2_snpe_jni")
        handle = nativeCreate(modelFile.absolutePath, installSkeleton(application).absolutePath)
        check(handle != 0L)
        try {
            inputShape = nativeShape(handle)
            require(inputShape.contentEquals(intArrayOf(1, HEIGHT, WIDTH, 3)) ||
                inputShape.contentEquals(intArrayOf(1, 3, HEIGHT, WIDTH))) {
                "Expected fixed DAV2 686x518 RGB; got ${inputShape.contentToString()}"
            }
        } catch (error: Throwable) {
            nativeDestroy(handle)
            handle = 0
            throw error
        }
    }

    /** For recorded snpe-net-run fixtures: floats are already in the SDK-reported native layout. */
    @Synchronized
    fun executeNativeInput(input: FloatArray): FloatArray {
        check(!closed) { "DAV2 runtime is closed" }
        require(input.size == WIDTH * HEIGHT * 3 && input.all { it.isFinite() })
        return nativeExecute(handle, input).also {
            check(it.size == WIDTH * HEIGHT && it.all { value -> value.isFinite() }) { "Invalid DAV2 disparity" }
        }
    }

    @Synchronized
    override fun estimate(upright: Bitmap): FloatArray {
        require(upright.width == 480 && upright.height == 640) { "Expected undistorted upright Atom 480x640" }
        val chw = preprocessNchw(upright)
        val input = if (inputShape[1] == 3) chw else FloatArray(chw.size).also { hwc ->
            val plane = WIDTH * HEIGHT
            for (i in 0 until plane) for (c in 0..2) hwc[3 * i + c] = chw[c * plane + i]
        }
        return resizeAlignCorners(executeNativeInput(input), 480, 640)
    }

    @Synchronized
    override fun close() {
        if (!closed) {
            closed = true
            nativeDestroy(handle)
            handle = 0
        }
    }

    private external fun nativeCreate(path: String, dspPath: String): Long
    private external fun nativeShape(handle: Long): IntArray
    private external fun nativeExecute(handle: Long, input: FloatArray): FloatArray
    private external fun nativeDestroy(handle: Long)

    companion object {
        const val WIDTH = 518
        const val HEIGHT = 686
        private data class CubicAxis(val indices: IntArray, val weights: FloatArray)

        // This runtime admits only 480x640 -> 518x686. The interpolation geometry is
        // immutable; rebuilding wx inside the pixel loop created 355,348 arrays/frame.
        private val horizontal = cubicAxis(480, WIDTH)
        private val vertical = cubicAxis(640, HEIGHT, 480)

        private fun cubicAxis(sourceSize: Int, targetSize: Int, stride: Int = 1): CubicAxis {
            val indices = IntArray(targetSize * 4)
            val weights = FloatArray(targetSize * 4)
            for (i in 0 until targetSize) {
                val source = (i + .5) * sourceSize / targetSize - .5
                val base = floor(source).toInt()
                val taps = cubic((source - base).toFloat())
                for (tap in 0..3) {
                    indices[i * 4 + tap] = (base + tap - 1).coerceIn(0, sourceSize - 1) * stride
                    weights[i * 4 + tap] = taps[tap]
                }
            }
            return CubicAxis(indices, weights)
        }

        // FastRPC needs an actual skeleton file even when Android loads arm64 libs from the APK.
        @Synchronized
        private fun installSkeleton(application: Application): File {
            val directory = File(application.codeCacheDir, "snpe-2.47-v75").apply {
                check(isDirectory || mkdirs())
            }
            val target = File(directory, "libSnpeHtpV75Skel.so")
            ZipFile(application.applicationInfo.sourceDir).use { apk ->
                val entry = requireNotNull(apk.getEntry("lib/arm64-v8a/${target.name}"))
                fun matches(file: File): Boolean {
                    if (!file.isFile || file.length() != entry.size) return false
                    val crc = CRC32()
                    file.inputStream().buffered().use { input ->
                        val buffer = ByteArray(65536)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            crc.update(buffer, 0, count)
                        }
                    }
                    return crc.value == entry.crc
                }
                if (!matches(target)) {
                    val temporary = File(directory, "${target.name}.tmp")
                    try {
                        apk.getInputStream(entry).use { input -> temporary.outputStream().use(input::copyTo) }
                        check(matches(temporary))
                        check(temporary.renameTo(target)) { "Install SNPE skeleton failed" }
                    } finally {
                        temporary.delete()
                    }
                }
            }
            check(target.setReadable(true, false) && target.setExecutable(true, false))
            return directory
        }

        // OpenCV INTER_CUBIC: half-pixel coordinates, a=-0.75, replicated edge.
        // Work on float RGB/255 before resizing, as upstream image2tensor does.
        fun preprocessNchw(bitmap: Bitmap): FloatArray {
            require(bitmap.width == 480 && bitmap.height == 640)
            val src = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(src, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            return Dav2NativePreprocess.apply(src, horizontal.indices, horizontal.weights,
                vertical.indices, vertical.weights)
        }

        private fun cubic(x: Float): FloatArray {
            val a = -.75f
            val c0 = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a
            val c1 = ((a + 2) * x - (a + 3)) * x * x + 1
            val t = 1 - x
            val c2 = ((a + 2) * t - (a + 3)) * t * t + 1
            return floatArrayOf(c0, c1, c2, 1 - c0 - c1 - c2)
        }

        fun resizeAlignCorners(input: FloatArray, width: Int, height: Int): FloatArray {
            require(input.size == WIDTH * HEIGHT && width > 1 && height > 1)
            return FloatArray(width * height) { i ->
                val x = (i % width).toDouble() * (WIDTH - 1) / (width - 1)
                val y = (i / width).toDouble() * (HEIGHT - 1) / (height - 1)
                val x0 = x.toInt(); val y0 = y.toInt()
                val x1 = (x0 + 1).coerceAtMost(WIDTH - 1)
                val y1 = (y0 + 1).coerceAtMost(HEIGHT - 1)
                val fx = (x - x0).toFloat(); val fy = (y - y0).toFloat()
                val top = input[y0 * WIDTH + x0] * (1 - fx) + input[y0 * WIDTH + x1] * fx
                val bottom = input[y1 * WIDTH + x0] * (1 - fx) + input[y1 * WIDTH + x1] * fx
                top * (1 - fy) + bottom * fy
            }
        }
    }
}

/** Loads only the existing packaged bridge; preprocessing does not initialize a model or HTP. */
internal object Dav2NativePreprocess {
    init { System.loadLibrary("dav2_snpe_jni") }
    external fun apply(pixels: IntArray, xIndices: IntArray, xWeights: FloatArray,
        yOffsets: IntArray, yWeights: FloatArray): FloatArray
}
