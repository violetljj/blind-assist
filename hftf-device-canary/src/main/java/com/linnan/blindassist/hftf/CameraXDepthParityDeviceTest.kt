package com.linnan.blindassist.hftf

import android.Manifest
import android.os.Bundle
import android.os.SystemClock
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.linnan.blindassist.hftf.metricdepth.KnownHeightGroundPipeline
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** One-shot, same-frame CameraX parity evidence capture. No production routing authority. */
@RunWith(AndroidJUnit4::class)
class CameraXDepthParityDeviceTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun captureSameFrameParityEvidence() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        val cachedDlc = File(requireNotNull(arguments.getString("cachedDlcPath")))
        val warmupFrames = arguments.getString("warmupFrames")?.toInt() ?: 12
        val lensFacing = arguments.getString("lensFacing") ?: "BACK"
        val cameraSelector = when (lensFacing) {
            "BACK" -> CameraSelector.DEFAULT_BACK_CAMERA
            "FRONT" -> CameraSelector.DEFAULT_FRONT_CAMERA
            else -> error("lensFacing must be BACK or FRONT")
        }
        require(warmupFrames in 1..120)
        assertTrue(cachedDlc.isFile)

        val outputRoot = File(
            requireNotNull(context.getExternalFilesDir(null)),
            arguments.getString("outputName") ?: "camerax-depth-parity-r0",
        )
        require(!outputRoot.exists()) { "output already exists: $outputRoot" }
        assertTrue(outputRoot.mkdirs())

        val provider = ProcessCameraProvider.getInstance(context).get(10, TimeUnit.SECONDS)
        val owner = TestLifecycleOwner()
        val executor = Executors.newSingleThreadExecutor()
        val latch = CountDownLatch(1)
        val seen = AtomicInteger()
        val captured = AtomicReference<OwnedYuv420Frame?>()
        val captureMetadata = AtomicReference<JSONObject?>()
        val failures = mutableListOf<String>()
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(WIDTH, HEIGHT), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
                ).build(),
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
        analysis.setAnalyzer(executor) { image ->
            try {
                val index = seen.incrementAndGet()
                if (captured.get() != null || index < warmupFrames) return@setAnalyzer
                require(image.width == WIDTH && image.height == HEIGHT && image.planes.size == 3)
                val frame = OwnedYuv420Frame(WIDTH, HEIGHT) { }.lease()
                copyPlane(image.planes[0], frame.y, WIDTH, HEIGHT)
                copyPlane(image.planes[1], frame.u, WIDTH / 2, HEIGHT / 2)
                copyPlane(image.planes[2], frame.v, WIDTH / 2, HEIGHT / 2)
                frame.width = WIDTH
                frame.height = HEIGHT
                frame.rotationDegrees = image.imageInfo.rotationDegrees
                frame.sensorTimestampNanos = image.imageInfo.timestamp
                frame.receivedAtNanos = SystemClock.elapsedRealtimeNanos()
                captureMetadata.set(
                    JSONObject()
                        .put("frame_index", index)
                        .put("width", image.width)
                        .put("height", image.height)
                        .put("format", "YUV_420_888")
                        .put("rotation_degrees", frame.rotationDegrees)
                        .put("sensor_timestamp_nanos", frame.sensorTimestampNanos)
                        .put("received_at_elapsed_realtime_nanos", frame.receivedAtNanos)
                        .put("planes", JSONArray(image.planes.map {
                            JSONObject().put("row_stride", it.rowStride).put("pixel_stride", it.pixelStride)
                                .put("remaining_bytes", it.buffer.remaining())
                        })),
                )
                captured.set(frame)
                latch.countDown()
            } catch (failure: Throwable) {
                failures += "${failure.javaClass.simpleName}: ${failure.message}"
                latch.countDown()
            } finally {
                image.close()
            }
        }

        try {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.resume()
                provider.bindToLifecycle(owner, cameraSelector, analysis)
            }
            assertTrue("CameraX frame capture timed out", latch.await(20, TimeUnit.SECONDS))
        } finally {
            instrumentation.runOnMainSync { provider.unbindAll(); owner.destroy() }
            analysis.clearAnalyzer()
            executor.shutdown()
            executor.awaitTermination(5, TimeUnit.SECONDS)
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
        val frame = requireNotNull(captured.get())
        val metadata = requireNotNull(captureMetadata.get())
        val nativeLibraryDir = arguments.getString("qnnRuntimeDir")
            ?: instrumentation.context.applicationInfo.nativeLibraryDir

        Dav2Yuv420RgbConverter().use { converter ->
            Dav2NativePreprocessor().use { preprocessor ->
                Dav2QnnCachedContext(cachedDlc.absolutePath, nativeLibraryDir).use { runtime ->
                    frame.y.writeTo(File(outputRoot, "camerax_y_640x480_u8.raw"))
                    frame.u.writeTo(File(outputRoot, "camerax_u_320x240_u8.raw"))
                    frame.v.writeTo(File(outputRoot, "camerax_v_320x240_u8.raw"))
                    val rgb = converter.convert(frame).copyOf()
                    rgb.writeTo(File(outputRoot, "rgb_crop_640x480_uint8.raw"))
                    val rgbStats = rgbStats(rgb)

                    val fastFp32 = preprocessor.preprocessFp32(rgb)
                    writeBuffer(fastFp32, File(outputRoot, "native_fast_normalized_nchw_fp32.raw"))
                    val fastResized = preprocessor.copyLastResizedHwcFp32()
                    writeBuffer(fastResized, File(outputRoot, "native_fast_resized_hwc_fp32.raw"))
                    val fastFp16 = preprocessor.preprocessFp16Strict(rgb)
                    writeBuffer(fastFp16, File(outputRoot, "native_fast_normalized_nchw_fp16.raw"))
                    val nativeFp32 = preprocessor.preprocessFp32Canonical(rgb)
                    writeBuffer(nativeFp32, File(outputRoot, "native_normalized_nchw_fp32.raw"))
                    val nativeFp16 = preprocessor.preprocessFp16CanonicalStrict(rgb)
                    writeBuffer(nativeFp16, File(outputRoot, "native_normalized_nchw_fp16.raw"))
                    // Restore the canonical FP32 -> strict-FP16 buffer before QNN.
                    preprocessor.preprocessFp16(rgb)
                    nativeFp16.position(0)
                    val kotlinInputHash = fnv1a64(nativeFp16)
                    val appDepth = runtime.execute(nativeFp16)
                    writeBuffer(appDepth, File(outputRoot, "app_qnn_depth_fp16.raw"))

                    val rawDepth = FloatArray(Dav2PreprocessContract.PLANE)
                    val shorts = appDepth.duplicate().order(ByteOrder.nativeOrder()).apply { position(0) }.asShortBuffer()
                    for (index in rawDepth.indices) rawDepth[index] = halfBitsToFloat(shorts.get(index))
                    val alignedDepth = FloatArray(WIDTH * HEIGHT)
                    resizeDepthAlignCorners(rawDepth, alignedDepth)
                    writeFloatArray(alignedDepth, File(outputRoot, "app_depth_aligned_640x480_fp32.raw"))
                    val geometry = KnownHeightGroundPipeline.evaluateGeometry(
                        alignedDepth, WIDTH, HEIGHT, 320.0, 320.0, 320.0, 240.0, CAMERA_HEIGHT_M,
                    )
                    val geometryJson = geometryJson(geometry)
                    File(outputRoot, "app_geometry.json").writeText(geometryJson.toString(2))

                    metadata
                        .put("schema", "blindassist_camerax_depth_parity_capture_r0")
                        .put("lens_facing", lensFacing)
                        .put("rgb_content", rgbStats)
                        .put("output_root", outputRoot.absolutePath)
                        .put("qnn_input_limit_bytes", nativeFp16.limit())
                        .put("qnn_input_fnv1a64_kotlin", java.lang.Long.toUnsignedString(kotlinInputHash, 16))
                        .put("qnn_input_fnv1a64_native", java.lang.Long.toUnsignedString(runtime.lastInputFnv1a64, 16))
                        .put("geometry", geometryJson)
                        .put("files", fileManifest(outputRoot))
                }
            }
        }
        frame.close()
        val rgbContent = metadata.getJSONObject("rgb_content")
        metadata.put("gate_pass", failures.isEmpty() &&
            metadata.getString("qnn_input_fnv1a64_kotlin") == metadata.getString("qnn_input_fnv1a64_native") &&
            rgbContent.getInt("range") >= 16 && rgbContent.getDouble("standard_deviation") >= 2.0)
        File(outputRoot, "capture.json").writeText(metadata.toString(2))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, metadata.toString()) })
        assertTrue(metadata.toString(2), metadata.getBoolean("gate_pass"))
    }

    private fun geometryJson(value: Any): JSONObject = when (value) {
        is KnownHeightGroundPipeline.Geometry -> JSONObject()
            .put("status", "VALID")
            .put("relative_height", value.relativeHeight)
            .put("normalized_median_residual", value.normalizedMedianResidual)
            .put("inlier_fraction", value.inlierFraction)
            .put("normal", JSONArray(value.normal.toList()))
            .put("features", JSONArray(value.features.toList()))
        is KnownHeightGroundPipeline.Unknown -> JSONObject().put("status", "UNKNOWN").put("reason", value.reason)
        else -> JSONObject().put("status", "UNKNOWN").put("reason", value.toString())
    }

    private fun fileManifest(root: File): JSONObject = JSONObject().also { output ->
        root.listFiles()?.filter(File::isFile)?.sortedBy(File::getName)?.forEach { file ->
            output.put(file.name, JSONObject().put("bytes", file.length()).put("sha256", sha256(file)))
        }
    }

    private fun copyPlane(plane: ImageProxy.PlaneProxy, target: ByteArray, width: Int, height: Int) {
        val source = plane.buffer.duplicate()
        for (row in 0 until height) {
            val sourceStart = row * plane.rowStride
            val targetStart = row * width
            if (plane.pixelStride == 1) {
                source.position(sourceStart)
                source.get(target, targetStart, width)
            } else {
                for (column in 0 until width) target[targetStart + column] = source.get(sourceStart + column * plane.pixelStride)
            }
        }
    }

    private fun writeBuffer(buffer: ByteBuffer, file: File) {
        val copy = buffer.duplicate().apply { position(0) }
        FileOutputStream(file).channel.use { channel -> while (copy.hasRemaining()) channel.write(copy) }
    }

    private fun ByteArray.writeTo(file: File) = file.writeBytes(this)

    private fun writeFloatArray(values: FloatArray, file: File) {
        val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        buffer.asFloatBuffer().put(values)
        file.writeBytes(buffer.array())
    }

    private fun fnv1a64(buffer: ByteBuffer): Long {
        var hash = -3750763034362895579L
        val copy = buffer.duplicate().apply { position(0) }
        while (copy.hasRemaining()) {
            hash = hash xor (copy.get().toLong() and 0xffL)
            hash *= 1099511628211L
        }
        return hash
    }

    private fun rgbStats(rgb: ByteArray): JSONObject {
        var minimum = 255
        var maximum = 0
        var sum = 0.0
        var squareSum = 0.0
        for (value in rgb) {
            val unsigned = value.toInt() and 0xff
            minimum = minOf(minimum, unsigned)
            maximum = maxOf(maximum, unsigned)
            sum += unsigned
            squareSum += unsigned.toDouble() * unsigned
        }
        val mean = sum / rgb.size
        val variance = (squareSum / rgb.size - mean * mean).coerceAtLeast(0.0)
        return JSONObject()
            .put("minimum", minimum)
            .put("maximum", maximum)
            .put("range", maximum - minimum)
            .put("mean", mean)
            .put("standard_deviation", kotlin.math.sqrt(variance))
    }

    private fun sha256(file: File): String = MessageDigest.getInstance("SHA-256")
        .digest(file.readBytes()).joinToString("") { "%02X".format(it) }

    private fun resizeDepthAlignCorners(input: FloatArray, output: FloatArray) {
        for (row in 0 until HEIGHT) {
            val sy = row.toDouble() * (Dav2PreprocessContract.OUTPUT_HEIGHT - 1) / (HEIGHT - 1)
            val y0 = sy.toInt(); val y1 = minOf(y0 + 1, Dav2PreprocessContract.OUTPUT_HEIGHT - 1); val fy = sy - y0
            for (column in 0 until WIDTH) {
                val sx = column.toDouble() * (Dav2PreprocessContract.OUTPUT_WIDTH - 1) / (WIDTH - 1)
                val x0 = sx.toInt(); val x1 = minOf(x0 + 1, Dav2PreprocessContract.OUTPUT_WIDTH - 1); val fx = sx - x0
                val top = input[y0 * Dav2PreprocessContract.OUTPUT_WIDTH + x0] * (1 - fx) +
                    input[y0 * Dav2PreprocessContract.OUTPUT_WIDTH + x1] * fx
                val bottom = input[y1 * Dav2PreprocessContract.OUTPUT_WIDTH + x0] * (1 - fx) +
                    input[y1 * Dav2PreprocessContract.OUTPUT_WIDTH + x1] * fx
                output[row * WIDTH + column] = (top * (1 - fy) + bottom * fy).toFloat()
            }
        }
    }

    private class TestLifecycleOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        override val lifecycle: Lifecycle get() = registry
        fun resume() { registry.currentState = Lifecycle.State.RESUMED }
        fun destroy() { registry.currentState = Lifecycle.State.DESTROYED }
    }

    private companion object {
        const val WIDTH = 640
        const val HEIGHT = 480
        const val CAMERA_HEIGHT_M = 1.0341161949454936
        const val REPORT_KEY = "camerax_depth_parity_capture_r0_report"
    }
}
