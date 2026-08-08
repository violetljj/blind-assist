package com.linnan.blindassist.ustrfbenchmark

import android.content.Context
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.security.MessageDigest
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

data class KnownHeightCaptureRequest(
    val form: CaptureFormState,
)

@ExperimentalCamera2Interop
class KnownHeightCaptureEngine(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val onState: (CaptureRunState) -> Unit,
) {
    private val mainHandler = Handler(Looper.getMainLooper())
    private var provider: ProcessCameraProvider? = null
    private var executor: ExecutorService? = null
    private val stopped = AtomicBoolean(false)
    private val terminal = AtomicBoolean(false)
    private var activeRoot: File? = null

    fun start(request: KnownHeightCaptureRequest) {
        check(request.form.canStart)
        if (executor != null) return
        stopped.set(false)
        terminal.set(false)
        emit(CaptureRunState.Preparing("正在生成测量记录并启动相机…"))
        val root = requireNotNull(context.getExternalFilesDir("known-height-phone-shadow"))
            .resolve(request.form.sessionId)
        if (root.exists() && root.listFiles()?.isNotEmpty() == true) {
            emit(CaptureRunState.Hold("Session 目录已存在且非空，请更换 Session ID"))
            return
        }
        root.mkdirs()
        activeRoot = root
        val referenceDirectory = root.resolve("reference").apply { mkdirs() }
        val referenceFile = referenceDirectory.resolve("reference.json")
        try {
            referenceFile.writeText(referenceJson(request.form).toString(2))
        } catch (error: Throwable) {
            referenceFile.delete()
            referenceDirectory.delete()
            root.delete()
            emit(CaptureRunState.Hold("测量记录生成失败：${error.message ?: error.javaClass.simpleName}"))
            return
        }

        val worker = Executors.newSingleThreadExecutor()
        executor = worker
        val cameraFuture = ProcessCameraProvider.getInstance(context)
        cameraFuture.addListener({
            try {
                val cameraProvider = cameraFuture.get(10, TimeUnit.SECONDS)
                provider = cameraProvider
                bindAndCapture(cameraProvider, request, root, referenceFile, worker)
            } catch (error: Throwable) {
                finishHold("相机启动失败：${error.message ?: error.javaClass.simpleName}")
            }
        }, mainHandler::post)
    }

    fun stop() {
        stopped.set(true)
        provider?.unbindAll()
        executor?.shutdownNow()
        executor = null
    }

    fun cancel() {
        stopped.set(true)
        finishHold("用户主动停止采集；部分帧不得进入评价")
    }

    private fun bindAndCapture(
        cameraProvider: ProcessCameraProvider,
        request: KnownHeightCaptureRequest,
        root: File,
        referenceFile: File,
        worker: ExecutorService,
    ) {
        val target = request.form.phase.frameTarget
        val imageDirectory = root.resolve("images").apply { mkdirs() }
        val counter = AtomicInteger()
        val finalized = AtomicBoolean(false)
        val rows = arrayOfNulls<JSONObject>(target)
        val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
                ).build(),
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(worker) { image ->
            val index = counter.getAndIncrement()
            try {
                if (stopped.get() || index >= target) return@setAnalyzer
                val output = imageDirectory.resolve("frame_%06d.png".format(index))
                saveRgbaPng(image.width, image.height, image.planes[0].rowStride, image.planes[0].pixelStride, image.planes[0].buffer, output)
                rows[index] = JSONObject()
                    .put("frame_id", index)
                    .put("capture_timestamp_ns", image.imageInfo.timestamp)
                    .put("rotation_degrees", image.imageInfo.rotationDegrees)
                    .put("width", image.width)
                    .put("height", image.height)
                    .put("rgb_file", "images/${output.name}")
                    .put("rgb_sha256", sha256(output))
                emit(CaptureRunState.Capturing(index + 1, target))
                if (index + 1 == target && finalized.compareAndSet(false, true)) {
                    finishCapture(request, root, referenceFile, rows.map { requireNotNull(it) })
                }
            } catch (error: Throwable) {
                if (finalized.compareAndSet(false, true)) finishHold("第 ${index + 1} 帧保存失败：${error.message ?: error.javaClass.simpleName}")
            } finally {
                image.close()
            }
        }
        cameraProvider.unbindAll()
        val camera = cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        observedCameraId = Camera2CameraInfo.from(camera.cameraInfo).cameraId
        emit(CaptureRunState.Capturing(0, target))
    }

    private var observedCameraId: String = ""

    private fun finishCapture(
        request: KnownHeightCaptureRequest,
        root: File,
        referenceFile: File,
        rows: List<JSONObject>,
    ) {
        val timestamps = rows.map { it.getLong("capture_timestamp_ns") }
        if (timestamps.zipWithNext().any { (left, right) -> right <= left }) {
            finishHold("CameraX capture timestamp 非严格递增")
            return
        }
        if (!terminal.compareAndSet(false, true)) return
        root.resolve("frames.json").writeText(JSONArray(rows).toString(2))
        val intrinsics = observedIntrinsicsReceipt(context.getSystemService(CameraManager::class.java), observedCameraId)
        root.resolve("intrinsics.json").writeText(intrinsics.first)
        val form = request.form
        val receipt = JSONObject()
            .put("schema", "blindassist_known_height_phone_capture_receipt_v1")
            .put("protocol_id", if (form.phase == CapturePhase.DEV) DEVELOPMENT_PROTOCOL_ID else PROTOCOL_ID)
            .put("model_id", MODEL_ID)
            .put("status", if (form.phase == CapturePhase.DEV) "DEVELOPMENT_CAPTURED_CONSUMED_REFERENCE" else "CAPTURED_PENDING_HOST_PREFLIGHT")
            .put("session_id", form.sessionId)
            .put("phase", form.phase.name)
            .put("device_serial", Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID))
            .put("device_model", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
            .put("camera_id", observedCameraId)
            .put("camera_height_m", requireNotNull(form.cameraHeightM))
            .put("camera_height_uncertainty_m", requireNotNull(form.cameraHeightUncertaintyM))
            .put("mount_profile_id", form.mountProfileId.trim())
            .put("intrinsics_sha256", intrinsics.second)
            .put("frame_manifest", "frames.json")
            .put("reference_manifest", "reference/${referenceFile.name}")
            .put("reference_manifest_sha256", sha256(referenceFile))
            .put(
                "authorization",
                JSONObject()
                    .put("development_only", form.phase == CapturePhase.DEV)
                    .put("formal_evaluation", form.phase != CapturePhase.DEV)
                    .put("shadow_capture_only", true)
                    .put("app_runtime", false)
                    .put("production", false),
            )
        root.resolve("receipt.json").writeText(receipt.toString(2))
        mainHandler.post {
            provider?.unbindAll()
            executor?.shutdown()
            executor = null
            onState(CaptureRunState.Complete(root.absolutePath, rows.size))
        }
    }

    private fun finishHold(reason: String) {
        if (!terminal.compareAndSet(false, true)) return
        activeRoot?.let { root ->
            runCatching {
                root.mkdirs()
                root.resolve("capture_hold.json").writeText(
                    JSONObject()
                        .put("schema", "blindassist_known_height_phone_capture_hold_v1")
                        .put("status", "HOLD")
                        .put("reason", reason)
                        .put("authorization", JSONObject().put("evaluation", false).put("app_runtime", false).put("production", false))
                        .toString(2),
                )
            }
        }
        mainHandler.post {
            provider?.unbindAll()
            executor?.shutdownNow()
            executor = null
            onState(CaptureRunState.Hold(reason))
        }
    }

    private fun emit(state: CaptureRunState) = mainHandler.post { onState(state) }

    private fun saveRgbaPng(width: Int, height: Int, rowStride: Int, pixelStride: Int, source: ByteBuffer, output: File) {
        require(pixelStride == 4) { "相机未输出 RGBA_8888" }
        val packed = ByteBuffer.allocate(width * height * 4)
        val row = ByteArray(width * 4)
        val input = source.duplicate()
        repeat(height) { y -> input.position(y * rowStride); input.get(row); packed.put(row) }
        packed.flip()
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(packed)
        FileOutputStream(output).use { check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)) }
        bitmap.recycle()
    }

    private fun observedIntrinsicsReceipt(manager: CameraManager, cameraId: String): Pair<String, String> {
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val receipt = JSONObject()
            .put("camera_id", cameraId)
            .put("intrinsic_calibration", JSONArray(characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.toList() ?: emptyList<Float>()))
            .put("distortion", JSONArray(characteristics.get(CameraCharacteristics.LENS_DISTORTION)?.toList() ?: emptyList<Float>()))
            .put("sensor_orientation_degrees", characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION))
            .put("active_array", characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)?.flattenToString())
        val rendered = receipt.toString()
        return rendered to sha256(rendered.toByteArray())
    }

    private fun sha256(file: File): String = file.inputStream().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        digest.digest().joinToString("") { "%02X".format(it) }
    }
    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02X".format(it) }

    private fun referenceJson(form: CaptureFormState): JSONObject = JSONObject()
        .put("schema", "blindassist_known_height_phone_reference_v1")
        .put("session_id", form.sessionId)
        .put("phase", form.phase.name)
        .put("reference_method", form.measurementMethod.receiptValue)
        .put("instrument", form.measurementMethod.label)
        .put("instrument_error_cm", form.instrumentErrorCm.toDouble())
        .put(
            "camera_height_readings_cm",
            JSONArray(
                if (form.phase == CapturePhase.DEV) listOf(form.heightReading1Cm.toDouble())
                else requireNotNull(form.heightReadingsCm),
            ),
        )
        .put("camera_height_m", requireNotNull(form.cameraHeightM))
        .put("camera_height_uncertainty_m", requireNotNull(form.cameraHeightUncertaintyM))
        .put("reference_points", JSONArray(requireNotNull(form.referencePoints).map { point ->
            JSONObject()
                .put("id", point.id)
                .put("description", point.label)
                .put("measurement_type", form.measurementMethod.receiptValue)
                .put("measured_distance_m", point.distanceM)
        }))
        .put(
            "truth_firewall",
            if (form.phase == CapturePhase.DEV) "DEVELOPMENT_LABEL_ONLY_NOT_FORMAL_GROUND_TRUTH"
            else "OFFLINE_EVALUATOR_ONLY_NOT_MODEL_INPUT",
        )

    private companion object {
        const val PROTOCOL_ID = "KNOWN_HEIGHT_PHONE_SHADOW_P0_R2_20260804"
        const val DEVELOPMENT_PROTOCOL_ID = "KNOWN_HEIGHT_PHONE_DEVELOPMENT_CAPTURE_R0"
        const val MODEL_ID = "CAMERA_CONDITIONED_SCALE_STUDENT_R0_FINAL_5P"
    }
}
