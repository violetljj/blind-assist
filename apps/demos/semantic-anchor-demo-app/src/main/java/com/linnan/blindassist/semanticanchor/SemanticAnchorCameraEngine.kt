package com.linnan.blindassist.semanticanchor

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.vision.barcode.BarcodeScanner
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

@androidx.annotation.OptIn(markerClass = [ExperimentalCamera2Interop::class])
internal class SemanticAnchorCameraEngine(
    context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    initialTarget: AnchorTarget,
    private val onObservation: (AnchorObservation) -> Unit,
    private val onStatus: (String) -> Unit,
) {
    private val appContext = context.applicationContext
    private val mainExecutor = ContextCompat.getMainExecutor(appContext)
    private val analyzerExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val target = AtomicReference(initialTarget)
    private val processing = AtomicBoolean(false)
    private val active = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private val replayPaused = AtomicBoolean(false)
    private val lensCalibration = AtomicReference<LensCalibration?>(null)
    private val activeFocalLengthMm = AtomicReference<Double?>(null)
    private val barcodeScanner: BarcodeScanner = BarcodeScanning.getClient(
        BarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_QR_CODE).build(),
    )
    private val textRecognizer: TextRecognizer =
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null

    fun updateTarget(value: AnchorTarget) {
        target.set(value)
    }

    fun setReplayPaused(paused: Boolean) {
        replayPaused.set(paused)
    }

    fun start() {
        check(!closed.get()) { "camera engine is already closed" }
        if (!active.compareAndSet(false, true)) return
        onStatus("正在启动后置相机…")
        val future = ProcessCameraProvider.getInstance(appContext)
        future.addListener({
            try {
                if (!active.get()) return@addListener
                val cameraProvider = future.get(10, TimeUnit.SECONDS)
                val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                val analysisBuilder = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder().setResolutionStrategy(
                            ResolutionStrategy(
                                Size(1280, 720),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                            ),
                        ).build(),
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                Camera2Interop.Extender(analysisBuilder).setSessionCaptureCallback(
                    object : CameraCaptureSession.CaptureCallback() {
                        override fun onCaptureCompleted(
                            session: CameraCaptureSession,
                            request: CaptureRequest,
                            result: TotalCaptureResult,
                        ) {
                            result.get(android.hardware.camera2.CaptureResult.LENS_FOCAL_LENGTH)
                                ?.toDouble()
                                ?.takeIf { it > 0.0 }
                                ?.let(activeFocalLengthMm::set)
                        }
                    },
                )
                val imageAnalysis = analysisBuilder.build()
                imageAnalysis.setAnalyzer(analyzerExecutor, ::analyze)
                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis,
                )
                lensCalibration.set(readLensCalibration(camera.cameraInfo))
                provider = cameraProvider
                analysis = imageAnalysis
                onStatus("LIVE · 等待独立语义证据")
            } catch (error: Throwable) {
                active.set(false)
                onStatus("CameraX 启动失败：${error.message ?: error.javaClass.simpleName}")
            }
        }, mainExecutor)
    }

    fun shutdown() {
        if (!closed.compareAndSet(false, true)) return
        active.set(false)
        analysis?.clearAnalyzer()
        analysis = null
        provider?.unbindAll()
        barcodeScanner.close()
        textRecognizer.close()
        analyzerExecutor.shutdownNow()
    }

    private fun analyze(imageProxy: ImageProxy) {
        if (!active.get() || replayPaused.get() || !processing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            processing.set(false)
            imageProxy.close()
            return
        }
        val input = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
        val snapshot = target.get()
        when (snapshot.mode) {
            AnchorMode.MARKER -> barcodeScanner.process(input)
                .addOnSuccessListener { barcodes ->
                    val intrinsics = lensCalibration.get()?.intrinsicsFor(
                        imageProxy.width,
                        imageProxy.height,
                        imageProxy.imageInfo.rotationDegrees,
                        activeFocalLengthMm.get(),
                    )
                    val poses = if (intrinsics == null) emptyList() else barcodes.mapNotNull { barcode ->
                        val payload = barcode.rawValue ?: return@mapNotNull null
                        val corners = barcode.cornerPoints?.takeIf { it.size == 4 } ?: return@mapNotNull null
                        SquareMarkerPoseSolver.solve(
                            payload = payload,
                            corners = corners.map { PixelPoint(it.x.toDouble(), it.y.toDouble()) },
                            intrinsics = intrinsics,
                            markerSizeMeters = MARKER_SIZE_METERS,
                            standoffMeters = STANDOFF_METERS,
                        )
                    }
                    emit(barcodes.mapNotNull { it.rawValue }, "LIVE QR + PnP", poses)
                }
                .addOnFailureListener(::emitFailure)
                .addOnCompleteListener { finishFrame(imageProxy) }
            AnchorMode.OCR -> textRecognizer.process(input)
                .addOnSuccessListener { text ->
                    emit(text.text.takeIf(String::isNotBlank)?.let(::listOf).orEmpty(), "LIVE OCR")
                }
                .addOnFailureListener(::emitFailure)
                .addOnCompleteListener { finishFrame(imageProxy) }
        }
    }

    private fun emit(
        candidates: List<String>,
        source: String,
        markerPoses: List<MarkerPoseEstimate> = emptyList(),
    ) {
        if (active.get() && !replayPaused.get()) {
            mainExecutor.execute { onObservation(AnchorObservation(candidates, source, markerPoses)) }
        }
    }

    private fun readLensCalibration(cameraInfo: androidx.camera.core.CameraInfo): LensCalibration? {
        val cameraId = Camera2CameraInfo.from(cameraInfo).cameraId
        val manager = appContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val focal = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
            ?: return null
        val sensor = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null
        return LensCalibration(focal.toDouble(), sensor.width.toDouble(), sensor.height.toDouble())
    }

    private fun emitFailure(error: Exception) {
        if (active.get()) mainExecutor.execute {
            onStatus("识别帧失败：${error.message ?: error.javaClass.simpleName}")
        }
    }

    private fun finishFrame(imageProxy: ImageProxy) {
        imageProxy.close()
        processing.set(false)
    }

    private data class LensCalibration(
        val focalLengthMm: Double,
        val sensorWidthMm: Double,
        val sensorHeightMm: Double,
    ) {
        fun intrinsicsFor(
            rawWidth: Int,
            rawHeight: Int,
            rotationDegrees: Int,
            activeFocalLengthMm: Double?,
        ): CameraIntrinsics {
            val rawAspect = rawWidth.toDouble() / rawHeight
            val sensorAspect = sensorWidthMm / sensorHeightMm
            val focalMm = activeFocalLengthMm ?: focalLengthMm
            val focalPixels = if (rawAspect >= sensorAspect) {
                focalMm / sensorWidthMm * rawWidth
            } else {
                focalMm / sensorHeightMm * rawHeight
            }
            val rotated = rotationDegrees % 180 != 0
            val width = if (rotated) rawHeight else rawWidth
            val height = if (rotated) rawWidth else rawHeight
            return CameraIntrinsics(focalPixels, focalPixels, width / 2.0, height / 2.0)
        }
    }

    private companion object {
        const val MARKER_SIZE_METERS = 0.16
        const val STANDOFF_METERS = 0.65
    }
}
