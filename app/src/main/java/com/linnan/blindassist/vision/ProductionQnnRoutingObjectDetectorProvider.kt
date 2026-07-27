package com.linnan.blindassist.vision

import android.content.Context
import android.os.Build
import android.util.Log
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.Interpreter

/**
 * Production device-capability route.
 *
 * QNN is preferred only inside the frozen SM8650/arm64 scope and after a live
 * HTP FP16 capability probe. Every other path returns the ordinary CPU detector.
 */
class ProductionQnnRoutingObjectDetectorProvider : ObjectDetectorProvider {
    override fun create(context: Context): ObjectDetector {
        val profile = ProductionDeviceProfile(
            socModel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                Build.SOC_MODEL.orEmpty()
            } else {
                ""
            },
            supportedAbis = Build.SUPPORTED_ABIS.orEmpty().toList()
        )
        if (!ProductionDetectorRoutePolicy.isQnnProbeEligible(profile)) {
            return cpu(context, ProductionDetectorRoutePolicy.decide(profile, false).reason)
        }

        var delegate: QnnDelegate? = null
        return try {
            System.loadLibrary("cdsprpc")
            val capabilityAvailable =
                QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
            val route = ProductionDetectorRoutePolicy.decide(profile, capabilityAvailable)
            if (route.backend != DetectorExecutionBackend.QUALCOMM_QNN_HTP) {
                return cpu(context, route.reason)
            }

            val detector = TfliteYoloDetector(
                context = context,
                executionBackend = DetectorExecutionBackend.QUALCOMM_QNN_HTP,
                externalInterpreterOptionsFactory = {
                    val options = QnnDelegate.Options().apply {
                        setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                        setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
                        setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
                        setHtpPerformanceMode(
                            QnnDelegate.Options.HtpPerformanceMode
                                .HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE
                        )
                        setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
                        setCacheDir(context.codeCacheDir.absolutePath)
                        setModelToken(MODEL_TOKEN)
                    }
                    val created = QnnDelegate(options)
                    require(created.isAvailable) { "QNN HTP delegate is unavailable" }
                    delegate = created
                    Interpreter.Options()
                        .setNumThreads(CPU_THREADS)
                        .addDelegate(created)
                },
                externalBackendCloser = {
                    delegate?.close()
                    delegate = null
                }
            )
            if (detector.isReady) {
                Log.i(TAG, "route=qualcomm_qnn_htp reason=${route.reason} soc=${profile.socModel}")
                detector
            } else {
                val reason = "qnn_detector_initialization_failed:${detector.statusMessage}"
                detector.close()
                cpu(context, reason)
            }
        } catch (error: Throwable) {
            if (error is VirtualMachineError || error is ThreadDeath) {
                throw error
            }
            try {
                delegate?.close()
            } catch (closeError: Throwable) {
                if (closeError is VirtualMachineError || closeError is ThreadDeath) {
                    throw closeError
                }
                Log.w(TAG, "QNN delegate cleanup failed before CPU fallback.", closeError)
            }
            delegate = null
            cpu(context, "qnn_route_exception:${error.javaClass.simpleName}:${error.message}")
        }
    }

    private fun cpu(context: Context, reason: String): ObjectDetector {
        Log.w(TAG, "route=cpu_xnnpack reason=$reason")
        return TfliteYoloDetector(
            context = context,
            executionBackend = DetectorExecutionBackend.CPU_XNNPACK
        )
    }

    private companion object {
        const val TAG = "ProductionDetectorRoute"
        const val CPU_THREADS = 4
        const val MODEL_TOKEN = "blindassist_yolo11n_fp16_320_qnn_2_47_production_v1"
    }
}
