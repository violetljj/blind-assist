package com.linnan.blindassist.npu

import android.content.Context
import com.linnan.blindassist.vision.DetectorExecutionBackend
import com.linnan.blindassist.vision.ObjectDetector
import com.linnan.blindassist.vision.ObjectDetectorProvider
import com.linnan.blindassist.vision.TfliteYoloDetector
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.Interpreter

/**
 * Candidate-only QNN HTP provider. Any capability, delegate, or graph creation
 * failure is surfaced by TfliteYoloDetector as NOT_READY; there is no CPU fallback.
 */
class QnnHtpObjectDetectorProvider : ObjectDetectorProvider {
    override fun create(context: Context): ObjectDetector {
        var delegate: QnnDelegate? = null
        return TfliteYoloDetector(
            context = context,
            executionBackend = DetectorExecutionBackend.QUALCOMM_QNN_HTP,
            externalInterpreterOptionsFactory = {
                System.loadLibrary("cdsprpc")
                require(
                    QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
                ) { "QNN HTP FP16 capability is unavailable" }
                val qnnOptions = QnnDelegate.Options().apply {
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
                val qnnDelegate = QnnDelegate(qnnOptions)
                require(qnnDelegate.isAvailable) { "QNN HTP delegate is unavailable" }
                delegate = qnnDelegate
                Interpreter.Options()
                    .setNumThreads(CPU_THREADS)
                    .addDelegate(qnnDelegate)
            },
            externalBackendCloser = {
                delegate?.close()
                delegate = null
            }
        )
    }

    private companion object {
        const val CPU_THREADS = 4
        const val MODEL_TOKEN = "blindassist_yolo11n_fp16_320_qnn_2_47_candidate_v1"
    }
}
