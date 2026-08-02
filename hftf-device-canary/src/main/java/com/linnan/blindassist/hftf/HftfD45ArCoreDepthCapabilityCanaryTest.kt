package com.linnan.blindassist.hftf

import android.os.Build
import android.os.SystemClock
import android.util.AtomicFile
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.linnan.blindassist.hftf.metricdepth.MetricDepthHistorySolverConfig
import com.linnan.blindassist.hftf.metricdepth.MetricDepthTargetSamplerConfig
import java.io.File
import java.util.EnumSet
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * D45 source-only capability canary.
 *
 * This test never resumes an ARCore session, opens the camera, requests an ARCore installation,
 * or touches the BlindAssist decision pipeline. It only emits a device capability receipt that
 * determines whether the separately frozen metric-depth measurement canary is executable.
 */
@RunWith(AndroidJUnit4::class)
class HftfD45ArCoreDepthCapabilityCanaryTest {
    @Test
    fun emitArCoreDepthCapabilityReceiptWithoutOpeningCamera() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val probeContext = instrumentation.context
        val targetContext = instrumentation.targetContext
        assertFrozenSourceContract()
        val startedAtNs = SystemClock.elapsedRealtimeNanos()
        val availability = settleAvailability(probeContext)

        var supportsAutomatic = false
        var supportsRaw = false
        var cameraConfigCount = 0
        var hardwareDepthCameraConfigCount = 0
        var errorType: String? = null
        var errorMessage: String? = null
        if (availability == ArCoreApk.Availability.SUPPORTED_INSTALLED) {
            var session: Session? = null
            try {
                session = Session(probeContext)
                supportsAutomatic = session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
                supportsRaw = session.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)
                val cameraConfigs = session.getSupportedCameraConfigs(
                    CameraConfigFilter(session).setDepthSensorUsage(
                        EnumSet.allOf(CameraConfig.DepthSensorUsage::class.java)
                    )
                )
                cameraConfigCount = cameraConfigs.size
                hardwareDepthCameraConfigCount = cameraConfigs.count {
                    it.depthSensorUsage == CameraConfig.DepthSensorUsage.REQUIRE_AND_USE
                }
            } catch (error: Exception) {
                errorType = error.javaClass.name
                errorMessage = error.message
            } finally {
                session?.close()
            }
        }

        val status = classify(
            availability = availability,
            supportsAutomatic = supportsAutomatic,
            supportsRaw = supportsRaw,
            hardwareDepthCameraConfigCount = hardwareDepthCameraConfigCount,
            errorType = errorType
        )
        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startedAtNs) / 1_000_000L
        val receipt = canonicalJson(
            status = status,
            availability = availability,
            supportsAutomatic = supportsAutomatic,
            supportsRaw = supportsRaw,
            cameraConfigCount = cameraConfigCount,
            hardwareDepthCameraConfigCount = hardwareDepthCameraConfigCount,
            elapsedMs = elapsedMs,
            errorType = errorType,
            errorMessage = errorMessage
        )
        val outputDirectory = requireNotNull(targetContext.getExternalFilesDir(OUTPUT_DIRECTORY))
        assertTrue(outputDirectory.exists() || outputDirectory.mkdirs())
        val receiptFile = File(outputDirectory, OUTPUT_FILE)
        val atomicReceipt = AtomicFile(receiptFile)
        val stream = atomicReceipt.startWrite()
        try {
            stream.write(receipt.toByteArray(Charsets.UTF_8))
            atomicReceipt.finishWrite(stream)
        } catch (error: Throwable) {
            atomicReceipt.failWrite(stream)
            throw error
        }
        Log.i(LOG_TAG, "receipt=${receiptFile.absolutePath} $receipt")

        assertTrue(receiptFile.isFile)
        assertTrue(receiptFile.readText(Charsets.UTF_8) == receipt)
    }

    private fun assertFrozenSourceContract() {
        val sampler = MetricDepthTargetSamplerConfig()
        assertTrue(sampler.innerCropRatio == 0.60f)
        assertTrue(sampler.minimumDepthMeters == 0.20f)
        assertTrue(sampler.maximumDepthMeters == 20.0f)
        assertTrue(sampler.minimumValidSamples == 12)
        assertTrue(sampler.minimumCoverage == 0.25f)
        assertTrue(sampler.maximumReceiptAgeNs == 150_000_000L)
        val solver = MetricDepthHistorySolverConfig()
        assertTrue(solver.historySize == 7)
        assertTrue(solver.forecastHorizonNs == 1_000_000_000L)
    }

    private fun settleAvailability(context: android.content.Context): ArCoreApk.Availability {
        var availability = ArCoreApk.getInstance().checkAvailability(context)
        repeat(MAX_AVAILABILITY_POLLS - 1) {
            if (!availability.isTransient) return availability
            SystemClock.sleep(AVAILABILITY_POLL_MS)
            availability = ArCoreApk.getInstance().checkAvailability(context)
        }
        return availability
    }

    private fun classify(
        availability: ArCoreApk.Availability,
        supportsAutomatic: Boolean,
        supportsRaw: Boolean,
        hardwareDepthCameraConfigCount: Int,
        errorType: String?
    ): String {
        if (errorType != null) return "PROBE_ERROR"
        return when (availability) {
            ArCoreApk.Availability.SUPPORTED_INSTALLED -> when {
                supportsAutomatic && hardwareDepthCameraConfigCount > 0 ->
                    "READY_AUTOMATIC_HARDWARE_DEPTH"
                supportsAutomatic -> "READY_AUTOMATIC_ESTIMATED_DEPTH"
                supportsRaw -> "READY_RAW_DEPTH_REGISTRATION_REQUIRED"
                else -> "DEPTH_UNSUPPORTED"
            }
            ArCoreApk.Availability.SUPPORTED_NOT_INSTALLED -> "ARCORE_NOT_INSTALLED"
            ArCoreApk.Availability.SUPPORTED_APK_TOO_OLD -> "ARCORE_APK_TOO_OLD"
            ArCoreApk.Availability.UNSUPPORTED_DEVICE_NOT_CAPABLE -> "DEVICE_UNSUPPORTED"
            ArCoreApk.Availability.UNKNOWN_CHECKING,
            ArCoreApk.Availability.UNKNOWN_TIMED_OUT -> "AVAILABILITY_UNRESOLVED"
            ArCoreApk.Availability.UNKNOWN_ERROR -> "AVAILABILITY_ERROR"
        }
    }

    private fun canonicalJson(
        status: String,
        availability: ArCoreApk.Availability,
        supportsAutomatic: Boolean,
        supportsRaw: Boolean,
        cameraConfigCount: Int,
        hardwareDepthCameraConfigCount: Int,
        elapsedMs: Long,
        errorType: String?,
        errorMessage: String?
    ): String = buildString {
        append("{\n")
        append("  \"schema_version\": \"hftf-d45-arcore-depth-capability-r0\",\n")
        append("  \"status\": \"").append(jsonEscape(status)).append("\",\n")
        append("  \"arcore_sdk_version\": \"1.33.0\",\n")
        append("  \"availability\": \"").append(jsonEscape(availability.name)).append("\",\n")
        append("  \"supports_automatic_depth\": ").append(supportsAutomatic).append(",\n")
        append("  \"supports_raw_depth\": ").append(supportsRaw).append(",\n")
        append("  \"camera_config_count\": ").append(cameraConfigCount).append(",\n")
        append("  \"hardware_depth_camera_config_count\": ")
            .append(hardwareDepthCameraConfigCount).append(",\n")
        append("  \"probe_elapsed_ms\": ").append(elapsedMs).append(",\n")
        append("  \"camera_opened\": false,\n")
        append("  \"arcore_install_requested\": false,\n")
        append("  \"risk_or_feedback_invoked\": false,\n")
        append("  \"device\": {\n")
        append("    \"manufacturer\": \"").append(jsonEscape(Build.MANUFACTURER)).append("\",\n")
        append("    \"model\": \"").append(jsonEscape(Build.MODEL)).append("\",\n")
        append("    \"android_sdk\": ").append(Build.VERSION.SDK_INT).append("\n")
        append("  },\n")
        append("  \"error_type\": ").append(jsonStringOrNull(errorType)).append(",\n")
        append("  \"error_message\": ").append(jsonStringOrNull(errorMessage)).append("\n")
        append("}\n")
    }

    private fun jsonStringOrNull(value: String?): String =
        value?.let { "\"${jsonEscape(it)}\"" } ?: "null"

    private fun jsonEscape(value: String): String = buildString(value.length) {
        value.forEach { character ->
            when (character) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                else -> {
                    if (character.code < 0x20) {
                        append("\\u")
                        append(character.code.toString(16).padStart(4, '0'))
                    } else {
                        append(character)
                    }
                }
            }
        }
    }

    private companion object {
        const val LOG_TAG = "BlindAssistHftfD45"
        const val OUTPUT_DIRECTORY = "hftf-d45"
        const val OUTPUT_FILE = "arcore-depth-capability-r0.json"
        const val MAX_AVAILABILITY_POLLS = 10
        const val AVAILABILITY_POLL_MS = 200L
    }
}
