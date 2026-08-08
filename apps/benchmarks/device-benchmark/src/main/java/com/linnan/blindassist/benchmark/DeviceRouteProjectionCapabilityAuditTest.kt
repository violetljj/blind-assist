package com.linnan.blindassist.benchmark

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class DeviceRouteProjectionCapabilityAuditTest {
    @Test
    fun auditRealCameraCalibrationAndRotationVectorStream() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val backCameras = cameraManager.cameraIdList.mapNotNull { id ->
            val characteristics = cameraManager.getCameraCharacteristics(id)
            if (characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK) {
                id to characteristics
            } else null
        }
        assertTrue("no back camera", backCameras.isNotEmpty())

        val cameraRows = JSONArray()
        var projectionCandidateCount = 0
        backCameras.forEach { (id, characteristics) ->
            val pixelSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            val active = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            val physical = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            val intrinsics = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            val focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val distortion = characteristics.get(CameraCharacteristics.LENS_DISTORTION)
            val derivedAvailable = pixelSize != null && physical != null &&
                physical.width > 0f && physical.height > 0f &&
                focalLengths != null && focalLengths.isNotEmpty()
            if (intrinsics?.size ?: 0 >= 5 || derivedAvailable) projectionCandidateCount += 1
            cameraRows.put(JSONObject().apply {
                put("camera_id", id)
                put("hardware_level", characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL))
                put("sensor_orientation_degrees", characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION))
                put("pixel_array", pixelSize?.let { JSONArray(listOf(it.width, it.height)) })
                put("active_array", active?.let { JSONArray(listOf(it.left, it.top, it.right, it.bottom)) })
                put("physical_size_mm", physical?.let { JSONArray(listOf(it.width, it.height)) })
                put("intrinsic_calibration", intrinsics?.let { JSONArray(it.toList()) })
                put("focal_lengths_mm", focalLengths?.let { JSONArray(it.toList()) })
                put("distortion", distortion?.let { JSONArray(it.toList()) })
                put("projection_input_mode", when {
                    intrinsics?.size ?: 0 >= 5 -> "camera_characteristics_intrinsic_calibration"
                    derivedAvailable -> "derived_from_focal_physical_and_pixel_size"
                    else -> "unavailable"
                })
                if (intrinsics == null && derivedAvailable) {
                    val focal = focalLengths!!.first().toDouble()
                    put("derived_intrinsics_px", JSONArray(listOf(
                        focal / physical!!.width * pixelSize!!.width,
                        focal / physical.height * pixelSize.height,
                        pixelSize.width / 2.0,
                        pixelSize.height / 2.0
                    )))
                }
            })
        }
        assertTrue("no camera projection candidate", projectionCandidateCount > 0)

        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        assertNotNull("rotation-vector sensor unavailable", rotationSensor)
        val samples = mutableListOf<Pair<Long, FloatArray>>()
        val thread = HandlerThread("r829-rotation-vector-audit").apply { start() }
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                synchronized(samples) { samples += event.timestamp to event.values.copyOf() }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }
        val registered = sensorManager.registerListener(
            listener, rotationSensor, SensorManager.SENSOR_DELAY_GAME, Handler(thread.looper)
        )
        assertTrue("rotation-vector listener registration failed", registered)
        try {
            Thread.sleep(2_500)
        } finally {
            sensorManager.unregisterListener(listener)
            thread.quitSafely()
            thread.join(2_000)
        }
        val frozenSamples = synchronized(samples) { samples.toList() }
        assertTrue("too few rotation-vector samples: ${frozenSamples.size}", frozenSamples.size >= 10)

        val quaternions = frozenSamples.map { (_, vector) ->
            FloatArray(4).also { SensorManager.getQuaternionFromVector(it, vector) }
        }
        val normErrors = quaternions.map { q -> abs(sqrt(q.sumOf { it.toDouble() * it }) - 1.0) }
        assertTrue("invalid quaternion norm", normErrors.maxOrNull()!! <= 0.05)
        val reference = quaternions.first()
        val anglesDegrees = quaternions.map { q ->
            val dot = abs(reference.indices.sumOf { reference[it].toDouble() * q[it] }).coerceIn(0.0, 1.0)
            Math.toDegrees(2.0 * acos(dot))
        }
        val intervalsMs = frozenSamples.zipWithNext { a, b -> (b.first - a.first) / 1_000_000.0 }.sorted()
        val medianIntervalMs = intervalsMs[intervalsMs.size / 2]

        val report = JSONObject().apply {
            put("schema", "blindassist_device_route_projection_capability_audit_v1")
            put("camera", JSONObject().apply {
                put("back_camera_count", backCameras.size)
                put("projection_candidate_count", projectionCandidateCount)
                put("rows", cameraRows)
            })
            put("rotation_vector", JSONObject().apply {
                put("name", rotationSensor!!.name)
                put("vendor", rotationSensor.vendor)
                put("version", rotationSensor.version)
                put("sample_count", frozenSamples.size)
                put("duration_ms", (frozenSamples.last().first - frozenSamples.first().first) / 1_000_000.0)
                put("median_interval_ms", medianIntervalMs)
                put("maximum_quaternion_norm_error", normErrors.maxOrNull())
                put("median_angle_from_first_degrees", anglesDegrees.sorted()[anglesDegrees.size / 2])
                put("maximum_angle_from_first_degrees", anglesDegrees.maxOrNull())
                put("stability_interpretation", "observed_only_device_may_have_moved")
            })
            put("checks", JSONObject().apply {
                put("back_camera_present", backCameras.isNotEmpty())
                put("projection_input_available", projectionCandidateCount > 0)
                put("rotation_vector_present", true)
                put("minimum_rotation_samples", frozenSamples.size >= 10)
                put("quaternion_norm_valid", normErrors.maxOrNull()!! <= 0.05)
            })
            put("authorization", JSONObject().apply {
                put("benchmark_only", true)
                put("real_projection_accuracy_validated", false)
                put("app_runtime_authorized", false)
                put("production_authorized", false)
            })
        }
        InstrumentationRegistry.getInstrumentation().sendStatus(
            2,
            Bundle().apply { putString("r829_report", report.toString()) }
        )
    }
}
