package com.linnan.blindassist.benchmark

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class DeviceCameraPermissionedCalibrationAuditTest {
    @Test
    fun auditPermissionedCameraCalibrationPoseAndTimestampSource() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val cameraPermission = context.checkSelfPermission(Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        assertTrue("target app CAMERA permission is not granted", cameraPermission)
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val rows = JSONArray()
        var backCameraCount = 0
        var exactIntrinsicCount = 0
        var distortionCount = 0
        var poseRotationCount = 0
        var gyroscopeReferenceCount = 0
        manager.cameraIdList.forEach { id ->
            val c = manager.getCameraCharacteristics(id)
            if (c.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_BACK) return@forEach
            backCameraCount += 1
            val intrinsic = c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            val distortion = c.get(CameraCharacteristics.LENS_DISTORTION)
            val poseRotation = c.get(CameraCharacteristics.LENS_POSE_ROTATION)
            val poseTranslation = c.get(CameraCharacteristics.LENS_POSE_TRANSLATION)
            val poseReference = c.get(CameraCharacteristics.LENS_POSE_REFERENCE)
            if ((intrinsic?.size ?: 0) >= 5) exactIntrinsicCount += 1
            if (distortion != null && distortion.isNotEmpty()) distortionCount += 1
            if ((poseRotation?.size ?: 0) >= 4) poseRotationCount += 1
            if (poseReference == CameraCharacteristics.LENS_POSE_REFERENCE_GYROSCOPE) gyroscopeReferenceCount += 1
            rows.put(JSONObject().apply {
                put("camera_id", id)
                put("sensor_orientation_degrees", c.get(CameraCharacteristics.SENSOR_ORIENTATION))
                put("timestamp_source", c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE))
                put("sync_max_latency", c.get(CameraCharacteristics.SYNC_MAX_LATENCY))
                put("pre_correction_active_array", c.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)?.let {
                    JSONArray(listOf(it.left, it.top, it.right, it.bottom))
                })
                put("intrinsic_calibration", intrinsic?.let { JSONArray(it.toList()) })
                put("distortion", distortion?.let { JSONArray(it.toList()) })
                put("lens_pose_rotation_xyzw", poseRotation?.let { JSONArray(it.toList()) })
                put("lens_pose_translation_m", poseTranslation?.let { JSONArray(it.toList()) })
                put("lens_pose_reference", poseReference)
            })
        }
        assertTrue("no back camera", backCameraCount > 0)
        val permissionKeys = manager.getCameraCharacteristics(manager.cameraIdList.first())
            .keysNeedingPermission.map { it.name }.sorted()
        val report = JSONObject().apply {
            put("schema", "blindassist_permissioned_camera_calibration_pose_audit_v1")
            put("target_package", context.packageName)
            put("camera_permission_granted", cameraPermission)
            put("keys_needing_permission", JSONArray(permissionKeys))
            put("summary", JSONObject().apply {
                put("back_camera_count", backCameraCount)
                put("exact_intrinsic_camera_count", exactIntrinsicCount)
                put("distortion_camera_count", distortionCount)
                put("pose_rotation_camera_count", poseRotationCount)
                put("gyroscope_reference_camera_count", gyroscopeReferenceCount)
            })
            put("cameras", rows)
            put("authorization", JSONObject().apply {
                put("benchmark_only", true)
                put("device_to_camera_transform_validated", false)
                put("real_reprojection_accuracy_validated", false)
                put("app_runtime_authorized", false)
                put("production_authorized", false)
            })
        }
        instrumentation.sendStatus(2, Bundle().apply { putString("r830_report", report.toString()) })
    }
}
