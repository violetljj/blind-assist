package com.linnan.blindassist

import android.app.Application
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.capture.CnhCaptureDecoder
import com.linnan.blindassist.capture.ImuCaptureDecoder
import com.linnan.blindassist.capture.ImuCaptureFrame
import com.linnan.blindassist.vision.AtomRgbGeometry
import com.linnan.blindassist.vision.Dav2DepthRuntimeFactory
import com.linnan.blindassist.vision.MobileFusionCamera
import com.linnan.blindassist.vision.UniDepthRuntimeFactory
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Offline replay of one recorded glasses capture through both RGB depth paths on the phone (direction A App
 * integration, 2026-11-18, no glasses hardware attached). Per ToF frame: the newest Atom JPEG and IMU record received
 * within 100 ms before it (the live MobileFusionInputBuffer rule), demo-rules geometry (pitch-0 extrinsic), default
 * height 1.35 m. Writes one JSON line per ToF frame with anchors, alarms, in-box distances and timings for DAV2-S
 * (current method) and UniDepth-S x one scale. Engineering replay; the capture has no labels used here. */
@RunWith(AndroidJUnit4::class)
class FusionReplayDeviceTest {
    @Test
    fun replayBothDepthPathsOnRecordedCapture() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as Application
        val args = InstrumentationRegistry.getArguments()
        val capture = File(args.getString("captureRoot") ?: File(context.filesDir, "fusion-replay").absolutePath)
        val models = File(context.noBackupFilesDir, "fusion-experiment")
        val height = (args.getString("heightM") ?: "1.35").toDouble()
        val camera = MobileFusionCamera(models)
        val calibration = camera.calibration.copy(cameraFromTof = MobileFusionDemoRules.withoutPitch(camera.calibration.cameraFromTof))
        val gDav2 = AtomRgbGeometry(480, 640, calibration)
        val gMetric = AtomRgbGeometry(480, 640, calibration)
        // RGB receipts.
        val rgb = File(capture, "rgb.jsonl").readLines().filter { it.isNotBlank() }.map { JSONObject(it) }
            .map { it.getLong("received_elapsed_ns") to it.getString("file") }
        // IMU records, decoded with the live decoder, timestamped by the recorded receipt.
        fun receipts(name: String) = File(capture, name).readLines().filter { it.isNotBlank() }.map { JSONObject(it).getLong("received_elapsed_ns") }
        val imuDecoder = ImuCaptureDecoder()
        val imu = File(capture, "imu.raw.jsonl").readLines().zip(receipts("imu.index.jsonl")).mapNotNull { (line, ns) ->
            imuDecoder.feed((line + "\n").toByteArray(), ns).firstOrNull()?.frame?.let { ns to it }
        }
        val tofDecoder = CnhCaptureDecoder()
        val tof = File(capture, "tof.raw.jsonl").readLines().zip(receipts("tof.index.jsonl")).mapNotNull { (line, ns) ->
            tofDecoder.feed((line + "\n").toByteArray(), ns).firstOrNull()?.frame?.let { ns to it }
        }
        val out = File(capture, "replay_${System.currentTimeMillis()}.jsonl")
        var rows = 0
        Dav2DepthRuntimeFactory.create(app, File(models, "dav2s_686x518_fp32.dlc")).use { dav2 ->
            UniDepthRuntimeFactory.create(app, models).use { ud ->
                out.bufferedWriter().use { writer ->
                    for ((ns, frame) in tof) {
                        val image = rgb.lastOrNull { ns - it.first in 0..100_000_000L }
                        val imuFrame: ImuCaptureFrame? = imu.lastOrNull { ns - it.first in 0..100_000_000L }?.second
                        val pose = imuFrame?.let { camera.attitude(it) }
                        val row = JSONObject().put("tof_receipt_ns", ns).put("tof_seq", frame.sequence)
                            .put("rgb", image?.second ?: JSONObject.NULL).put("pose_ok", pose != null)
                        if (image != null && pose != null) {
                            val dist = frame.distanceMm.map(Int::toDouble).toDoubleArray(); val status = frame.targetStatus.toIntArray()
                            val raw = requireNotNull(BitmapFactory.decodeFile(File(capture, image.second).absolutePath))
                            val upright = try { camera.undistort(raw) } finally { raw.recycle() }
                            var t = SystemClock.elapsedRealtimeNanos()
                            val disparity = dav2.estimate(upright.bitmap)
                            val dav2Ms = (SystemClock.elapsedRealtimeNanos() - t) / 1e6
                            val a = gDav2.process(disparity, upright.valid, dist, status, pose.first, pose.second, height)
                            t = SystemClock.elapsedRealtimeNanos()
                            val z = ud.estimate(upright.bitmap)
                            val udMs = (SystemClock.elapsedRealtimeNanos() - t) / 1e6
                            t = SystemClock.elapsedRealtimeNanos()
                            val b = gMetric.processMetric(z, upright.valid, dist, status, pose.first, pose.second, height)
                            val chainMs = (SystemClock.elapsedRealtimeNanos() - t) / 1e6
                            upright.bitmap.recycle()
                            val tofQ = gDav2.tofQueryDepth(dist, status, pose.first, pose.second)
                            fun put(prefix: String, r: AtomRgbGeometry.Result) = row.put("${prefix}_anchor", r.anchor.name)
                                .put("${prefix}_a", r.fit?.a ?: JSONObject.NULL).put("${prefix}_b", r.fit?.b ?: JSONObject.NULL)
                                .put("${prefix}_head", r.head.alarm).put("${prefix}_body", r.body.alarm)
                                .put("${prefix}_head_m", r.head.medianDepthM.takeIf { it.isFinite() } ?: JSONObject.NULL)
                                .put("${prefix}_body_m", r.body.medianDepthM.takeIf { it.isFinite() } ?: JSONObject.NULL)
                            put("dav2", a); put("ud", b)
                            row.put("pitch_deg", pose.first).put("roll_deg", pose.second)
                                .put("tof_valid_zones", status.count { it in setOf(5, 6, 9, 10) })
                                .put("tof_head_m", tofQ.first.takeIf { it.isFinite() } ?: JSONObject.NULL)
                                .put("tof_body_m", tofQ.second.takeIf { it.isFinite() } ?: JSONObject.NULL)
                                .put("dav2_ms", dav2Ms).put("ud_ms", udMs).put("ud_chain_ms", chainMs)
                                .put("ud_depth_median_m", z.filter { it.isFinite() }.sorted().let { it[it.size / 2] })
                            rows++
                        }
                        writer.write(row.toString()); writer.newLine()
                    }
                }
            }
        }
        // Also copy to app-specific external storage so a non-debuggable (demo, AOT) install can be read back with adb.
        context.getExternalFilesDir(null)?.let { ext -> out.copyTo(File(ext, out.name), overwrite = true) }
        assertTrue("no replayable ToF frame in $capture", rows > 0)
    }
}
