package com.linnan.blindassist.vision

import android.graphics.Bitmap
import com.linnan.blindassist.capture.ImuCaptureFrame
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.*

/** Installation-bound calibration; raw recording/display paths never use this warp. */
class MobileFusionCamera(directory: File) {
    private val config = JSONObject(File(directory, "camera.json").readText())
    private val map = File(directory, "atom_remap.bin").readBytes()
    private val fixed = ByteBuffer.wrap(map).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
    private val imuRotation = matrix(config.getJSONArray("tof_from_imu"))
    private val deviceId = config.getString("atom_device_id")
    val width = config.getInt("width")
    val height = config.getInt("height")
    val calibration: AtomRgbGeometry.Calibration

    init {
        require(width == 480 && height == 640 && map.size == width * height * 6)
        require(MessageDigest.getInstance("SHA-256").digest(map).joinToString("") { "%02x".format(it) } ==
            config.getString("remap_sha256")) { "Atom remap identity mismatch" }
        val k = config.getJSONArray("K")
        calibration = AtomRgbGeometry.Calibration(k.getJSONArray(0).getDouble(0), k.getJSONArray(1).getDouble(1),
            k.getJSONArray(0).getDouble(2), k.getJSONArray(1).getDouble(2), matrix(config.getJSONArray("camera_from_tof")))
    }

    data class Upright(val bitmap: Bitmap, val valid: BooleanArray)
    fun undistort(raw: Bitmap): Upright {
        require(raw.width == 640 && raw.height == 480) { "Calibration requires 640×480 Atom frames" }
        val source = IntArray(640 * 480)
        raw.getPixels(source, 0, 640, 0, 0, 640, 480)
        val result = IntArray(width * height)
        val valid = BooleanArray(result.size)
        fun pixel(x: Int, y: Int) = if (x in 0..639 && y in 0..479) source[y * 640 + x] else 0
        for (i in result.indices) {
            val x = fixed.get(i * 3).toInt(); val y = fixed.get(i * 3 + 1).toInt()
            val frac = fixed.get(i * 3 + 2).toInt() and 1023
            val fx = frac and 31; val fy = frac shr 5
            val p0 = pixel(x,y); val p1 = pixel(x+1,y); val p2 = pixel(x,y+1); val p3 = pixel(x+1,y+1)
            val w0 = (32-fx)*(32-fy); val w1 = fx*(32-fy); val w2 = (32-fx)*fy; val w3 = fx*fy
            var color = -0x1000000
            for (channel in 0..2) {
                val shift = (2-channel)*8
                val sum = 512 + ((p0 ushr shift) and 255)*w0 + ((p1 ushr shift) and 255)*w1 +
                    ((p2 ushr shift) and 255)*w2 + ((p3 ushr shift) and 255)*w3
                color = color or ((sum shr 10).coerceIn(0,255) shl shift)
            }
            result[i] = color; valid[i] = (color and 0xffffff) != 0
        }
        return Upright(Bitmap.createBitmap(result, width, height, Bitmap.Config.ARGB_8888), valid)
    }

    fun attitude(frame: ImuCaptureFrame): Pair<Double, Double>? {
        if (!frame.deviceId.equals(deviceId, ignoreCase = true)) return null
        val gravity = frame.gravityUnit ?: return null
        val t = DoubleArray(3) { r -> (0..2).sumOf { c -> imuRotation[r * 3 + c] * gravity[c] } }
        val norm = sqrt(t.sumOf { it * it })
        if (!norm.isFinite() || norm < .9 || norm > 1.1) return null
        for (i in 0..2) t[i] /= norm
        if (hypot(t[0], t[1]) < 1e-3) return null
        return Math.toDegrees(asin(t[2].coerceIn(-1.0, 1.0))) to Math.toDegrees(atan2(-t[0], -t[1]))
    }

    private fun matrix(rows: JSONArray) = DoubleArray(9) { rows.getJSONArray(it / 3).getDouble(it % 3) }
}
