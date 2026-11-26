package com.linnan.blindassist.vision.egogeom

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.ln1p
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Numerical input/output contracts for EgoGeom-Omni v4.1 (288x384 NCHW) on Hexagon v75.
 * Matches `egogeom_omni_v41_model.py` and `eval_v42_real_h3_replay.py` bit-for-bit.
 */
object EgoGeomContracts {
    const val HEIGHT = 288
    const val WIDTH = 384
    const val STRIDE = 4
    const val S4_HEIGHT = HEIGHT / STRIDE // 72
    const val S4_WIDTH = WIDTH / STRIDE   // 96
    const val RGB_FLOATS = 3 * HEIGHT * WIDTH
    const val RAYS_FLOATS = 3 * HEIGHT * WIDTH
    const val TOF_S4_FLOATS = 20 * S4_HEIGHT * S4_WIDTH
    const val DEPTH_FLOATS = HEIGHT * WIDTH
    const val NORMAL_FLOATS = 3 * HEIGHT * WIDTH
    const val CANONICAL_FOCAL_PX = 350.0

    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /**
     * Exact 4-row largest-chord analytical host_focal (350 / fx_eff), matching FP32 rays to <3.3e-16.
     * Port of `analytical_host_focal_from_K` in `egogeom_omni_v41_model.py`.
     */
    fun analyticalHostFocal(
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        width: Int = WIDTH,
        height: Int = HEIGHT,
        canonicalFocalPx: Double = CANONICAL_FOCAL_PX,
    ): Float {
        val left = (cx - 1.0).roundToInt().coerceIn(0, width - 2)
        val a = (left + 0.5 - cx) / fx
        val b = a + 1.0 / fx
        val yStar = sqrt(max(a * b - 1.0, 0.0))
        val rows = LinkedHashSet<Int>(4)
        for (y in doubleArrayOf(-yStar, yStar)) {
            val target = cy - 0.5 + fy * y
            rows.add(floor(target).toInt().coerceIn(0, height - 1))
            rows.add(ceil(target).toInt().coerceIn(0, height - 1))
        }
        var maxChord = 0.0
        for (row in rows) {
            val r = hypot(1.0, (row + 0.5 - cy) / fy)
            val theta = atan2(r / fx, r * r + a * b)
            maxChord = max(maxChord, 2.0 * sin(0.5 * theta))
        }
        val value = canonicalFocalPx * sqrt(maxChord * maxChord + 1e-12)
        return value.coerceIn(0.20, 4.0).toFloat()
    }

    /**
     * Build gravity-aligned unit ray map `[3, H, W]` NCHW for pinhole `(fx, fy, cx, cy)` and
     * camera-frame gravity down-vector `gCam = (gx, gy, gz)` (default `(0, 1, 0)` for level mount).
     * Mirrors `compute_gravity_aligned_ray_map` in `egogeom_losses_and_physics.py`.
     */
    fun buildGravityAlignedRays(
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        gCamX: Float = 0f,
        gCamY: Float = 1f,
        gCamZ: Float = 0f,
        out: FloatArray = FloatArray(RAYS_FLOATS),
    ): FloatArray {
        require(out.size == RAYS_FLOATS) { "Expected $RAYS_FLOATS floats for rays" }
        val gNorm = sqrt((gCamX * gCamX + gCamY * gCamY + gCamZ * gCamZ).toDouble()).coerceAtLeast(1e-6)
        val yx = (gCamX / gNorm).toFloat()
        val yy = (gCamY / gNorm).toFloat()
        val yz = (gCamZ / gNorm).toFloat()

        // World-aligned camera basis: y_axis = g_down, x_axis = normalize(y x z_cam), z_axis = x x y
        var xx = yy * 1f - yz * 0f
        var xy = yz * 0f - yx * 1f
        var xz = 0f
        val xNorm = sqrt((xx * xx + xy * xy + xz * xz).toDouble())
        if (xNorm < 1e-4) {
            xx = 1f; xy = 0f; xz = 0f
        } else {
            xx = (xx / xNorm).toFloat()
            xy = (xy / xNorm).toFloat()
            xz = (xz / xNorm).toFloat()
        }
        val zx = xy * yz - xz * yy
        val zy = xz * yx - xx * yz
        val zz = xx * yy - xy * yx

        val plane = HEIGHT * WIDTH
        val invFx = 1f / fx
        val invFy = 1f / fy
        for (v in 0 until HEIGHT) {
            val ry = (v + 0.5f - cy) * invFy
            val rowOff = v * WIDTH
            for (u in 0 until WIDTH) {
                val rx = (u + 0.5f - cx) * invFx
                val invLen = 1f / sqrt(rx * rx + ry * ry + 1f)
                val ux = rx * invLen
                val uy = ry * invLen
                val uz = invLen
                val idx = rowOff + u
                out[idx] = xx * ux + xy * uy + xz * uz
                out[plane + idx] = yx * ux + yy * uy + yz * uz
                out[2 * plane + idx] = zx * ux + zy * uy + zz * uz
            }
        }
        return out
    }

    /**
     * Bilinear resize + ImageNet normalize ARGB_8888 pixel array (`srcW x srcH`) into NCHW `[3, 288, 384]`.
     * Also optionally writes s4 (`72 x 96`) grayscale luma `[0, 1]` for fast causal parallax gating.
     */
    fun preprocessArgbToNchw(
        argbPixels: IntArray,
        srcW: Int,
        srcH: Int,
        outRgbNchw: FloatArray = FloatArray(RGB_FLOATS),
        outGrayS4: FloatArray? = null,
    ): FloatArray {
        require(argbPixels.size == srcW * srcH) { "Expected ${srcW * srcH} pixels, got ${argbPixels.size}" }
        require(outRgbNchw.size == RGB_FLOATS) { "Expected $RGB_FLOATS floats" }
        if (outGrayS4 != null) require(outGrayS4.size == S4_HEIGHT * S4_WIDTH)

        val plane = HEIGHT * WIDTH
        val scaleX = srcW.toFloat() / WIDTH.toFloat()
        val scaleY = srcH.toFloat() / HEIGHT.toFloat()
        val m0 = IMAGENET_MEAN[0]; val m1 = IMAGENET_MEAN[1]; val m2 = IMAGENET_MEAN[2]
        val s0 = 1f / IMAGENET_STD[0]; val s1 = 1f / IMAGENET_STD[1]; val s2 = 1f / IMAGENET_STD[2]

        for (y in 0 until HEIGHT) {
            val sy = ((y + 0.5f) * scaleY - 0.5f).coerceIn(0f, (srcH - 1).toFloat())
            val y0 = sy.toInt()
            val y1 = min(y0 + 1, srcH - 1)
            val wy = sy - y0
            val wy0 = 1f - wy
            val row0 = y0 * srcW
            val row1 = y1 * srcW
            val dstRow = y * WIDTH
            val sampleGrayRow = outGrayS4 != null && (y and 3) == 1
            val gyRow = (y shr 2) * S4_WIDTH

            for (x in 0 until WIDTH) {
                val sx = ((x + 0.5f) * scaleX - 0.5f).coerceIn(0f, (srcW - 1).toFloat())
                val x0 = sx.toInt()
                val x1 = min(x0 + 1, srcW - 1)
                val wx = sx - x0
                val wx0 = 1f - wx

                val p00 = argbPixels[row0 + x0]
                val p01 = argbPixels[row0 + x1]
                val p10 = argbPixels[row1 + x0]
                val p11 = argbPixels[row1 + x1]

                val w00 = wx0 * wy0; val w01 = wx * wy0; val w10 = wx0 * wy; val w11 = wx * wy
                val r = (w00 * ((p00 ushr 16) and 0xFF) + w01 * ((p01 ushr 16) and 0xFF) +
                    w10 * ((p10 ushr 16) and 0xFF) + w11 * ((p11 ushr 16) and 0xFF)) * (1f / 255f)
                val g = (w00 * ((p00 ushr 8) and 0xFF) + w01 * ((p01 ushr 8) and 0xFF) +
                    w10 * ((p10 ushr 8) and 0xFF) + w11 * ((p11 ushr 8) and 0xFF)) * (1f / 255f)
                val b = (w00 * (p00 and 0xFF) + w01 * (p01 and 0xFF) +
                    w10 * (p10 and 0xFF) + w11 * (p11 and 0xFF)) * (1f / 255f)

                val idx = dstRow + x
                outRgbNchw[idx] = (r - m0) * s0
                outRgbNchw[plane + idx] = (g - m1) * s1
                outRgbNchw[2 * plane + idx] = (b - m2) * s2
                if (sampleGrayRow && (x and 3) == 1) {
                    outGrayS4[gyRow + (x shr 2)] = 0.299f * r + 0.587f * g + 0.114f * b
                }
            }
        }
        return outRgbNchw
    }

    /**
     * Pack real ST VL53L8CH H3 signed CNH frame (`64 x 16` signed bins + `64` mm distances + `64` status + `64` targets)
     * into the v4.1 20-channel s4 tensor `[20, 72, 96]` (each 8x8 zone mapped to exact 9x12 block).
     * Mirrors `pack_real_h3_to_20ch` in `eval_v42_real_h3_replay.py`.
     */
    fun packRealH3ToTofGridS4(
        signedHist64x16: FloatArray,
        distanceMm64: IntArray,
        status64: IntArray,
        nbTargets64: IntArray,
        outTofS4: FloatArray = FloatArray(TOF_S4_FLOATS),
    ): FloatArray {
        require(signedHist64x16.size == 64 * 16) { "Expected 1024 signed H3 bins" }
        require(distanceMm64.size == 64 && status64.size == 64 && nbTargets64.size == 64)
        require(outTofS4.size == TOF_S4_FLOATS)

        val cell = FloatArray(20 * 64)
        for (z in 0 until 64) {
            val st = status64[z]
            val dMm = distanceMm64[z]
            val validD = (st == 5 || st == 9) && nbTargets64[z] > 0 && dMm in 50..4800
            cell[z] = if (validD) ln1p(dMm * 0.001f) else 0f
            cell[64 + z] = if (validD) 1f else 0f

            val hOff = z * 16
            var sumSq = 0.0
            for (b in 0 until 16) {
                val hv = signedHist64x16[hOff + b].toDouble()
                sumSq += hv * hv
            }
            val invRms = (1.0 / sqrt(sumSq / 16.0 + 1e-6)).toFloat()
            for (b in 0 until 16) {
                val unit = signedHist64x16[hOff + b] * invRms
                val enc = ln1p(10.0 * abs(unit.toDouble())).toFloat()
                cell[(2 + b) * 64 + z] = when {
                    unit > 0f -> enc
                    unit < 0f -> -enc
                    else -> 0f
                }
            }
            cell[18 * 64 + z] = 1f // cnh_valid
            cell[19 * 64 + z] = 1f // has_tof_ch
        }

        // Nearest-neighbor repeat: 8x8 -> 72x96 (exact 9 rows x 12 cols per zone)
        val planeS4 = S4_HEIGHT * S4_WIDTH
        for (c in 0 until 20) {
            val cOff = c * 64
            val dstPlane = c * planeS4
            for (zy in 0 until 8) {
                val rStart = zy * 9
                for (zx in 0 until 8) {
                    val v = cell[cOff + zy * 8 + zx]
                    val cStart = zx * 12
                    for (dy in 0 until 9) {
                        val rowOff = dstPlane + (rStart + dy) * S4_WIDTH + cStart
                        for (dx in 0 until 12) outTofS4[rowOff + dx] = v
                    }
                }
            }
        }
        return outTofS4
    }
}
