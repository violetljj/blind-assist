package com.linnan.blindassist.vision.egogeom

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.expm1
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Mount-agnostic elevation profile for scale-free gravity-aligned risk cones.
 * Switching between head-mounted glasses and neck-mounted pendants requires zero backbone retraining.
 */
data class WearableMountProfile(
    val name: String,
    val elLoDeg: Float,
    val elHiDeg: Float,
    val gyroElLoDeg: Float,
    val gyroElHiDeg: Float,
    val glassElLoDeg: Float,
    val glassElHiDeg: Float,
) {
    companion object {
        val HEAD_MOUNTED = WearableMountProfile(
            name = "head_mounted",
            elLoDeg = -18f,
            elHiDeg = 12f,
            gyroElLoDeg = -16f,
            gyroElHiDeg = 12f,
            glassElLoDeg = -14f,
            glassElHiDeg = 14f,
        )

        val NECK_MOUNTED = WearableMountProfile(
            name = "neck_mounted",
            elLoDeg = -14f,
            elHiDeg = 20f,
            gyroElLoDeg = -12f,
            gyroElHiDeg = 20f,
            glassElLoDeg = -10f,
            glassElHiDeg = 22f,
        )
    }
}

enum class CameraImuFusionMode {
    /** 100% Zero-VIO, Zero-ToF pure Camera+IMU (ang08_front_log_ttc + gyro_ttc_gated). */
    ZERO_VIO_CAMERA_IMU,

    /** 2-Channel balanced profile (m6_ha70_par_refute + ang08_front_log_ttc). */
    BALANCED_2CH,

    /** 3-Channel full Camera+IMU profile (m6_ha70_par_refute + ang08_front_log_ttc + gyro_ttc_gated). */
    FULL_3CH,
}

enum class GlassAlertTier {
    NONE,
    CAUTION,
    HIGH,
}

enum class WearableHazardChannel {
    NONE,
    SOLID_OBSTACLE,
    OVERHANG_LOOMING,
    GLASS_HAZARD,
}

internal data class AngularRaysGrid(
    val hs: Int,
    val ws: Int,
    val azRad: FloatArray,
    val elRad: FloatArray,
    val fwdDot: FloatArray,
)

internal fun computeAngularRaysGrid(
    h: Int,
    w: Int,
    fx: Float,
    fy: Float,
    cx: Float,
    cy: Float,
    rWc: FloatArray,
    strictFwd: FloatArray,
    stride: Int = 4,
    step: Int = 2,
): AngularRaysGrid? {
    val fwX0 = strictFwd[0].toDouble()
    val fwY0 = strictFwd[1].toDouble()
    val nf = hypot(fwX0, fwY0)
    if (nf < 1e-6) return null
    val fwx = (fwX0 / nf).toFloat()
    val fwy = (fwY0 / nf).toFloat()
    val rtx = fwy
    val rty = -fwx

    val hs = (h + step - 1) / step
    val ws = (w + step - 1) / step
    val n = hs * ws
    val az = FloatArray(n)
    val el = FloatArray(n)
    val fwdDot = FloatArray(n)

    val sPix = (stride * step).toFloat()
    val invFx = 1f / max(fx, 1f)
    val invFy = 1f / max(fy, 1f)
    val r00 = rWc[0]; val r01 = rWc[1]; val r02 = rWc[2]
    val r10 = rWc[3]; val r11 = rWc[4]; val r12 = rWc[5]
    val r20 = rWc[6]; val r21 = rWc[7]; val r22 = rWc[8]

    var k = 0
    for (vi in 0 until hs) {
        val ry = (sPix * vi + 0.5f - cy) * invFy
        for (ui in 0 until ws) {
            val rx = (sPix * ui + 0.5f - cx) * invFx
            val invN = 1f / sqrt(rx * rx + ry * ry + 1f)
            val cxR = rx * invN
            val cyR = ry * invN
            val czR = invN
            val wx = r00 * cxR + r01 * cyR + r02 * czR
            val wy = r10 * cxR + r11 * cyR + r12 * czR
            val wz = r20 * cxR + r21 * cyR + r22 * czR
            val fd = wx * fwx + wy * fwy
            val ld = wx * rtx + wy * rty
            fwdDot[k] = fd
            az[k] = atan2(ld.toDouble(), max(fd, 1e-5f).toDouble()).toFloat()
            el[k] = asin(wz.coerceIn(-1f, 1f).toDouble()).toFloat()
            k++
        }
    }
    return AngularRaysGrid(hs, ws, az, el, fwdDot)
}

internal fun percentileLinear(values: FloatArray, count: Int, qPercent: Double): Float {
    if (count <= 0) return 0f
    val arr = values.copyOf(count)
    arr.sort()
    val pos = (qPercent / 100.0) * (count - 1)
    val lo = floor(pos).toInt().coerceIn(0, count - 1)
    val hi = min(lo + 1, count - 1)
    val frac = (pos - lo).toFloat()
    return arr[lo] + frac * (arr[hi] - arr[lo])
}

/**
 * Pure Kotlin 1:1 port of Python `NarrowAngularConeTTC` (`eval_unified_camera_imu_fusion_local.py`).
 *
 * Scale-free gravity-horizon narrow angular cone with `normal_grav` frontal-normal side-wall suppression,
 * azimuth cosine weighting, and `-d(log Z)/dt` scale-free approach boost. Uses ZERO VIO translation (`t_w`).
 */
class NarrowAngularConeTtc(
    val azHalfDeg: Float = 8f,
    val elLoDeg: Float = WearableMountProfile.HEAD_MOUNTED.elLoDeg,
    val elHiDeg: Float = WearableMountProfile.HEAD_MOUNTED.elHiDeg,
    val useLogTtc: Boolean = true,
    val useFrontality: Boolean = true,
    val useAzCos: Boolean = true,
    val nAz: Int = 3,
    val nEl: Int = 4,
    val supportScale: Float = 12f,
) {
    private val azHalfRad = (azHalfDeg * PI / 180.0).toFloat()
    private val elLoRad = (elLoDeg * PI / 180.0).toFloat()
    private val elHiRad = (elHiDeg * PI / 180.0).toFloat()

    private var prevCells: DoubleArray? = null
    private var prevLogZ: FloatArray? = null
    private var floorDhEma: Float = 0f

    fun reset() {
        prevCells = null
        prevLogZ = null
        floorDhEma = 0f
    }

    /**
     * Step on stride-4 (`72 x 96`) feature maps:
     * - `nrmRightS4`: `normal_grav[0]` (+Right horizontal)
     * - `nrmDownS4`: `normal_grav[1]` (+Down along gravity)
     * - `nrmFwdS4`: `normal_grav[2]` (+Forward horizontal)
     */
    fun step(
        depthS4: FloatArray,
        deltaHS4: FloatArray,
        nrmRightS4: FloatArray,
        nrmDownS4: FloatArray,
        nrmFwdS4: FloatArray,
        logSigmaS4: FloatArray,
        hS4: Int = EgoGeomContracts.S4_HEIGHT,
        wS4: Int = EgoGeomContracts.S4_WIDTH,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        rWc: FloatArray,
        strictFwd: FloatArray,
    ): Float {
        val ang = computeAngularRaysGrid(hS4, wS4, fx, fy, cx, cy, rWc, strictFwd, stride = 4, step = 2)
        if (ang == null) {
            prevCells = null
            prevLogZ = null
            return 0f
        }
        val hs = ang.hs
        val ws = ang.ws
        val n = hs * ws

        val zArr = FloatArray(n)
        val dhArr = FloatArray(n)
        val nxArr = FloatArray(n)
        val ndArr = FloatArray(n)
        val nzArr = FloatArray(n)
        val lsArr = FloatArray(n)
        val dispArr = FloatArray(n)
        val valid = BooleanArray(n)

        val validLs = FloatArray(n)
        val validDisp = FloatArray(n)
        val bgDisp = FloatArray(n)
        val flDh = FloatArray(n)

        var validN = 0
        var bgN = 0
        var flN = 0
        var k = 0
        for (vi in 0 until hs) {
            val rowOff = (vi * 2) * wS4
            for (ui in 0 until ws) {
                val idx = rowOff + ui * 2
                val z = depthS4[idx]
                val dh = deltaHS4[idx]
                val nx = nrmRightS4[idx]
                val nd = nrmDownS4[idx]
                val nz = nrmFwdS4[idx]
                val ls = logSigmaS4[idx]
                zArr[k] = z
                dhArr[k] = dh
                nxArr[k] = nx
                ndArr[k] = nd
                nzArr[k] = nz
                lsArr[k] = ls

                val ok = z.isFinite() && z > 0.20f && z < 25.0f && ang.fwdDot[k] > 0.25f
                valid[k] = ok
                if (ok) {
                    val disp = 1f / z.coerceIn(0.25f, 25.0f)
                    dispArr[k] = disp
                    validDisp[validN] = disp
                    validLs[validN] = ls
                    validN++
                    if (ang.elRad[k] >= -0.22f) {
                        bgDisp[bgN++] = disp
                    }
                    if (ang.elRad[k] < -0.25f && abs(nd) >= 0.72f) {
                        flDh[flN++] = dh
                    }
                }
                k++
            }
        }

        if (validN < 20) {
            prevCells = null
            prevLogZ = null
            return 0f
        }

        if (flN >= 12) {
            val obsDh = percentileLinear(flDh, flN, 50.0).coerceIn(-0.15f, 0.20f)
            floorDhEma = 0.75f * floorDhEma + 0.25f * obsDh
        }

        val dispBgRaw = if (bgN >= 15) {
            percentileLinear(bgDisp, bgN, 30.0)
        } else {
            percentileLinear(validDisp, validN, 30.0)
        }
        val dispBg = max(dispBgRaw, 0.04f)
        val lsMed = percentileLinear(validLs, validN, 50.0)

        val nCells = nAz * nEl
        val nowCells = DoubleArray(nCells)
        val curLogZ = FloatArray(nCells) { Float.NaN }
        val cellLogZLists = if (useLogTtc) Array(nCells) { ArrayList<Float>(16) } else null

        val azSpan = 2f * azHalfRad
        val elSpan = elHiRad - elLoRad
        for (i in 0 until n) {
            if (!valid[i]) continue
            val az = ang.azRad[i]
            val el = ang.elRad[i]
            if (abs(az) > azHalfRad || el < elLoRad || el > elHiRad) continue

            val disp = dispArr[i]
            val relProm = ((disp / dispBg) - 1f).coerceIn(0f, 6f)
            val nearSig = (disp - 0.22f).coerceIn(0f, 3f)
            val conf = exp(-(lsArr[i] - lsMed).coerceIn(-2f, 2f).toDouble()).toFloat()
            val nd = ndArr[i]
            val vert = (1f - nd * nd).coerceIn(0f, 1f)
            val dhW = ((dhArr[i] - floorDhEma) / 0.22f).coerceIn(0f, 1f)
            val elRamp = ((el - elLoRad) / 0.14f).coerceIn(0f, 1f)

            var ev = (0.55f * relProm + 0.45f * nearSig * 2.5f) * vert * dhW * conf * elRamp
            if (useFrontality) {
                val horizN = hypot(nxArr[i].toDouble(), nzArr[i].toDouble()).toFloat() + 1e-4f
                val frontCos = abs(nzArr[i]) / horizN
                ev *= (0.15f + 0.85f * frontCos * frontCos)
            }
            if (useAzCos) {
                val azFrac = (abs(az) / azHalfRad).coerceIn(0f, 1f)
                ev *= cos(azFrac * (0.5 * PI)).toFloat()
            }

            val ia = (((az + azHalfRad) / azSpan) * nAz).toInt().coerceIn(0, nAz - 1)
            val ie = (((el - elLoRad) / elSpan) * nEl).toInt().coerceIn(0, nEl - 1)
            val cid = ie * nAz + ia
            nowCells[cid] += ev.toDouble()
            if (cellLogZLists != null) {
                cellLogZLists[cid].add(ln(zArr[i].coerceIn(0.25f, 25.0f).toDouble()).toFloat())
            }
        }

        if (cellLogZLists != null) {
            for (c in 0 until nCells) {
                val lst = cellLogZLists[c]
                if (lst.isNotEmpty()) {
                    val tmp = FloatArray(lst.size) { lst[it] }
                    curLogZ[c] = percentileLinear(tmp, tmp.size, 20.0)
                }
            }
        }

        val prev = prevCells
        val pLogZ = prevLogZ
        var maxRisk = 0.0
        for (c in 0 until nCells) {
            val prevVal = prev?.get(c) ?: 0.0
            var sup = 0.35 * nowCells[c] + 0.65 * min(nowCells[c], prevVal)
            if (useLogTtc && pLogZ != null && curLogZ[c].isFinite() && pLogZ[c].isFinite()) {
                val approach = (pLogZ[c] - curLogZ[c]).coerceIn(0f, 0.5f)
                sup *= (1.0 + 3.0 * approach.toDouble())
            }
            val risk = -expm1(-sup / supportScale.toDouble())
            if (risk > maxRisk) maxRisk = risk
        }

        prevCells = nowCells
        prevLogZ = curLogZ
        return maxRisk.toFloat()
    }
}

/**
 * Pure Kotlin 1:1 port of Python `GatedGyroEpipolarTTCRisk` (`eval_unified_camera_imu_fusion_local.py`).
 *
 * ZERO neural network, ZERO depth, ZERO VIO translation (`t_w`). Uses only two grayscale `72 x 96` frames,
 * camera intrinsics, gyroscope relative rotation `R_rel = R_prev^T R_cur`, and forward unit heading `strictFwd`.
 * Includes IMU rotation-rate muting during rapid head turns and wide-field walking coherence gating.
 */
class GatedGyroTtc(
    val lag: Int = 2,
    val azHalfDeg: Float = 12f,
    val elLoDeg: Float = WearableMountProfile.HEAD_MOUNTED.gyroElLoDeg,
    val elHiDeg: Float = WearableMountProfile.HEAD_MOUNTED.gyroElHiDeg,
    val ttcSteps: FloatArray = floatArrayOf(32f, 20f, 13f, 8f),
    val minGrad: Float = 0.012f,
    val minParPx: Float = 1.2f,
    val maxBestCost: Float = 0.22f,
    val minInfMargin: Float = 0.14f,
    val minUnimodalMargin: Float = 0.04f,
    val rotGateLoDeg: Float = 4.0f,
    val rotGateHiDeg: Float = 9.0f,
) {
    private data class HistItem(
        val gray: FloatArray,
        val fxS4: Float,
        val fyS4: Float,
        val cxS4: Float,
        val cyS4: Float,
        val rWc: FloatArray,
    )

    private val azHalfRad = (azHalfDeg * PI / 180.0).toFloat()
    private val elLoRad = (elLoDeg * PI / 180.0).toFloat()
    private val elHiRad = (elHiDeg * PI / 180.0).toFloat()
    private val wideAzRad = (28.0 * PI / 180.0).toFloat()
    private val wideElLoRad = (-26.0 * PI / 180.0).toFloat()

    private val hist = ArrayDeque<HistItem>(lag + 2)
    private var prevGrid: DoubleArray? = null
    private var walkEma: Float = 0f

    var lastRotDeg: Float = 0f
        private set
    var lastWalkEma: Float = 0f
        private set

    fun reset() {
        hist.clear()
        prevGrid = null
        walkEma = 0f
        lastRotDeg = 0f
        lastWalkEma = 0f
    }

    fun step(
        grayS4: FloatArray,
        hS4: Int = EgoGeomContracts.S4_HEIGHT,
        wS4: Int = EgoGeomContracts.S4_WIDTH,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        rWc: FloatArray,
        strictFwd: FloatArray,
    ): Float {
        val fxS4 = fx * 0.25f
        val fyS4 = fy * 0.25f
        val cxS4 = cx * 0.25f
        val cyS4 = cy * 0.25f
        if (hist.size >= lag + 1) hist.removeFirst()
        hist.addLast(HistItem(grayS4.copyOf(), fxS4, fyS4, cxS4, cyS4, rWc.copyOf()))
        if (hist.size <= lag) {
            prevGrid = null
            return 0f
        }

        val prev = hist.first()
        val rp = prev.rWc
        val rRel = FloatArray(9)
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                rRel[i * 3 + j] = rp[0 * 3 + i] * rWc[0 * 3 + j] +
                    rp[1 * 3 + i] * rWc[1 * 3 + j] +
                    rp[2 * 3 + i] * rWc[2 * 3 + j]
            }
        }
        val trace = rRel[0] + rRel[4] + rRel[8]
        val cosTh = ((trace - 1f) * 0.5f).coerceIn(-1f, 1f).toDouble()
        val rotDeg = (acos(cosTh) * 180.0 / PI).toFloat()
        lastRotDeg = rotDeg
        val rotGate = ((rotGateHiDeg - rotDeg) / max(rotGateHiDeg - rotGateLoDeg, 1e-3f)).coerceIn(0f, 1f)
        val nCells = 9
        if (rotGate <= 1e-3f) {
            prevGrid = DoubleArray(nCells)
            return 0f
        }

        val ang = computeAngularRaysGrid(hS4, wS4, fx, fy, cx, cy, rWc, strictFwd, stride = 4, step = 2)
        if (ang == null) {
            prevGrid = null
            return 0f
        }

        val fwX0 = strictFwd[0].toDouble()
        val fwY0 = strictFwd[1].toDouble()
        val nf = hypot(fwX0, fwY0) + 1e-6
        val fw0 = (fwX0 / nf).toFloat()
        val fw1 = (fwY0 / nf).toFloat()
        // u_fwd_prev = R_prev^T @ [fw0, fw1, 0]
        val uFwdX = rp[0] * fw0 + rp[3] * fw1
        val uFwdY = rp[1] * fw0 + rp[4] * fw1
        val uFwdZ = rp[2] * fw0 + rp[5] * fw1

        val invLams = FloatArray(1 + ttcSteps.size)
        invLams[0] = 0f
        for (j in ttcSteps.indices) invLams[1 + j] = lag.toFloat() / ttcSteps[j]
        val nHyp = invLams.size

        val invFxC = 1f / max(fxS4, 1f)
        val invFyC = 1f / max(fyS4, 1f)
        val fxP = prev.fxS4; val fyP = prev.fyS4; val cxP = prev.cxS4; val cyP = prev.cyS4
        val prevGray = prev.gray

        val pc = FloatArray(9)
        val pp = FloatArray(9)
        val costs = FloatArray(nHyp)
        val upArr = FloatArray(nHyp)
        val vpArr = FloatArray(nHyp)
        val inImgArr = BooleanArray(nHyp)

        // First pass: collect valid wide-field pixels to compute walk_ema before weighting center hits
        val passedCids = IntArray(ang.hs * ang.ws)
        val passedBestJ = IntArray(ang.hs * ang.ws)
        var passedN = 0
        var okCount = 0
        var fwdAgreeCount = 0

        val azSpan = 2f * azHalfRad
        val elSpan = elHiRad - elLoRad

        for (vi in 0 until ang.hs) {
            val v = vi * 2
            if (v < 2 || v >= hS4 - 2) continue
            val rowOff = v * wS4
            for (ui in 0 until ang.ws) {
                val u = ui * 2
                if (u < 2 || u >= wS4 - 2) continue
                val subIdx = vi * ang.ws + ui
                val az = ang.azRad[subIdx]
                val el = ang.elRad[subIdx]
                if (abs(az) > wideAzRad || el < wideElLoRad || el > elHiRad || ang.fwdDot[subIdx] <= 0.3f) continue

                val gx = grayS4[rowOff + u + 1] - grayS4[rowOff + u - 1]
                val gy = grayS4[(v + 1) * wS4 + u] - grayS4[(v - 1) * wS4 + u]
                if (hypot(gx.toDouble(), gy.toDouble()) < minGrad) continue

                val rx = (u - cxS4) * invFxC
                val ry = (v - cyS4) * invFyC
                val drx = rRel[0] * rx + rRel[1] * ry + rRel[2]
                val dry = rRel[3] * rx + rRel[4] * ry + rRel[5]
                val drz = rRel[6] * rx + rRel[7] * ry + rRel[8]

                for (s in 0 until nHyp) {
                    val lam = invLams[s]
                    val px = drx + lam * uFwdX
                    val py = dry + lam * uFwdY
                    val pz = drz + lam * uFwdZ
                    if (pz > 0.15f) {
                        val up = fxP * (px / pz) + cxP
                        val vp = fyP * (py / pz) + cyP
                        upArr[s] = up
                        vpArr[s] = vp
                        inImgArr[s] = up >= 2f && up < wS4 - 2f && vp >= 2f && vp < hS4 - 2f
                    } else {
                        inImgArr[s] = false
                    }
                }
                if (!inImgArr[0] || !inImgArr[nHyp - 1]) continue
                val parPx = hypot((upArr[0] - upArr[nHyp - 1]).toDouble(), (vpArr[0] - vpArr[nHyp - 1]).toDouble())
                if (parPx < minParPx) continue

                okCount++
                // 3x3 patch in current grayS4
                var meanC = 0f
                var pIdx = 0
                for (dy in -1..1) {
                    val rO = (v + dy) * wS4 + u
                    for (dx in -1..1) {
                        val vv = grayS4[rO + dx]
                        pc[pIdx++] = vv
                        meanC += vv
                    }
                }
                meanC /= 9f
                var normC = 1e-6f
                for (m in 0 until 9) {
                    pc[m] -= meanC
                    normC += pc[m] * pc[m]
                }
                val invNormC = 1f / sqrt(normC)

                for (s in 0 until nHyp) {
                    if (!inImgArr[s]) {
                        costs[s] = 9f
                        continue
                    }
                    costs[s] = zncc3x3Bilinear(prevGray, hS4, wS4, pc, invNormC, upArr[s], vpArr[s], pp)
                }

                val cInf = costs[0]
                var bestJ = 0
                var cBest = costs[1]
                for (j in 1 until ttcSteps.size) {
                    if (costs[1 + j] < cBest) {
                        cBest = costs[1 + j]
                        bestJ = j
                    }
                }
                if (cBest < 0.25f && (cInf - cBest) >= 0.08f) {
                    fwdAgreeCount++
                }

                if (abs(az) <= azHalfRad && el >= elLoRad && el <= elHiRad) {
                    var cOther = 9f
                    for (j in ttcSteps.indices) {
                        if (abs(j - bestJ) > 1 && costs[1 + j] < cOther) cOther = costs[1 + j]
                    }
                    if (cBest < maxBestCost && (cInf - cBest) >= minInfMargin && (cOther - cBest) >= minUnimodalMargin) {
                        val ia = (((az + azHalfRad) / azSpan) * 3f).toInt().coerceIn(0, 2)
                        val ie = (((el - elLoRad) / elSpan) * 3f).toInt().coerceIn(0, 2)
                        passedCids[passedN] = ie * 3 + ia
                        passedBestJ[passedN] = bestJ
                        passedN++
                    }
                }
            }
        }

        val nowGrid = DoubleArray(nCells)
        if (okCount > 0) {
            val fwdAgree = fwdAgreeCount.toFloat() / okCount.toFloat()
            walkEma = 0.6f * walkEma + 0.4f * fwdAgree
            lastWalkEma = walkEma
            val walkGate = ((walkEma - 0.025f) / 0.075f).coerceIn(0f, 1f)
            if (passedN > 0 && walkGate > 1e-3f) {
                val cnts = IntArray(nCells)
                val sums = DoubleArray(nCells)
                for (p in 0 until passedN) {
                    val cid = passedCids[p]
                    cnts[cid]++
                    sums[cid] += (10.0 / ttcSteps[passedBestJ[p]].toDouble()) * rotGate.toDouble() * walkGate.toDouble()
                }
                for (c in 0 until nCells) {
                    if (cnts[c] >= 3) nowGrid[c] = sums[c]
                }
            }
        }

        val prevG = prevGrid
        var maxRisk = 0.0
        for (c in 0 until nCells) {
            val prevVal = prevG?.get(c) ?: 0.0
            val sup = 0.35 * nowGrid[c] + 0.65 * min(nowGrid[c], prevVal)
            val risk = -expm1(-sup / 5.0)
            if (risk > maxRisk) maxRisk = risk
        }
        prevGrid = nowGrid
        return maxRisk.toFloat()
    }

    private fun zncc3x3Bilinear(
        prevGray: FloatArray,
        h: Int,
        w: Int,
        pcCentered: FloatArray,
        invNormC: Float,
        up: Float,
        vp: Float,
        ppBuf: FloatArray,
    ): Float {
        var meanP = 0f
        var idx = 0
        for (dy in -1..1) {
            val ys = vp + dy
            val y0 = floor(ys.toDouble()).toInt().coerceIn(0, h - 2)
            val fy = ys - y0
            val fy0 = 1f - fy
            val row0 = y0 * w
            val row1 = (y0 + 1) * w
            for (dx in -1..1) {
                val xs = up + dx
                val x0 = floor(xs.toDouble()).toInt().coerceIn(0, w - 2)
                val fx = xs - x0
                val fx0 = 1f - fx
                val v = fx0 * fy0 * prevGray[row0 + x0] +
                    fx * fy0 * prevGray[row0 + x0 + 1] +
                    fx0 * fy * prevGray[row1 + x0] +
                    fx * fy * prevGray[row1 + x0 + 1]
                ppBuf[idx++] = v
                meanP += v
            }
        }
        meanP /= 9f
        var normP = 1e-6f
        var dot = 0f
        for (m in 0 until 9) {
            val d = ppBuf[m] - meanP
            normP += d * d
            dot += pcCentered[m] * d
        }
        return 1f - dot * invNormC / sqrt(normP)
    }
}

/**
 * Evaluates semantic glass-hazard probability map `glass_prob_s16` (`18 x 24`) from the frozen-DINOv3 G2 head
 * inside a gravity-aligned forward corridor with 2-frame temporal confirmation and optional ToF corroboration.
 *
 * Calibrated on H2 benchmarks: `highProbThr = 0.70`, `minAreaFrac = 0.10` achieved `0.00%` (0/670) false alarms
 * on unseen real outdoor streets while rescuing `82.9%` of depth-penetrated glass pixels.
 */
class GlassCorridorRiskEvaluator(
    val azHalfDeg: Float = 14f,
    val elLoDeg: Float = WearableMountProfile.HEAD_MOUNTED.glassElLoDeg,
    val elHiDeg: Float = WearableMountProfile.HEAD_MOUNTED.glassElHiDeg,
    val cautionProbThr: Float = 0.50f,
    val highProbThr: Float = 0.70f,
    val minAreaFrac: Float = 0.10f,
) {
    data class GlassResult(
        val corridorMeanProb: Float,
        val areaAboveCaution: Float,
        val areaAboveHigh: Float,
        val confirmedFrames: Int,
        val tier: GlassAlertTier,
    )

    companion object {
        const val S16_HEIGHT = 18
        const val S16_WIDTH = 24
        const val S16_FLOATS = S16_HEIGHT * S16_WIDTH
    }

    private val azHalfRad = (azHalfDeg * PI / 180.0).toFloat()
    private val elLoRad = (elLoDeg * PI / 180.0).toFloat()
    private val elHiRad = (elHiDeg * PI / 180.0).toFloat()
    private var consecutiveHits: Int = 0

    fun reset() {
        consecutiveHits = 0
    }

    fun step(
        glassProbS16: FloatArray,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        rWc: FloatArray,
        strictFwd: FloatArray,
        optionalTofNearestM: Float = Float.POSITIVE_INFINITY,
        nrmDownS4: FloatArray? = null,
    ): GlassResult {
        require(glassProbS16.size == S16_FLOATS) { "Expected $S16_FLOATS floats for glass_prob_s16" }
        val ang = computeAngularRaysGrid(
            S16_HEIGHT, S16_WIDTH, fx, fy, cx, cy, rWc, strictFwd, stride = 16, step = 1
        )
        if (ang == null) {
            consecutiveHits = 0
            return GlassResult(0f, 0f, 0f, 0, GlassAlertTier.NONE)
        }

        var count = 0
        var sumProb = 0.0
        var cntCaution = 0
        var cntHigh = 0
        val wS4 = EgoGeomContracts.S4_WIDTH
        for (i in 0 until S16_FLOATS) {
            if (ang.fwdDot[i] > 0.25f &&
                abs(ang.azRad[i]) <= azHalfRad &&
                ang.elRad[i] >= elLoRad &&
                ang.elRad[i] <= elHiRad
            ) {
                var p = glassProbS16[i].coerceIn(0f, 1f)
                if (nrmDownS4 != null && nrmDownS4.size == EgoGeomContracts.S4_HEIGHT * wS4) {
                    val sy = (i / S16_WIDTH) * 4 + 2
                    val sx = (i % S16_WIDTH) * 4 + 2
                    if (abs(nrmDownS4[sy * wS4 + sx]) >= 0.65f) {
                        p = 0f // Reject flat horizontal floors/ceilings
                    }
                }
                count++
                sumProb += p
                if (p >= cautionProbThr) cntCaution++
                if (p >= highProbThr) cntHigh++
            }
        }
        if (count == 0) {
            consecutiveHits = 0
            return GlassResult(0f, 0f, 0f, 0, GlassAlertTier.NONE)
        }

        val meanP = (sumProb / count).toFloat()
        val fracCaution = cntCaution.toFloat() / count.toFloat()
        val fracHigh = cntHigh.toFloat() / count.toFloat()

        val rawHit = fracCaution >= minAreaFrac
        consecutiveHits = if (rawHit) consecutiveHits + 1 else 0

        val tofCorroborated = optionalTofNearestM.isFinite() && optionalTofNearestM in 0.20f..2.50f
        val tier = when {
            consecutiveHits >= 2 && (fracHigh >= minAreaFrac || tofCorroborated) -> GlassAlertTier.HIGH
            consecutiveHits >= 2 -> GlassAlertTier.CAUTION
            else -> GlassAlertTier.NONE
        }
        return GlassResult(meanP, fracCaution, fracHigh, consecutiveHits, tier)
    }
}

/**
 * Unified Camera + IMU (+ Optional ToF) wearable hazard arbitrator combining:
 *  - Channel 1: Metric ground/corridor `m6_ha70_par_refute` (when VIO/odometry available)
 *  - Channel 2: Scale-free frontal-normal narrow angular cone + log-TTC (`NarrowAngularConeTtc`)
 *  - Channel 3: Zero-VIO, zero-depth motion-gated gyroscope epipolar TTC (`GatedGyroTtc`)
 *  - Channel 4: Frozen-DINOv3 G2 semantic glass corridor evaluator (`GlassCorridorRiskEvaluator`)
 */
class WearableRiskArbitrator(
    val mode: CameraImuFusionMode = CameraImuFusionMode.FULL_3CH,
    val mount: WearableMountProfile = WearableMountProfile.HEAD_MOUNTED,
    val solidAlertThr: Float = 0.14f,
    val overhangAlertThr: Float = 0.18f,
) {
    data class ArbitratedRisk(
        val fusedScore: Float,
        val solidScore: Float,
        val narrowConeScore: Float,
        val gyroTtcScore: Float,
        val glass: GlassCorridorRiskEvaluator.GlassResult,
        val primaryChannel: WearableHazardChannel,
    )

    val narrowCone = NarrowAngularConeTtc(
        azHalfDeg = 8f,
        elLoDeg = mount.elLoDeg,
        elHiDeg = mount.elHiDeg,
    )
    val gyroTtc = GatedGyroTtc(
        elLoDeg = mount.gyroElLoDeg,
        elHiDeg = mount.gyroElHiDeg,
    )
    val glassEvaluator = GlassCorridorRiskEvaluator(
        elLoDeg = mount.glassElLoDeg,
        elHiDeg = mount.glassElHiDeg,
    )

    fun reset() {
        narrowCone.reset()
        gyroTtc.reset()
        glassEvaluator.reset()
    }

    fun step(
        solidM6ParRisk: Float,
        depthS4: FloatArray,
        deltaHS4: FloatArray,
        nrmRightS4: FloatArray,
        nrmDownS4: FloatArray,
        nrmFwdS4: FloatArray,
        logSigmaS4: FloatArray,
        grayS4: FloatArray,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        rWc: FloatArray,
        strictFwd: FloatArray,
        glassProbS16: FloatArray? = null,
        optionalTofNearestM: Float = Float.POSITIVE_INFINITY,
    ): ArbitratedRisk {
        val rSolid = solidM6ParRisk.coerceIn(0f, 1f)
        val rCone = narrowCone.step(
            depthS4, deltaHS4, nrmRightS4, nrmDownS4, nrmFwdS4, logSigmaS4,
            EgoGeomContracts.S4_HEIGHT, EgoGeomContracts.S4_WIDTH,
            fx, fy, cx, cy, rWc, strictFwd,
        )
        val rGyro = if (mode == CameraImuFusionMode.BALANCED_2CH) {
            0f
        } else {
            gyroTtc.step(grayS4, EgoGeomContracts.S4_HEIGHT, EgoGeomContracts.S4_WIDTH, fx, fy, cx, cy, rWc, strictFwd)
        }

        val fused = when (mode) {
            CameraImuFusionMode.ZERO_VIO_CAMERA_IMU ->
                (1.0 - (1.0 - 0.65 * rCone) * (1.0 - 0.45 * rGyro)).toFloat()
            CameraImuFusionMode.BALANCED_2CH ->
                (1.0 - (1.0 - rSolid) * (1.0 - 0.18 * rCone)).toFloat()
            CameraImuFusionMode.FULL_3CH ->
                (1.0 - (1.0 - rSolid) * (1.0 - 0.15 * rCone) * (1.0 - 0.18 * rGyro)).toFloat()
        }

        val glassRes = if (glassProbS16 != null) {
            glassEvaluator.step(glassProbS16, fx, fy, cx, cy, rWc, strictFwd, optionalTofNearestM, nrmDownS4)
        } else {
            GlassCorridorRiskEvaluator.GlassResult(0f, 0f, 0f, 0, GlassAlertTier.NONE)
        }

        val overhangSignal = max(rCone, rGyro)
        val channel = when {
            rSolid >= solidAlertThr && mode != CameraImuFusionMode.ZERO_VIO_CAMERA_IMU -> WearableHazardChannel.SOLID_OBSTACLE
            glassRes.tier != GlassAlertTier.NONE -> WearableHazardChannel.GLASS_HAZARD
            overhangSignal >= overhangAlertThr -> WearableHazardChannel.OVERHANG_LOOMING
            else -> WearableHazardChannel.NONE
        }
        return ArbitratedRisk(fused, rSolid, rCone, rGyro, glassRes, channel)
    }
}
