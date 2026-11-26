package com.linnan.blindassist.vision.egogeom

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.expm1
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Pure Kotlin port of `EgoGeomTemporalBEVV3` (`research/active/egogeom-omni-v0/egogeom_temporal_bev_v3.py`),
 * implementing:
 *  1. M6 Scale-Invariant 4-Head Geometric Readout (`depth_s4`, `delta_h_s4`, `normal_grav_s4`, `log_sigma_s4`);
 *  2. Fast (<2 ms) Strictly-Causal Two-Frame Epipolar Parallax Refutation (`stepWithParallax`) for pure-RGB mode.
 *
 * Uses world Z-up convention: `R_wc` (row-major 3x3) maps camera frame (+X right, +Y down, +Z forward)
 * to world frame (+Z up), `t_w` is world camera position (3,), `causalFwd` is world XY heading (2 or 3,).
 */
class EgoGeomTemporalBevV3(
    val cfg: Config = Config(),
) {
    data class Config(
        val stride: Int = 4,
        val geomSubsample: Int = 2,
        val geomFloorNMin: Float = 0.65f,
        val geomMinFloorPts: Int = 12,
        val geomRelHLo: Float = 0.06f,
        val geomRelHHi: Float = 1.30f,
        val geomLatHalfH: Float = 0.22f,
        val geomFwdLoH: Float = 0.20f,
        val geomFwdHiH: Float = 3.00f,
        val geomNBins: Int = 14,
        val geomNowWeight: Float = 0.35f,
        val geomSupportScale: Float = 10.0f,
        val parallaxMaxLag: Int = 6,
        val parallaxMinBaselineM: Float = 0.08f,
        val parallaxMaxZM: Float = 4.2f,
        val parallaxMinPx: Float = 1.0f,
        val parallaxMinGrad: Float = 0.006f,
        val parallaxCostMargin: Float = 0.08f,
        val parallaxMaxFarCost: Float = 0.35f,
        val parallaxMults: FloatArray = floatArrayOf(1.6f, 2.5f, 4.5f),
    )

    data class StepResult(
        val riskGeomM6: Float,
        val groundScaleM: Float,
        val nearestCorridorM: Float,
        val parallaxRefutedPixels: Int = 0,
    )

    private class PrevSupport(val worldX: FloatArray, val worldY: FloatArray, val weight: FloatArray)
    private class ParallaxFrame(
        val gray: FloatArray,
        val fx: Float,
        val fy: Float,
        val cx: Float,
        val cy: Float,
        val rWc: FloatArray,
        val tW: FloatArray,
    )

    private var prevSupport: PrevSupport? = null
    private val parallaxHist = ArrayDeque<ParallaxFrame>(cfg.parallaxMaxLag + 1)

    fun reset() {
        prevSupport = null
        parallaxHist.clear()
    }

    /**
     * Subsample full `288x384` NCHW outputs to `72x96` s4 tensors (`depth_s4`, `dh_s4`, `nrmY_s4`, `ls_s4`)
     * and execute one causal M6 geometric step.
     */
    fun stepFromFullOutputs(
        depth288x384: FloatArray,
        deltaH288x384: FloatArray,
        normalGrav3x288x384: FloatArray,
        logSigma288x384: FloatArray,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        rWc: FloatArray,
        tW: FloatArray,
        causalFwd: FloatArray,
        grayS4: FloatArray? = null,
    ): StepResult {
        val hS4 = EgoGeomContracts.S4_HEIGHT
        val wS4 = EgoGeomContracts.S4_WIDTH
        val nS4 = hS4 * wS4
        val depS4 = FloatArray(nS4)
        val dhS4 = FloatArray(nS4)
        val nrmYS4 = FloatArray(nS4)
        val lsS4 = FloatArray(nS4)
        val nrmYPlane = EgoGeomContracts.DEPTH_FLOATS
        for (y in 0 until hS4) {
            val srcRow = (y * 4) * EgoGeomContracts.WIDTH
            val dstRow = y * wS4
            for (x in 0 until wS4) {
                val srcIdx = srcRow + x * 4
                val dstIdx = dstRow + x
                depS4[dstIdx] = depth288x384[srcIdx]
                dhS4[dstIdx] = deltaH288x384[srcIdx]
                nrmYS4[dstIdx] = normalGrav3x288x384[nrmYPlane + srcIdx]
                lsS4[dstIdx] = logSigma288x384[srcIdx]
            }
        }
        return if (grayS4 != null) {
            stepWithParallax(depS4, dhS4, nrmYS4, lsS4, grayS4, hS4, wS4, fx, fy, cx, cy, rWc, tW, causalFwd)
        } else {
            step(depS4, dhS4, nrmYS4, lsS4, hS4, wS4, fx, fy, cx, cy, rWc, tW, causalFwd)
        }
    }

    /**
     * Causal M6 geometric readout step on `hS4 x wS4` (`72 x 96`) feature maps.
     * Matches `EgoGeomTemporalBEVV3._m6_geom_step` numerically.
     */
    fun step(
        depthS4: FloatArray,
        deltaHS4: FloatArray,
        nrmDownS4: FloatArray,
        logSigmaS4: FloatArray,
        hS4: Int = EgoGeomContracts.S4_HEIGHT,
        wS4: Int = EgoGeomContracts.S4_WIDTH,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        rWc: FloatArray,
        tW: FloatArray,
        causalFwd: FloatArray,
        refutedCount: Int = 0,
    ): StepResult {
        val sub = cfg.geomSubsample
        val sh = (hS4 + sub - 1) / sub
        val sw = (wS4 + sub - 1) / sub
        val n = sh * sw
        val sPix = (cfg.stride * sub).toFloat()
        val invFx = 1f / max(fx, 1f)
        val invFy = 1f / max(fy, 1f)

        val r00 = rWc[0]; val r01 = rWc[1]; val r02 = rWc[2]
        val r10 = rWc[3]; val r11 = rWc[4]; val r12 = rWc[5]
        val r20 = rWc[6]; val r21 = rWc[7]; val r22 = rWc[8]

        val wxArr = FloatArray(n)
        val wyArr = FloatArray(n)
        val wzArr = FloatArray(n)
        val dhArr = FloatArray(n)
        val ndArr = FloatArray(n)
        val lsArr = FloatArray(n)
        val valid = BooleanArray(n)
        val floorMask = BooleanArray(n)

        var validCount = 0
        var floorCount = 0
        var k = 0
        for (vy in 0 until sh) {
            val rowOff = (vy * sub) * wS4
            val vPix = sPix * vy + 0.5f
            val rayY = (vPix - cy) * invFy
            for (ux in 0 until sw) {
                val idx = rowOff + ux * sub
                val z = depthS4[idx]
                val uPix = sPix * ux + 0.5f
                val cxCam = (uPix - cx) * z * invFx
                val cyCam = rayY * z
                val xw = r00 * cxCam + r01 * cyCam + r02 * z
                val yw = r10 * cxCam + r11 * cyCam + r12 * z
                val zw = r20 * cxCam + r21 * cyCam + r22 * z
                wxArr[k] = xw; wyArr[k] = yw; wzArr[k] = zw
                dhArr[k] = deltaHS4[idx]
                ndArr[k] = nrmDownS4[idx]
                lsArr[k] = logSigmaS4[idx]
                val ok = z > 0f && z.isFinite() && xw.isFinite() && yw.isFinite() && zw.isFinite()
                valid[k] = ok
                if (ok) {
                    validCount++
                    if (zw < 0f && abs(ndArr[k]) >= cfg.geomFloorNMin) {
                        floorMask[k] = true
                        floorCount++
                    }
                }
                k++
            }
        }

        if (floorCount < cfg.geomMinFloorPts) {
            floorCount = 0
            for (i in 0 until n) {
                val f = valid[i] && wzArr[i] < 0f
                floorMask[i] = f
                if (f) floorCount++
            }
        }
        if (floorCount < cfg.geomMinFloorPts || validCount == 0) {
            prevSupport = null
            return StepResult(0f, Float.NaN, Float.POSITIVE_INFINITY, refutedCount)
        }

        val floorH = FloatArray(floorCount)
        val floorDh = FloatArray(floorCount)
        val validLs = FloatArray(validCount)
        var fi = 0; var vi = 0
        for (i in 0 until n) {
            if (valid[i]) validLs[vi++] = lsArr[i]
            if (floorMask[i]) {
                floorH[fi] = wzArr[i]
                floorDh[fi] = dhArr[i]
                fi++
            }
        }
        val ground = percentile(floorH, 15.0)
        val scale = -ground
        if (scale < 1e-3f) {
            prevSupport = null
            return StepResult(0f, Float.NaN, Float.POSITIVE_INFINITY, refutedCount)
        }
        val medFloorDh = percentile(floorDh, 50.0)
        val medLs = percentile(validLs, 50.0)

        val fwxRaw = causalFwd[0].toDouble()
        val fwyRaw = causalFwd[1].toDouble()
        val nf = hypot(fwxRaw, fwyRaw)
        if (nf < 1e-7) {
            prevSupport = null
            return StepResult(0f, scale, Float.POSITIVE_INFINITY, refutedCount)
        }
        val fwx = (fwxRaw / nf).toFloat()
        val fwy = (fwyRaw / nf).toFloat()
        val rtx = fwy
        val rty = -fwx

        val keepX = FloatArray(validCount)
        val keepY = FloatArray(validCount)
        val keepW = FloatArray(validCount)
        var keepN = 0
        val twX = tW[0]; val twY = tW[1]
        for (i in 0 until n) {
            if (!valid[i]) continue
            val relH = (wzArr[i] - ground) / scale
            if (relH < cfg.geomRelHLo || relH > cfg.geomRelHHi) continue
            val hHead = (dhArr[i] - medFloorDh) / scale
            if (hHead <= 0f) continue
            val nd = ndArr[i]
            val vert = (1f - nd * nd).coerceIn(0f, 1f)
            if (vert <= 0f) continue
            val conf = exp(-(lsArr[i] - medLs).coerceIn(-2f, 2f).toDouble()).toFloat()
            val w = (relH / 0.22f).coerceIn(0f, 1f) * (hHead / 0.16f).coerceIn(0f, 1f) * vert * conf
            if (w > 0f) {
                keepX[keepN] = wxArr[i] + twX
                keepY[keepN] = wyArr[i] + twY
                keepW[keepN] = w
                keepN++
            }
        }

        val curX = keepX.copyOf(keepN)
        val curY = keepY.copyOf(keepN)
        val curW = keepW.copyOf(keepN)

        val nB = cfg.geomNBins
        val binW = (cfg.geomFwdHiH - cfg.geomFwdLoH) / nB.toFloat()
        var nearestM = Float.POSITIVE_INFINITY

        fun binSupport(px: FloatArray, py: FloatArray, pw: FloatArray, trackNearest: Boolean): DoubleArray {
            val counts = DoubleArray(nB)
            for (i in px.indices) {
                val ox = (px[i] - twX) / scale
                val oy = (py[i] - twY) / scale
                val fwd = ox * fwx + oy * fwy
                val lat = ox * rtx + oy * rty
                if (abs(lat) <= cfg.geomLatHalfH && fwd >= cfg.geomFwdLoH && fwd < cfg.geomFwdHiH) {
                    val b = floor(((fwd - cfg.geomFwdLoH) / binW).toDouble()).toInt().coerceIn(0, nB - 1)
                    counts[b] += pw[i].toDouble()
                    if (trackNearest) {
                        val dM = fwd * scale
                        if (dM < nearestM) nearestM = dM
                    }
                }
            }
            val adj = DoubleArray(nB - 1)
            for (i in 0 until nB - 1) adj[i] = counts[i] + counts[i + 1]
            return adj
        }

        val now = binSupport(curX, curY, curW, trackNearest = true)
        val prev = prevSupport
        val past = if (prev == null) DoubleArray(nB - 1) else binSupport(prev.worldX, prev.worldY, prev.weight, trackNearest = false)
        prevSupport = PrevSupport(curX, curY, curW)

        var maxRisk = 0.0
        for (i in 0 until nB - 1) {
            val sup = cfg.geomNowWeight * now[i] + (1.0 - cfg.geomNowWeight) * min(now[i], past[i])
            val dBin = cfg.geomFwdLoH + binW + binW * i
            val prox = 1.0 / (1.0 + dBin * dBin)
            val risk = (-expm1(-sup / cfg.geomSupportScale)) * prox
            if (risk > maxRisk) maxRisk = risk
        }
        return StepResult(maxRisk.toFloat(), scale, nearestM, refutedCount)
    }

    /**
     * Causal M6 step preceded by fast (<1.5 ms) 3x3 ZNCC two-frame epipolar parallax refutation.
     * Refuted spurious close pixels have their depth pushed to the matched far depth and `delta_h` clamped <= 0.
     */
    fun stepWithParallax(
        depthS4: FloatArray,
        deltaHS4: FloatArray,
        nrmDownS4: FloatArray,
        logSigmaS4: FloatArray,
        grayS4: FloatArray,
        hS4: Int = EgoGeomContracts.S4_HEIGHT,
        wS4: Int = EgoGeomContracts.S4_WIDTH,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        rWc: FloatArray,
        tW: FloatArray,
        causalFwd: FloatArray,
    ): StepResult {
        val invS = 1f / cfg.stride.toFloat()
        val fxS4 = fx * invS; val fyS4 = fy * invS; val cxS4 = cx * invS; val cyS4 = cy * invS

        var bestPrev: ParallaxFrame? = null
        for (idx in parallaxHist.size - 1 downTo 0) {
            val item = parallaxHist[idx]
            val dx = tW[0] - item.tW[0]; val dy = tW[1] - item.tW[1]; val dz = tW[2] - item.tW[2]
            if (sqrt((dx * dx + dy * dy + dz * dz).toDouble()) >= cfg.parallaxMinBaselineM) {
                bestPrev = item
                break
            }
        }
        if (parallaxHist.size >= cfg.parallaxMaxLag) parallaxHist.removeFirst()
        parallaxHist.addLast(ParallaxFrame(grayS4.copyOf(), fxS4, fyS4, cxS4, cyS4, rWc.copyOf(), tW.copyOf()))

        if (bestPrev == null) {
            return step(depthS4, deltaHS4, nrmDownS4, logSigmaS4, hS4, wS4, fx, fy, cx, cy, rWc, tW, causalFwd, 0)
        }

        // Relative transform cur -> prev: R_rel = R_prev^T R_cur, t_rel = R_prev^T (t_cur - t_prev)
        val rp = bestPrev.rWc
        val rRel = FloatArray(9)
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                rRel[i * 3 + j] = rp[0 * 3 + i] * rWc[0 * 3 + j] +
                    rp[1 * 3 + i] * rWc[1 * 3 + j] +
                    rp[2 * 3 + i] * rWc[2 * 3 + j]
            }
        }
        val dtx = tW[0] - bestPrev.tW[0]; val dty = tW[1] - bestPrev.tW[1]; val dtz = tW[2] - bestPrev.tW[2]
        val tRelX = rp[0] * dtx + rp[3] * dty + rp[6] * dtz
        val tRelY = rp[1] * dtx + rp[4] * dty + rp[7] * dtz
        val tRelZ = rp[2] * dtx + rp[5] * dty + rp[8] * dtz

        var zOut = depthS4
        var dhOut = deltaHS4
        var copied = false
        var refuted = 0

        val sub = cfg.geomSubsample
        val mults = cfg.parallaxMults
        val prevGray = bestPrev.gray
        val maxM = mults[mults.size - 1]
        val invFxC = 1f / fxS4; val invFyC = 1f / fyS4

        var y = 2
        while (y < hS4 - 2) {
            val rowOff = y * wS4
            val ry = (y - cyS4) * invFyC
            var x = 2
            while (x < wS4 - 2) {
                val idx = rowOff + x
                val z = depthS4[idx]
                if (z > 0.25f && z < cfg.parallaxMaxZM && z.isFinite()) {
                    val gx = grayS4[idx + 1] - grayS4[idx - 1]
                    val gy = grayS4[idx + wS4] - grayS4[idx - wS4]
                    if (gx * gx + gy * gy >= cfg.parallaxMinGrad * cfg.parallaxMinGrad) {
                        val rx = (x - cxS4) * invFxC
                        val drx = rRel[0] * rx + rRel[1] * ry + rRel[2]
                        val dry = rRel[3] * rx + rRel[4] * ry + rRel[5]
                        val drz = rRel[6] * rx + rRel[7] * ry + rRel[8]

                        val pz0 = z * drz + tRelZ
                        val zMax = min(z * maxM, 20f)
                        val pzMax = zMax * drz + tRelZ
                        if (pz0 > 0.15f && pzMax > 0.15f) {
                            val up0 = bestPrev.fx * ((z * drx + tRelX) / pz0) + bestPrev.cx
                            val vp0 = bestPrev.fy * ((z * dry + tRelY) / pz0) + bestPrev.cy
                            val upM = bestPrev.fx * ((zMax * drx + tRelX) / pzMax) + bestPrev.cx
                            val vpM = bestPrev.fy * ((zMax * dry + tRelY) / pzMax) + bestPrev.cy
                            if (up0 >= 2f && up0 < wS4 - 2f && vp0 >= 2f && vp0 < hS4 - 2f &&
                                upM >= 2f && upM < wS4 - 2f && vpM >= 2f && vpM < hS4 - 2f &&
                                hypot((up0 - upM).toDouble(), (vp0 - vpM).toDouble()) >= cfg.parallaxMinPx
                            ) {
                                val cObs = zncc3x3(grayS4, prevGray, hS4, wS4, x, y, up0, vp0)
                                var bestFarCost = 9f
                                var bestFarZ = z
                                for (m in mults) {
                                    val zh = min(z * m, 20f)
                                    val pzh = zh * drz + tRelZ
                                    if (pzh <= 0.15f) continue
                                    val uph = bestPrev.fx * ((zh * drx + tRelX) / pzh) + bestPrev.cx
                                    val vph = bestPrev.fy * ((zh * dry + tRelY) / pzh) + bestPrev.cy
                                    if (uph < 2f || uph >= wS4 - 2f || vph < 2f || vph >= hS4 - 2f) continue
                                    val ch = zncc3x3(grayS4, prevGray, hS4, wS4, x, y, uph, vph)
                                    if (ch < bestFarCost) {
                                        bestFarCost = ch
                                        bestFarZ = zh
                                    }
                                }
                                if (bestFarCost < cfg.parallaxMaxFarCost && (cObs - bestFarCost) >= cfg.parallaxCostMargin) {
                                    if (!copied) {
                                        zOut = depthS4.copyOf()
                                        dhOut = deltaHS4.copyOf()
                                        copied = true
                                    }
                                    zOut[idx] = bestFarZ
                                    if (dhOut[idx] > 0f) dhOut[idx] = 0f
                                    refuted++
                                }
                            }
                        }
                    }
                }
                x += sub
            }
            y += sub
        }

        return step(zOut, dhOut, nrmDownS4, logSigmaS4, hS4, wS4, fx, fy, cx, cy, rWc, tW, causalFwd, refuted)
    }

    private fun zncc3x3(
        cur: FloatArray,
        prev: FloatArray,
        h: Int,
        w: Int,
        xc: Int,
        yc: Int,
        up: Float,
        vp: Float,
    ): Float {
        var mc = 0f; var mp = 0f
        val pc = FloatArray(9)
        val pp = FloatArray(9)
        var k = 0
        for (dy in -1..1) {
            val cyRow = (yc + dy) * w
            val py = vp + dy
            val y0 = floor(py.toDouble()).toInt().coerceIn(0, h - 2)
            val fy = py - y0
            val fy0 = 1f - fy
            val pRow0 = y0 * w
            val pRow1 = pRow0 + w
            for (dx in -1..1) {
                val cVal = cur[cyRow + xc + dx]
                val px = up + dx
                val x0 = floor(px.toDouble()).toInt().coerceIn(0, w - 2)
                val fx = px - x0
                val pVal = (1f - fx) * fy0 * prev[pRow0 + x0] +
                    fx * fy0 * prev[pRow0 + x0 + 1] +
                    (1f - fx) * fy * prev[pRow1 + x0] +
                    fx * fy * prev[pRow1 + x0 + 1]
                pc[k] = cVal; pp[k] = pVal
                mc += cVal; mp += pVal
                k++
            }
        }
        mc /= 9f; mp /= 9f
        var nc = 1e-6f; var np = 1e-6f; var dot = 0f
        for (i in 0 until 9) {
            val ac = pc[i] - mc
            val ap = pp[i] - mp
            nc += ac * ac
            np += ap * ap
            dot += ac * ap
        }
        return 1f - dot / sqrt(nc * np)
    }

    private fun percentile(values: FloatArray, qPct: Double): Float {
        values.sort()
        val pos = (qPct / 100.0) * (values.size - 1)
        val lo = floor(pos).toInt().coerceIn(0, values.size - 1)
        val hi = min(lo + 1, values.size - 1)
        val frac = (pos - lo).toFloat()
        return values[lo] * (1f - frac) + values[hi] * frac
    }
}
