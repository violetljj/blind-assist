package com.linnan.blindassist.egoclear

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.SystemClock
import java.io.File
import java.nio.FloatBuffer
import java.util.ArrayDeque
import kotlin.math.exp

/**
 * Isolated, reusable 2-stage streaming engine for EgoClear-v6.6 / v10.5 / v10.6 / v11 (Omni-BlindAssist).
 *
 * Supports:
 *  - Stage 1 Encoder on Qualcomm SM8650 Hexagon v75 HTP NPU (SNPE) or ONNX Runtime CPU
 *  - Stage 2 Step (2-lag parallax stereo + ConvGRU BEV memory + survival heads) on EITHER
 *    Hexagon v75 HTP NPU (~4.9 ms, 100% VTCM SRAM!) OR ONNX Runtime CPU fallback
 *  - Automatic decoding of EgoClear-v11 6-channel Omni survival curves:
 *    [0: center_static, 1: left_bypass, 2: right_bypass, 3: head_hazard, 4: drop_off, 5: dyn_closing]
 */
class EgoClearStreamingEngine(
    encoderModel: File?,
    stepModel: File?,
    val backend: Backend = Backend.CPU_4T,
    val stepBackend: Backend = Backend.CPU_4T,
    private val snpeEncoder: EgoClearSnpeEncoder? = null,
    val profile: ModelProfile = ModelProfile.V66_DAV2,
    private val snpeStep: EgoClearSnpeStep? = null,
    private val applyNearRangeGeomClamp: Boolean? = null,
) : AutoCloseable {

    enum class ModelProfile(
        val featH: Long,
        val featW: Long,
        val auxCh: Long,
        val auxStepPrefix: String,
    ) {
        V66_DAV2(42L, 74L, 4L, "dc"),
        V10_MOGE2(84L, 148L, 10L, "geom"),
    }

    enum class Backend(val tag: String) {
        CPU_4T("cpu_4t"),
        CPU_6T("cpu_6t"),
        XNNPACK_4T("xnnpack_4t"),
        XNNPACK_6T("xnnpack_6t"),
        NNAPI("nnapi"),
        HTP_V75_NPU("htp_v75_npu"),
    }

    enum class BypassAdvice(val shortZh: String, val spokenZh: String) {
        CLEAR("", ""),
        GO_LEFT("← 左侧可绕", "左侧可绕行"),
        GO_RIGHT("右侧可绕 →", "右侧可绕行"),
        BLOCKED_ALL("✕ 前方全堵", "前方全堵，请停下"),
    }

    data class FrameCache(
        val f2d: FloatArray,
        val dc: FloatArray,
        val kScaled: FloatArray,
        val rC2w: FloatArray,
        val tWorld: FloatArray,
        val fwdWorld: FloatArray,
    )

    data class StepOutput(
        val corrProbsMonotonic: FloatArray,
        val uncLogits: FloatArray,
        val staticLogits: FloatArray,
        val dynamicLogits: FloatArray,
        val pAt4m: Float,
        val pAt44m: Float,
        val uncProbAt4m: Float,
        val monotonicOk: Boolean,
        val alertActive: Boolean,
        val encoderMs: Double,
        val stepMs: Double,
        val totalMs: Double,
        val geomCorridorMinMeters: Float? = null,
        // EgoClear-v11 Omni-BlindAssist outputs (null on legacy v6.6/v10.5/v10.6 models)
        val leftBypassProbs: FloatArray? = null,
        val rightBypassProbs: FloatArray? = null,
        val headHazardProbs: FloatArray? = null,
        val dropOffProbs: FloatArray? = null,
        val dynClosingProbs: FloatArray? = null,
        val bypassAdvice: BypassAdvice = BypassAdvice.CLEAR,
        val dropOffHitMeters: Float? = null,
        val headHitMeters: Float? = null,
        val dynClosingHitMeters: Float? = null,
        val isUnknownUnobservable: Boolean = false,
    )

    private val env: OrtEnvironment?
    private val encoderOptions: OrtSession.SessionOptions?
    private val stepOptions: OrtSession.SessionOptions?
    private val encoderSession: OrtSession?
    private val stepSession: OrtSession?

    val encoderLoadMs: Double
    val stepLoadMs: Double
    val hasOmniOutputs: Boolean
    val isStepOnNpu: Boolean get() = snpeStep != null

    private val history = ArrayDeque<FrameCache>(3)
    private var memState = FloatArray(MEM_SIZE)
    private var ageState = FloatArray(AGE_SIZE) { 10.0f }

    private var alertOn = false
    private var dropOn = false
    private var prevScore = 0.0f
    private var hasPrevScore = false
    private val geomBinCounts = IntArray(BEV_NZ)
    private val geomSupportMem = FloatArray(BEV_NZ)

    init {
        val needOrt = (snpeEncoder == null) || (snpeStep == null)
        env = if (needOrt) OrtEnvironment.getEnvironment() else null

        if (snpeEncoder != null) {
            encoderOptions = null
            encoderSession = null
            encoderLoadMs = snpeEncoder.loadMs
        } else {
            val enc = requireNotNull(encoderModel) { "encoderModel required when snpeEncoder is null" }
            require(enc.isFile) { "Missing encoder model: ${enc.absolutePath}" }
            encoderOptions = buildOptions(backend)
            val t0 = SystemClock.elapsedRealtimeNanos()
            encoderSession = env!!.createSession(enc.absolutePath, encoderOptions)
            encoderLoadMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
        }

        if (snpeStep != null) {
            stepOptions = null
            stepSession = null
            stepLoadMs = snpeStep.loadMs
            hasOmniOutputs = snpeStep.hasOmni
        } else {
            val stp = requireNotNull(stepModel) { "stepModel required when snpeStep is null" }
            require(stp.isFile) { "Missing step model: ${stp.absolutePath}" }
            stepOptions = buildOptions(stepBackend)
            val t1 = SystemClock.elapsedRealtimeNanos()
            stepSession = env!!.createSession(stp.absolutePath, stepOptions)
            stepLoadMs = (SystemClock.elapsedRealtimeNanos() - t1) / 1_000_000.0
            hasOmniOutputs = stepSession.outputNames.contains("omni_probs")
        }
    }

    fun resetState() {
        history.clear()
        memState = FloatArray(MEM_SIZE)
        ageState = FloatArray(AGE_SIZE) { 10.0f }
        alertOn = false
        dropOn = false
        prevScore = 0.0f
        hasPrevScore = false
        geomSupportMem.fill(0.0f)
    }

    fun step(
        rgbNchw: FloatArray,
        kScaled: FloatArray,
        rC2w: FloatArray,
        tWorld: FloatArray,
        fwdWorld: FloatArray,
        dsMeters: Float,
        alertThreshold: Float = 0.55f,
        alertHorizonIdx: Int = IDX_4M,
    ): StepOutput {
        val tTotalStart = SystemClock.elapsedRealtimeNanos()

        // Stage 1: Encoder (Hexagon v75 HTP NPU or ONNX Runtime CPU)
        val tEncStart = tTotalStart
        val f2dArr: FloatArray
        val dcArr: FloatArray
        if (snpeEncoder != null) {
            val (f, d) = snpeEncoder.encode(rgbNchw, kScaled, rC2w)
            f2dArr = f
            dcArr = d
        } else {
            val e = env!!
            OnnxTensor.createTensor(e, FloatBuffer.wrap(rgbNchw), SHAPE_RGB).use { rgbT ->
                OnnxTensor.createTensor(e, FloatBuffer.wrap(kScaled), SHAPE_K).use { kT ->
                    OnnxTensor.createTensor(e, FloatBuffer.wrap(rC2w), SHAPE_R).use { rT ->
                        encoderSession!!.run(
                            mapOf("rgb" to rgbT, "K_scaled" to kT, "R_c2w" to rT),
                        ).use { res ->
                            f2dArr = copyFloatTensor(res[0] as OnnxTensor)
                            dcArr = copyFloatTensor(res[1] as OnnxTensor)
                        }
                    }
                }
            }
        }
        val encoderMs = (SystemClock.elapsedRealtimeNanos() - tEncStart) / 1_000_000.0

        val cur = FrameCache(
            f2d = f2dArr,
            dc = dcArr,
            kScaled = kScaled.copyOf(),
            rC2w = rC2w.copyOf(),
            tWorld = tWorld.copyOf(),
            fwdWorld = fwdWorld.copyOf(),
        )
        if (history.size == 3) history.removeFirst()
        history.addLast(cur)
        val histList = history.toList()
        val prev1 = if (histList.size >= 2) histList[histList.size - 2] else cur
        val prev2 = if (histList.size >= 3) histList[histList.size - 3] else prev1

        // Stage 2: 2-Lag Parallax Stereo + Metric-Warped ConvGRU Memory + Survival Heads
        val tStepStart = SystemClock.elapsedRealtimeNanos()
        val staticArr: FloatArray
        val dynamicArr: FloatArray
        val corrArr: FloatArray
        val uncArr: FloatArray
        var omniFlat: FloatArray? = null
        val dsArr = floatArrayOf(dsMeters)

        if (snpeStep != null) {
            val inArrays = arrayOf(
                cur.f2d, cur.dc,
                prev1.f2d, prev1.dc,
                prev2.f2d, prev2.dc,
                cur.kScaled, cur.rC2w, cur.tWorld, cur.fwdWorld,
                prev1.kScaled, prev1.rC2w, prev1.tWorld, prev1.fwdWorld,
                prev2.kScaled, prev2.rC2w, prev2.tWorld, dsArr,
                memState, ageState,
            )
            val out = snpeStep.execute(inArrays)
            staticArr = out.staticLogits
            dynamicArr = out.dynamicLogits
            corrArr = out.corrProbs
            uncArr = out.uncLogits
            omniFlat = out.omniProbs
            memState = out.memNew
            ageState = out.ageNew
        } else {
            val e = env!!
            val tensors = ArrayList<OnnxTensor>(20)
            try {
                fun put(arr: FloatArray, shape: LongArray): OnnxTensor {
                    val t = OnnxTensor.createTensor(e, FloatBuffer.wrap(arr), shape)
                    tensors.add(t)
                    return t
                }
                val shapeF2d = longArrayOf(1, 64, profile.featH, profile.featW)
                val shapeAux = longArrayOf(1, profile.auxCh, profile.featH, profile.featW)
                val auxPfx = profile.auxStepPrefix
                val feed = LinkedHashMap<String, OnnxTensor>(20).apply {
                    put("f2d_t", put(cur.f2d, shapeF2d))
                    put("${auxPfx}_t", put(cur.dc, shapeAux))
                    put("f2d_1", put(prev1.f2d, shapeF2d))
                    put("${auxPfx}_1", put(prev1.dc, shapeAux))
                    put("f2d_2", put(prev2.f2d, shapeF2d))
                    put("${auxPfx}_2", put(prev2.dc, shapeAux))
                    put("K_t", put(cur.kScaled, SHAPE_K))
                    put("R_t", put(cur.rC2w, SHAPE_R))
                    put("t_t", put(cur.tWorld, SHAPE_T))
                    put("fwd_t", put(cur.fwdWorld, SHAPE_T))
                    put("K_1", put(prev1.kScaled, SHAPE_K))
                    put("R_1", put(prev1.rC2w, SHAPE_R))
                    put("t_1", put(prev1.tWorld, SHAPE_T))
                    put("fwd_1", put(prev1.fwdWorld, SHAPE_T))
                    put("K_2", put(prev2.kScaled, SHAPE_K))
                    put("R_2", put(prev2.rC2w, SHAPE_R))
                    put("t_2", put(prev2.tWorld, SHAPE_T))
                    put("ds_t", put(dsArr, SHAPE_DS))
                    put("mem_prev", put(memState, SHAPE_MEM))
                    put("age_prev", put(ageState, SHAPE_AGE))
                }
                stepSession!!.run(feed).use { res ->
                    staticArr = copyFloatTensor(res[0] as OnnxTensor)
                    dynamicArr = copyFloatTensor(res[1] as OnnxTensor)
                    corrArr = copyFloatTensor(res[2] as OnnxTensor)
                    uncArr = copyFloatTensor(res[3] as OnnxTensor)
                    if (hasOmniOutputs && res.size() >= 7) {
                        omniFlat = copyFloatTensor(res[4] as OnnxTensor)
                        memState = copyFloatTensor(res[5] as OnnxTensor)
                        ageState = copyFloatTensor(res[6] as OnnxTensor)
                    } else {
                        memState = copyFloatTensor(res[4] as OnnxTensor)
                        ageState = copyFloatTensor(res[5] as OnnxTensor)
                    }
                }
            } finally {
                for (t in tensors) t.close()
            }
        }
        val stepMs = (SystemClock.elapsedRealtimeNanos() - tStepStart) / 1_000_000.0
        val totalMs = (SystemClock.elapsedRealtimeNanos() - tTotalStart) / 1_000_000.0

        // Compute exact 3D physical corridor clearance from V10_MOGE2 geom_cues (<0.15 ms);
        // apply near-range de-front-loading clamp on legacy v10.5 (disabled by default on calibrated v11 Omni).
        var geomMinM: Float? = null
        val doClamp = applyNearRangeGeomClamp ?: (!hasOmniOutputs)
        if (profile == ModelProfile.V10_MOGE2 && cur.dc.size >= 10 * 84 * 148) {
            geomMinM = scanMetricGeomCorridor(cur.dc, kScaled, rC2w, fwdWorld, corrArr, applyClamp = doClamp)
        }

        var monoOk = true
        for (i in 1 until corrArr.size) {
            if (corrArr[i] - corrArr[i - 1] < -1e-6f) {
                monoOk = false
                break
            }
        }
        val p4m = corrArr[IDX_4M]
        val p44m = corrArr[IDX_44M]
        val hIdx = alertHorizonIdx.coerceIn(0, corrArr.lastIndex)
        val pHorizon = corrArr[hIdx]
        val uncProb = (1.0 / (1.0 + exp(-uncArr[hIdx].toDouble()))).toFloat()
        var uncMinBand = 1.0f
        for (i in 4..IDX_4M) {
            val u = (1.0 / (1.0 + exp(-uncArr[i].toDouble()))).toFloat()
            if (u < uncMinBand) uncMinBand = u
        }
        // Only abstain on genuine whole-frame optical blindness (blackout/finger/whiteout/severe blur
        // saturate ALL bins >= 0.96..0.999), never silencing real close obstacles in dim indoor rooms.
        val isUnknown = hasOmniOutputs && uncMinBand >= UNKNOWN_ABSTAIN_THR

        // Zero-delay causal slope-aware hysteresis
        val delta = if (hasPrevScore) pHorizon - prevScore else 0.0f
        val offThr = alertThreshold * 0.72f
        val fastRise = delta >= 0.08f && pHorizon >= alertThreshold * 0.90f
        alertOn = if (isUnknown) {
            false
        } else if (!alertOn) {
            pHorizon >= alertThreshold || fastRise
        } else {
            pHorizon >= offThr
        }
        prevScore = pHorizon
        hasPrevScore = true

        // Decode EgoClear-v11 6-channel Omni outputs [6 x 24]
        var leftProbs: FloatArray? = null
        var rightProbs: FloatArray? = null
        var headProbs: FloatArray? = null
        var dropProbs: FloatArray? = null
        var dynCloseProbs: FloatArray? = null
        var bypass = BypassAdvice.CLEAR
        var dropHitM: Float? = null
        var headHitM: Float? = null
        var dynCloseHitM: Float? = null

        val om = omniFlat
        if (om != null && om.size >= 6 * BEV_NZ) {
            leftProbs = om.copyOfRange(1 * BEV_NZ, 2 * BEV_NZ)
            rightProbs = om.copyOfRange(2 * BEV_NZ, 3 * BEV_NZ)
            headProbs = om.copyOfRange(3 * BEV_NZ, 4 * BEV_NZ)
            dropProbs = om.copyOfRange(4 * BEV_NZ, 5 * BEV_NZ)
            dynCloseProbs = om.copyOfRange(5 * BEV_NZ, 6 * BEV_NZ)

            if (!isUnknown) {
                val dropHorizon = minOf(hIdx + 2, 11) // Check drop-offs up to 2.7m ahead
                val dropThr = if (dropOn) DROP_OFF_THR else DROP_ON_THR
                dropHitM = firstHitMeters(dropProbs, dropHorizon, dropThr)
                dropOn = (dropHitM != null)
                headHitM = firstHitMeters(headProbs, hIdx, 0.48f)
                dynCloseHitM = firstHitMeters(dynCloseProbs, minOf(hIdx + 3, 14), 0.50f)

                if (alertOn) {
                    val bIdx = minOf(maxOf(hIdx, 7), 11) // Evaluate bypass openness at 1.9..2.7m
                    val pL = leftProbs[bIdx]
                    val pR = rightProbs[bIdx]
                    val clearThr = 0.45f
                    bypass = when {
                        pL < clearThr && pR < clearThr -> if (pL <= pR) BypassAdvice.GO_LEFT else BypassAdvice.GO_RIGHT
                        pL < clearThr -> BypassAdvice.GO_LEFT
                        pR < clearThr -> BypassAdvice.GO_RIGHT
                        else -> BypassAdvice.BLOCKED_ALL
                    }
                }
            } else {
                dropOn = false
            }
        }

        return StepOutput(
            corrProbsMonotonic = corrArr,
            uncLogits = uncArr,
            staticLogits = staticArr,
            dynamicLogits = dynamicArr,
            pAt4m = p4m,
            pAt44m = p44m,
            uncProbAt4m = uncProb,
            monotonicOk = monoOk,
            alertActive = alertOn,
            encoderMs = encoderMs,
            stepMs = stepMs,
            totalMs = totalMs,
            geomCorridorMinMeters = geomMinM,
            leftBypassProbs = leftProbs,
            rightBypassProbs = rightProbs,
            headHazardProbs = headProbs,
            dropOffProbs = dropProbs,
            dynClosingProbs = dynCloseProbs,
            bypassAdvice = bypass,
            dropOffHitMeters = dropHitM,
            headHitMeters = headHitM,
            dynClosingHitMeters = dynCloseHitM,
            isUnknownUnobservable = isUnknown,
        )
    }

    private fun firstHitMeters(probs: FloatArray, maxIdx: Int, thr: Float): Float? {
        val lim = maxIdx.coerceIn(0, probs.lastIndex)
        for (i in 0..lim) {
            if (probs[i] >= thr) return 0.30f + (i + 1) * 0.20f
        }
        return null
    }

    private fun scanMetricGeomCorridor(
        geom: FloatArray,
        kScaled: FloatArray,
        rC2w: FloatArray,
        fwdWorld: FloatArray,
        corrProbs: FloatArray,
        applyClamp: Boolean,
    ): Float? {
        val h = 84
        val w = 148
        val plane = h * w
        val offZ = 0
        val offVert = 6 * plane
        val offMsk = 7 * plane

        val fx = kScaled[0].coerceAtLeast(10f)
        val fy = kScaled[1].coerceAtLeast(10f)
        val cx = kScaled[2]
        val cy = kScaled[3]
        val r00 = rC2w[0]; val r01 = rC2w[1]; val r02 = rC2w[2]
        val r10 = rC2w[3]; val r11 = rC2w[4]; val r12 = rC2w[5]
        val r20 = rC2w[6]; val r21 = rC2w[7]; val r22 = rC2w[8]
        val fxW = fwdWorld[0]; val fyW = fwdWorld[1]
        val rxW = fyW; val ryW = -fxW

        geomBinCounts.fill(0)
        val sx = RGB_W.toFloat() / w.toFloat()
        val sy = RGB_H.toFloat() / h.toFloat()

        var vy = 0
        while (vy < h) {
            val vImg = (vy + 0.5f) * sy
            val ryCam = (vImg - cy) / fy
            val rowOff = vy * w
            var vx = 0
            while (vx < w) {
                val idx = rowOff + vx
                if (geom[offMsk + idx] > 0.50f && geom[offVert + idx] > 0.42f) {
                    val z = geom[offZ + idx]
                    if (z in 0.25f..5.10f) {
                        val uImg = (vx + 0.5f) * sx
                        val xc = z * ((uImg - cx) / fx)
                        val yc = z * ryCam
                        val wx = r00 * xc + r01 * yc + r02 * z
                        val wy = r10 * xc + r11 * yc + r12 * z
                        val wz = r20 * xc + r21 * yc + r22 * z
                        if (wz in -1.15f..0.48f) {
                            val xRight = wx * rxW + wy * ryW
                            if (xRight in -0.36f..0.36f) {
                                val zFwd = wx * fxW + wy * fyW
                                if (zFwd in 0.30f..5.099f) {
                                    val bin = ((zFwd - 0.30f) / 0.20f).toInt().coerceIn(0, BEV_NZ - 1)
                                    geomBinCounts[bin]++
                                }
                            }
                        }
                    }
                }
                vx += 2
            }
            vy += 2
        }

        var cumPix = 0
        var minClearanceM: Float? = null
        var runningMaxP = 0.0f
        for (k in 0 until BEV_NZ) {
            cumPix += geomBinCounts[k]
            val zEdge = 0.30f + (k + 1) * 0.20f
            if (minClearanceM == null && cumPix >= 12) {
                minClearanceM = zEdge
            }
            if (applyClamp) {
                val instSupport = ((cumPix - 6).toFloat() / 18.0f).coerceIn(0.0f, 1.0f)
                val memSupport = maxOf(instSupport, geomSupportMem[k] * 0.55f)
                geomSupportMem[k] = memSupport
                val farBlend = ((zEdge - 2.30f) / 0.80f).coerceIn(0.0f, 1.0f)
                val effSupport = maxOf(memSupport, farBlend)
                val pCalib = corrProbs[k] * (0.14f + 0.86f * effSupport)
                if (pCalib > runningMaxP) runningMaxP = pCalib
                corrProbs[k] = runningMaxP
            }
        }
        return minClearanceM
    }

    override fun close() {
        encoderSession?.close()
        stepSession?.close()
        encoderOptions?.close()
        stepOptions?.close()
    }

    private fun buildOptions(b: Backend): OrtSession.SessionOptions {
        return OrtSession.SessionOptions().apply {
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            when (b) {
                Backend.CPU_4T, Backend.HTP_V75_NPU -> setIntraOpNumThreads(4)
                Backend.CPU_6T -> setIntraOpNumThreads(6)
                Backend.XNNPACK_4T -> {
                    setIntraOpNumThreads(4)
                    addXnnpack(mapOf("intra_op_num_threads" to "4"))
                }
                Backend.XNNPACK_6T -> {
                    setIntraOpNumThreads(6)
                    addXnnpack(mapOf("intra_op_num_threads" to "6"))
                }
                Backend.NNAPI -> {
                    setIntraOpNumThreads(4)
                    addNnapi()
                }
            }
        }
    }

    private fun copyFloatTensor(tensor: OnnxTensor): FloatArray {
        val fb = tensor.floatBuffer
        val out = FloatArray(fb.remaining())
        fb.get(out)
        return out
    }

    companion object {
        const val RGB_H = 294
        const val RGB_W = 518
        const val RGB_FLOATS = 3 * RGB_H * RGB_W
        const val BEV_NZ = 24
        const val BEV_NX = 15
        const val IDX_4M = 18
        const val IDX_44M = 20
        const val UNKNOWN_ABSTAIN_THR = 0.96f
        const val DROP_ON_THR = 0.62f
        const val DROP_OFF_THR = 0.48f

        private const val MEM_SIZE = 128 * BEV_NZ * BEV_NX
        private const val AGE_SIZE = BEV_NZ * BEV_NX

        private val SHAPE_RGB = longArrayOf(1, 3, RGB_H.toLong(), RGB_W.toLong())
        private val SHAPE_K = longArrayOf(1, 4)
        private val SHAPE_R = longArrayOf(1, 3, 3)
        private val SHAPE_T = longArrayOf(1, 3)
        private val SHAPE_DS = longArrayOf(1)
        private val SHAPE_MEM = longArrayOf(1, 128, BEV_NZ.toLong(), BEV_NX.toLong())
        private val SHAPE_AGE = longArrayOf(1, 1, BEV_NZ.toLong(), BEV_NX.toLong())
    }
}
