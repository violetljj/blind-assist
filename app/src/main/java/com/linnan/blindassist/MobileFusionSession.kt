package com.linnan.blindassist

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import com.linnan.blindassist.capture.*
import com.linnan.blindassist.vision.*
import org.json.JSONObject
import androidx.compose.ui.graphics.asImageBitmap
import java.io.File
import java.util.concurrent.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

internal data class MobileFusionView(
    val message: String = "准备模型…", val ready: Boolean = false,
    val head: Boolean = false, val body: Boolean = false,
    val tofHead: Boolean = false, val tofBody: Boolean = false,
    val rgbHead: Boolean = false, val rgbBody: Boolean = false,
    val anchor: String = "NONE", val heightM: Double = 1.35,
    val heightSource: String = "默认 1.35 m，建议先低头标定", val latencyMs: Long = 0,
    val receiptNs: Long = 0, val sequence: Long = -1, val image: Bitmap? = null,
    val processed: Long = 0, val skipped: Long = 0,
    val inputEpoch: Long = -1,
    val terminal: Boolean = false,
    // Demo presentation and v7w gating; NaN distances mean "not measured in that box".
    val demoRules: Boolean = true, val walking: Boolean = false, val limitM: Double = Double.NaN,
    val headDistanceM: Double = Double.NaN, val bodyDistanceM: Double = Double.NaN,
    val pitchDeg: Double = Double.NaN, val rollDeg: Double = Double.NaN, val poseOk: Boolean = false,
    val boxes: List<List<Pair<Float, Float>?>> = emptyList(), val tofGrid: FloatArray? = null,
    val extrinsicPitchDeg: Double = Double.NaN,
) {
    val nearestM: Double get() = listOf(headDistanceM, bodyDistanceM).filter { it.isFinite() }.minOrNull() ?: Double.NaN
}

/** Foreground, opt-in engineering path. Speech/vibration belong to the Activity; this session never
 * records evaluation labels. There is one H3 consumer and bounded latest-only RGB work.
 * [demoRules] applies the EXPLORE pitch-0 extrinsic and v7w walk/still gate ([MobileFusionDemoRules]);
 * off reproduces the frozen v2 path. */
internal class MobileFusionSession(
    private val application: Application, private val directory: File,
    private val atomHost: String, private val tofHost: String,
    private val demoRules: Boolean = true,
    private val publishFrame: (androidx.compose.ui.graphics.ImageBitmap, Long) -> Unit = { _, _ -> },
    private val publish: (MobileFusionView) -> Unit,
) : AutoCloseable {
    private val lease = MobileFusionLease.acquire()
    private val closed = AtomicBoolean()
    private val inputs = MobileFusionInputBuffer()
    private val tofWork = ThreadPoolExecutor(1,1,0,TimeUnit.MILLISECONDS,ArrayBlockingQueue(2),
        { r -> Thread(r,"fusion-tof") },ThreadPoolExecutor.AbortPolicy())
    private val rgbWork = Executors.newSingleThreadExecutor { Thread(it,"fusion-rgb") }
    // Display-only preview, decoupled from inference, as in the capture page: one worker always
    // decodes the newest camera frame and older ones are skipped.
    // Low priority and at most ~15 fps so preview work never starves ToF/RGB inference.
    private val displayWork = Executors.newSingleThreadExecutor { r ->
        Thread({ android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND); r.run() },"fusion-display")
    }
    @Volatile private var displayLastNs = 0L
    /** Preview frame interval; the Activity raises it when the phone heats up. */
    @Volatile var previewIntervalNs = 66_000_000L
    private val displayLatest = AtomicReference<AtomCameraCaptureFrame?>()
    private val displayQueued = AtomicBoolean()
    private var displayPixels = IntArray(0)
    private val displayProcessor by lazy {
        runCatching { AtomPreviewPixels.fromRemap(File(directory,"atom_remap.bin").readBytes()) }.getOrElse { AtomPreviewPixels() }
    }
    private val rgbPending = AtomicReference<RgbJob?>()
    private val rgbBusy = AtomicBoolean()
    @Volatile private var cameraSource: AtomMjpegCaptureSource? = null
    @Volatile private var tofSource: TcpCnhCaptureSource? = null
    @Volatile private var imuSource: TcpImuCaptureSource? = null
    // Per-stream reconnect without reloading models. A generation per stream ignores late callbacks
    // from a source that has been replaced; a watchdog also restarts a stream that went silent.
    private enum class Stream(val label:String) { CAMERA("相机"), TOF("ToF"), IMU("IMU") }
    private val reconnects = Executors.newSingleThreadScheduledExecutor { Thread(it,"fusion-reconnect") }
    private val streamGeneration = java.util.EnumMap<Stream,Long>(Stream::class.java).apply { Stream.values().forEach { put(it,0L) } }
    private val streamLastNs = java.util.EnumMap<Stream,Long>(Stream::class.java)
    private val streamPending = java.util.EnumSet.noneOf(Stream::class.java)
    private val streamAttempts = java.util.EnumMap<Stream,Int>(Stream::class.java)
    @Volatile private var reconnectNote = ""
    val reconnectCount = AtomicLong()
    @Volatile private var camera: MobileFusionCamera? = null
    private var tofEngine: CnhStrongEngine? = null
    private var depth: Dav2DepthRuntime? = null
    private var geometry: AtomRgbGeometry? = null
    private val votes = AtomRgbTwoOfThree()
    private val gait = MobileGaitDetector()
    @Volatile private var extrinsicPitch = Double.NaN
    private var rgbEpoch = -1L
    private var tofEpoch = -1L
    private var rgbPreviousSequence = -1L
    @Volatile private var height = 1.35
    @Volatile private var heightSource = "默认 1.35 m，建议先低头标定"
    private val calibrateRequested = AtomicBoolean()
    private var floor: MobileFloorHeight? = null
    private val skipped = AtomicLong()
    private val processed = AtomicLong()
    private val cameraFrames = AtomicLong()
    private val cameraFirstReceiptNs = AtomicLong()
    private val cameraLastReceiptNs = AtomicLong()
    private val tofQueueRejected = AtomicLong()
    private val rgbPendingReplaced = AtomicLong()
    private var previousLogMs = 0.0
    private val receipt = File(directory,"live-${System.currentTimeMillis()}.jsonl")
    private data class RgbJob(val frame: CnhCaptureFrame, val receiptNs: Long, val image: AtomCameraCaptureFrame?,
        val imuReceiptNs: Long, val pitch: Double, val roll: Double, val tof: CnhStrongResult, val epoch: Long,
        val tofQueueMs: Double, val tofWorkMs: Double)

    fun start() {
        rgbWork.execute {
            try {
                val models=MobileFusionModels.acquire(application,directory)
                camera = models.camera
                val calibration=models.camera.calibration
                geometry = AtomRgbGeometry(480,640,if(demoRules) calibration.copy(
                    cameraFromTof=MobileFusionDemoRules.withoutPitch(calibration.cameraFromTof)) else calibration)
                extrinsicPitch=MobileFusionDemoRules.pitchDeg(geometry!!.calibration.cameraFromTof)
                tofEngine = models.tof.also { it.reset() }
                depth = models.depth
                if (closed.get()) return@execute
                emit(MobileFusionView(message="模型已就绪，正在连接三路传感器"))
                openSources()
            } catch (error: Exception) {
                close()
                // Close advances the epoch first. The final status can then pass the same
                // generation check as every other view, and the Activity can offer retry.
                publish(MobileFusionView(message="实验不可用：${error.cause?.message ?: error.message}",
                    inputEpoch=inputs.epoch,terminal=true))
            }
        }
    }

    fun calibrateHeight() { calibrateRequested.set(true) }
    fun accepts(value:MobileFusionView) = value.inputEpoch == inputs.epoch

    private fun openSources() {
        if (closed.get()) return
        Stream.values().forEach(::openStream)
        reconnects.scheduleWithFixedDelay({
            val now=SystemClock.elapsedRealtimeNanos()
            Stream.values().forEach { stream ->
                val last=synchronized(streamLastNs) { streamLastNs[stream] }
                if(last!=null && now-last>3_000_000_000L) scheduleReconnect(stream,"超过 3 秒无数据")
            }
        },3,1,TimeUnit.SECONDS)
    }

    private fun openStream(stream:Stream) {
        if(closed.get()) return
        val generation=synchronized(streamGeneration) { (streamGeneration[stream]!!+1).also { streamGeneration[stream]=it } }
        fun current()=!closed.get() && synchronized(streamGeneration) { streamGeneration[stream]==generation }
        fun alive() { synchronized(streamLastNs) { streamLastNs[stream]=SystemClock.elapsedRealtimeNanos() }
            if(synchronized(streamPending) { streamAttempts.remove(stream) }!=null) reconnectNote="" }
        fun status(value:CnhCaptureStatus) {
            if(current() && value.state in listOf(CnhConnectionState.ERROR,CnhConnectionState.DISCONNECTED))
                scheduleReconnect(stream,value.message)
        }
        // Mark the stream as just started so the watchdog gives a fresh connection 3 s to deliver.
        synchronized(streamLastNs) { streamLastNs[stream]=SystemClock.elapsedRealtimeNanos() }
        try { when(stream) {
            Stream.CAMERA -> AtomMjpegCaptureSource(atomHost).also { cameraSource=it }.start({ if(current()) {
                alive()
                cameraFirstReceiptNs.compareAndSet(0,it.receivedElapsedNs)
                cameraLastReceiptNs.set(it.receivedElapsedNs)
                cameraFrames.incrementAndGet()
                inputs.addCamera(it)
                showPreview(it)
            } }, ::status)
            Stream.IMU -> TcpImuCaptureSource(atomHost).also { imuSource=it }.start(onRecord={ r ->
                if(current()) { alive(); inputs.addImu(r); r.frame?.let { gait.add(r.hostReceivedMonotonicNs,it.accelG) } }
            },onStatus=::status)
            Stream.TOF -> TcpCnhCaptureSource(tofHost).also { tofSource=it }.start(onRecord={ r ->
                val f=r.frame ?: return@start
                if(!current()) return@start
                alive()
                val match=inputs.match(r.hostReceivedMonotonicNs)
                val pose=match.imu?.frame?.let { camera?.attitude(it) }
                if(pose==null) {
                    inputs.invalidate(expectedEpoch=match.epoch)?.let {
                        emit(MobileFusionView(message="等待匹配的 IMU 姿态"),it)
                    }
                    return@start
                }
                try { tofWork.execute { processTof(f,r.hostReceivedMonotonicNs,match,pose) } }
                catch (_:RejectedExecutionException) {
                    tofQueueRejected.incrementAndGet(); skipped.incrementAndGet()
                    inputs.invalidate(expectedEpoch=match.epoch)
                }
            },onStatus=::status)
        } } catch(error:Exception) { scheduleReconnect(stream,error.message ?: "连接失败") }
    }

    private fun scheduleReconnect(stream:Stream, reason:String) {
        if(closed.get()) return
        val attempt=synchronized(streamPending) {
            if(!streamPending.add(stream)) return
            ((streamAttempts[stream] ?: 0)+1).also { streamAttempts[stream]=it }
        }
        reconnectCount.incrementAndGet()
        reconnectNote="${stream.label} 连接中断，正在自动重连（第 $attempt 次）"
        inputs.invalidate(clearInputs=true)?.let {
            rgbPending.set(null)
            emit(MobileFusionView(message=reconnectNote,heightM=height,heightSource=heightSource),it)
        }
        synchronized(streamLastNs) { streamLastNs.remove(stream) }
        val delayMs=minOf(4_000L,1_000L shl minOf(attempt-1,2))
        try { reconnects.schedule({
            try {
                val old=when(stream) { Stream.CAMERA -> cameraSource; Stream.TOF -> tofSource; Stream.IMU -> imuSource }
                // Advance the generation first so callbacks from the closing source are ignored.
                synchronized(streamGeneration) { streamGeneration[stream]=streamGeneration[stream]!!+1 }
                runCatching { old?.close() }
                synchronized(streamPending) { streamPending.remove(stream) }
                openStream(stream)
            } catch(_:Exception) { synchronized(streamPending) { streamPending.remove(stream) } }
        },delayMs,TimeUnit.MILLISECONDS) } catch(_:RejectedExecutionException) { }
    }

    private fun showPreview(frame: AtomCameraCaptureFrame) {
        displayLatest.set(frame)
        if(!displayQueued.compareAndSet(false,true)) return
        try { displayWork.execute {
            try {
                while(!closed.get()) {
                    val wait=previewIntervalNs-(SystemClock.elapsedRealtimeNanos()-displayLastNs)
                    if(wait>0) Thread.sleep(wait/1_000_000,(wait%1_000_000).toInt())
                    val next=displayLatest.getAndSet(null) ?: break
                    displayLastNs=SystemClock.elapsedRealtimeNanos()
                    // Undistort only, no tone enhancement. Without a usable remap the raw frame is
                    // published and its orientation is fixed when drawing.
                    val raw=BitmapFactory.decodeByteArray(next.jpeg,0,next.jpeg.size) ?: continue
                    val processor=displayProcessor
                    if(!processor.supportsUndistort || raw.width!=640 || raw.height!=480) {
                        if(!closed.get()) publishFrame(raw.asImageBitmap(),next.receivedElapsedNs); continue
                    }
                    try {
                        if(displayPixels.size!=raw.width*raw.height) displayPixels=IntArray(raw.width*raw.height)
                        raw.getPixels(displayPixels,0,raw.width,0,0,raw.width,raw.height)
                        val out=processor.apply(displayPixels,raw.width,raw.height,true,false)
                        val bitmap=Bitmap.createBitmap(out.pixels,out.width,out.height,Bitmap.Config.ARGB_8888)
                        if(!closed.get()) publishFrame(bitmap.asImageBitmap(),next.receivedElapsedNs)
                    } finally { raw.recycle() }
                }
            } catch(error:Exception) { android.util.Log.w("FusionDisplay","preview failed",error) }
            finally { displayQueued.set(false); if(displayLatest.get()!=null && !closed.get()) showPreview(displayLatest.get()!!) }
        } } catch(_:RejectedExecutionException) { displayQueued.set(false) }
    }

    private fun processTof(f:CnhCaptureFrame,t:Long,match:MobileFusionInputBuffer.Match,pose:Pair<Double,Double>) {
        val e=match.epoch
        if(closed.get() || e!=inputs.epoch) return
        val tofStartedNs=SystemClock.elapsedRealtimeNanos()
        if(tofStartedNs-t>750_000_000L) {
            skipped.incrementAndGet(); inputs.invalidate(expectedEpoch=e); return
        }
        try {
            if(tofEpoch!=e) { tofEngine!!.reset(); floor=null; tofEpoch=e }
            if(calibrateRequested.getAndSet(false)) { floor=MobileFloorHeight(); heightSource="低头标定中，请看空地" }
            floor?.let { estimator ->
                val p=estimator.add(f.distanceMm.map(Int::toDouble).toDoubleArray(),f.targetStatus.toIntArray())
                heightSource="低头标定 ${p.frameCount}/${p.targetFrames} 帧"
                if(p.frameCount>=p.targetFrames) {
                    if(closed.get() || e!=inputs.epoch) return
                    p.calibration?.let { height=it.heightM; heightSource="本次低头标定 %.2f m".format(height) }
                        ?: run { heightSource="未找到稳定地面，仍用 %.2f m；可重试".format(height) }
                    floor=null
                    inputs.invalidate(expectedEpoch=e)?.let {
                        emit(MobileFusionView(message="高度标定结束，重新积累历史",heightM=height,heightSource=heightSource),it)
                    }
                    return
                }
            }
            val hist=FloatArray(1024) { i -> Math.scalb(f.histRaw[i/16][i%16].toDouble(),-f.histScaler[i/16][i%16]).toFloat() }
            val ambient=FloatArray(64) { i -> Math.scalb(f.ambientRaw[i].toDouble(),-f.ambientScaler[i]).toFloat() }
            val result=tofEngine!!.process(hist,ambient,f.sequence,f.deviceMs,pose.first,pose.second)
            if(!result.ready) {
                emit(MobileFusionView(message="积累 ToF 历史 ${result.historyLength}/8",heightM=height,heightSource=heightSource),e); return
            }
            if(closed.get() || e!=inputs.epoch) return
            if(rgbPending.getAndSet(RgbJob(f,t,match.image,match.imu!!.hostReceivedMonotonicNs,
                    pose.first,pose.second,result,e,(tofStartedNs-t)/1e6,
                    (SystemClock.elapsedRealtimeNanos()-tofStartedNs)/1e6))!=null) {
                rgbPendingReplaced.incrementAndGet(); skipped.incrementAndGet()
            }
            scheduleRgb()
        } catch(error:Exception) {
            inputs.invalidate(expectedEpoch=e)?.let {
                emit(MobileFusionView(message="推理暂停：${error.message}"),it)
            }
        }
    }

    private fun scheduleRgb() {
        if(closed.get() || !rgbBusy.compareAndSet(false,true)) return
        try { rgbWork.execute {
            try {
                while(!closed.get()) {
                    val job=rgbPending.getAndSet(null) ?: break
                    processRgb(job)
                }
            } finally { rgbBusy.set(false); if(rgbPending.get()!=null && !closed.get()) scheduleRgb() }
        } } catch(_:RejectedExecutionException) { rgbBusy.set(false) }
    }

    private fun processRgb(job:RgbJob) {
        if(closed.get() || job.epoch!=inputs.epoch) return
        try {
            val rgbStartedNs=SystemClock.elapsedRealtimeNanos()
            var jpegDecodeMs:Double?=null
            var undistortMs:Double?=null
            var depthMs:Double?=null
            var geometryMs:Double?=null
            val g=geometry!!
            if(rgbEpoch!=job.epoch || (rgbPreviousSequence>=0 && job.frame.sequence!=rgbPreviousSequence+1)) {
                g.reset(); votes.reset(); rgbEpoch=job.epoch
            }
            rgbPreviousSequence=job.frame.sequence
            val poseOk=AtomRgbGeometry.poseInEnvelope(job.pitch,job.roll)
            var rgb:AtomRgbGeometry.Result?=null
            var image:Bitmap?=null
            if(job.image!=null && poseOk && SystemClock.elapsedRealtimeNanos()-job.receiptNs<=750_000_000L) {
                var stageNs=SystemClock.elapsedRealtimeNanos()
                val raw=BitmapFactory.decodeByteArray(job.image.jpeg,0,job.image.jpeg.size)
                    ?: error("JPEG decode failed")
                jpegDecodeMs=(SystemClock.elapsedRealtimeNanos()-stageNs)/1e6
                stageNs=SystemClock.elapsedRealtimeNanos()
                val upright=try { camera!!.undistort(raw) } finally { raw.recycle() }
                undistortMs=(SystemClock.elapsedRealtimeNanos()-stageNs)/1e6
                image=upright.bitmap
                stageNs=SystemClock.elapsedRealtimeNanos()
                val disparity=depth!!.estimate(upright.bitmap)
                depthMs=(SystemClock.elapsedRealtimeNanos()-stageNs)/1e6
                stageNs=SystemClock.elapsedRealtimeNanos()
                rgb=g.process(disparity,upright.valid,job.frame.distanceMm.map(Int::toDouble).toDoubleArray(),
                    job.frame.targetStatus.toIntArray(),job.pitch,job.roll,height)
                geometryMs=(SystemClock.elapsedRealtimeNanos()-stageNs)/1e6
            }
            if(closed.get() || job.epoch!=inputs.epoch) return
            val rawDistance=job.frame.distanceMm.map(Int::toDouble).toDoubleArray()
            val rawStatus=job.frame.targetStatus.toIntArray()
            val walking=gait.walking(job.receiptNs)
            val limit=if(demoRules) MobileFusionDemoRules.limitM(walking) else 2.0
            val tofDepth=g.tofQueryDepth(rawDistance,rawStatus,job.pitch,job.roll)
            var tofHead=job.tof.headStrong && poseOk; var tofBody=job.tof.bodyStrong && poseOk
            var rgbHead=rgb?.head?.alarm==true; var rgbBody=rgb?.body?.alarm==true
            if(demoRules) {
                tofHead=MobileFusionDemoRules.keepTof(tofHead,tofDepth.first,limit)
                tofBody=MobileFusionDemoRules.keepTof(tofBody,tofDepth.second,limit)
                rgbHead=MobileFusionDemoRules.keepRgb(rgbHead,rgb?.head?.medianDepthM ?: Double.NaN,limit)
                rgbBody=MobileFusionDemoRules.keepRgb(rgbBody,rgb?.body?.medianDepthM ?: Double.NaN,limit)
            }
            fun nearest(tofOn:Boolean,tofM:Double,rgbOn:Boolean,rgbM:Double)=
                listOf(if(tofOn) tofM else Double.NaN,if(rgbOn) rgbM else Double.NaN).filter { it.isFinite() }.minOrNull() ?: Double.NaN
            val headM=nearest(tofHead,tofDepth.first,rgbHead,rgb?.head?.medianDepthM ?: Double.NaN)
            val bodyM=nearest(tofBody,tofDepth.second,rgbBody,rgb?.body?.medianDepthM ?: Double.NaN)
            val grid=FloatArray(64) { zone -> val raw=63-zone
                if(rawStatus[raw] in intArrayOf(5,6,9,10) && rawDistance[raw]>0) (rawDistance[raw]/1000).toFloat() else Float.NaN }
            val confirmed=votes.push(tofHead || rgbHead,tofBody || rgbBody)
            if(!poseOk) votes.reset()
            val elapsed=(SystemClock.elapsedRealtimeNanos()-job.receiptNs)/1_000_000
            if(elapsed>750) {
                votes.reset(); skipped.incrementAndGet()
                emit(MobileFusionView(message="处理超过 750 ms，当前结果已过期",latencyMs=elapsed,
                    heightM=height,heightSource=heightSource,skipped=skipped.get()),job.epoch); return
            }
            val count=processed.incrementAndGet()
            val result=MobileFusionView(message=when {
                !poseOk -> "姿态超出实验范围"
                job.image==null -> "没有 100 ms 内的配对画面，仅 ToF 强档"
                rgb?.fit==null -> "RGB 尺度暂不可用，仅 ToF 强档"
                else -> "实时实验 · 两帧确认"
            },ready=poseOk,head=confirmed.first && poseOk,body=confirmed.second && poseOk,
                tofHead=tofHead,tofBody=tofBody,rgbHead=rgbHead,rgbBody=rgbBody,
                demoRules=demoRules,walking=walking,limitM=limit,headDistanceM=headM,bodyDistanceM=bodyM,
                pitchDeg=job.pitch,rollDeg=job.roll,poseOk=poseOk,boxes=if(poseOk && image!=null) g.boxOutlines(job.pitch,job.roll) else emptyList(),
                tofGrid=grid,extrinsicPitchDeg=extrinsicPitch,
                anchor=rgb?.anchor?.name ?: "NONE",heightM=height,heightSource=heightSource,latencyMs=elapsed,
                receiptNs=job.receiptNs,sequence=job.frame.sequence,image=image,processed=count,skipped=skipped.get())
            val logStartedNs=SystemClock.elapsedRealtimeNanos()
            val receivedCameraFrames=cameraFrames.get()
            val cameraIntervalNs=cameraLastReceiptNs.get()-cameraFirstReceiptNs.get()
            receipt.appendText(JSONObject().put("sequence",job.frame.sequence).put("receipt_ns",job.receiptNs)
                .put("input_epoch",job.epoch).put("imu_receipt_ns",job.imuReceiptNs)
                .put("rgb_receipt_ns",job.image?.receivedElapsedNs ?: JSONObject.NULL)
                .put("imu_pair_age_ms",(job.receiptNs-job.imuReceiptNs)/1_000_000.0)
                .put("rgb_pair_age_ms",job.image?.let { (job.receiptNs-it.receivedElapsedNs)/1_000_000.0 } ?: JSONObject.NULL)
                .put("rgb_sequence",job.image?.sequence ?: JSONObject.NULL)
                .put("camera_received_frames",receivedCameraFrames)
                .put("camera_received_fps",if(cameraIntervalNs>0) (receivedCameraFrames-1)*1e9/cameraIntervalNs else JSONObject.NULL)
                .put("tof_queue_ms",job.tofQueueMs).put("tof_work_ms",job.tofWorkMs)
                .put("rgb_start_receipt_age_ms",(rgbStartedNs-job.receiptNs)/1e6)
                .put("jpeg_decode_ms",jpegDecodeMs ?: JSONObject.NULL)
                .put("undistort_ms",undistortMs ?: JSONObject.NULL)
                .put("depth_ms",depthMs ?: JSONObject.NULL).put("geometry_ms",geometryMs ?: JSONObject.NULL)
                .put("previous_log_ms",previousLogMs)
                .put("tof_queue_rejected",tofQueueRejected.get()).put("rgb_pending_replaced",rgbPendingReplaced.get())
                .put("processing_receipt_age_ms",elapsed).put("head",result.head).put("body",result.body)
                .put("tof_head",tofHead).put("tof_body",tofBody).put("rgb_head",result.rgbHead).put("rgb_body",result.rgbBody)
                .put("anchor",result.anchor).put("pose_ok",poseOk).put("pitch_deg",job.pitch).put("roll_deg",job.roll)
                .put("demo_rules",demoRules).put("walking",walking).put("limit_m",limit)
                .put("tof_head_depth_m",tofDepth.first.takeIf { it.isFinite() } ?: JSONObject.NULL)
                .put("tof_body_depth_m",tofDepth.second.takeIf { it.isFinite() } ?: JSONObject.NULL)
                .put("rgb_head_depth_m",rgb?.head?.medianDepthM?.takeIf { it.isFinite() } ?: JSONObject.NULL)
                .put("rgb_body_depth_m",rgb?.body?.medianDepthM?.takeIf { it.isFinite() } ?: JSONObject.NULL).put("height_m",height).put("skipped",skipped.get()).toString()+"\n")
            previousLogMs=(SystemClock.elapsedRealtimeNanos()-logStartedNs)/1e6
            emit(result,job.epoch)
        } catch(error:Exception) {
            votes.reset()
            emit(MobileFusionView(message="RGB 推理暂停：${error.cause?.message ?: error.message}"),job.epoch)
        }
    }

    private fun emit(value:MobileFusionView, inputEpoch:Long=inputs.epoch) {
        if(!closed.get() && inputEpoch==inputs.epoch) publish(value.copy(inputEpoch=inputEpoch))
    }

    override fun close() {
        if(!closed.compareAndSet(false,true)) return
        inputs.retire(); rgbPending.set(null)
        // Never block the Android lifecycle thread on socket or accelerator shutdown.
        Thread({
            try {
            // A reconnect in flight must finish before the final source close, or it could leak one.
            reconnects.shutdownNow(); reconnects.awaitTermination(5,TimeUnit.SECONDS)
            listOf(tofSource,imuSource,cameraSource).forEach { runCatching { it?.close() } }
            displayWork.shutdown(); displayWork.awaitTermination(5,TimeUnit.SECONDS)
            tofWork.shutdown(); tofWork.awaitTermination(10,TimeUnit.SECONDS)
            rgbWork.shutdown(); rgbWork.awaitTermination(30,TimeUnit.SECONDS)
            // Cached engines stay loaded (MobileFusionModels); wait for any native call to finish
            // before the lease lets the next session use them.
            while(!tofWork.isTerminated || !rgbWork.isTerminated) Thread.sleep(100)
            } finally { MobileFusionLease.release(lease) }
        },"fusion-cleanup").start()
    }
}
