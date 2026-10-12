package com.linnan.blindassist

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import com.linnan.blindassist.capture.*
import com.linnan.blindassist.vision.*
import org.json.JSONObject
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
)

/** Foreground, opt-in engineering path. It has no speaker/haptic side effects and never records
 * evaluation labels. There is one H3 consumer and bounded latest-only RGB work. */
internal class MobileFusionSession(
    private val application: Application, private val directory: File,
    private val atomHost: String, private val tofHost: String,
    private val publish: (MobileFusionView) -> Unit,
) : AutoCloseable {
    private val lease = MobileFusionLease.acquire()
    private val closed = AtomicBoolean()
    private val epoch = AtomicLong()
    private val tofWork = ThreadPoolExecutor(1,1,0,TimeUnit.MILLISECONDS,ArrayBlockingQueue(2),
        { r -> Thread(r,"fusion-tof") },ThreadPoolExecutor.AbortPolicy())
    private val rgbWork = Executors.newSingleThreadExecutor { Thread(it,"fusion-rgb") }
    private val rgbPending = AtomicReference<RgbJob?>()
    private val rgbBusy = AtomicBoolean()
    private val latestCamera = AtomicReference<AtomCameraCaptureFrame?>()
    private val imuLock = Any()
    private val imuHistory = ArrayDeque<ImuCaptureRecord>()
    private var cameraSource: AtomMjpegCaptureSource? = null
    private var tofSource: TcpCnhCaptureSource? = null
    private var imuSource: TcpImuCaptureSource? = null
    private var camera: MobileFusionCamera? = null
    private var tofEngine: CnhStrongEngine? = null
    private var depth: Dav2DepthRuntime? = null
    private var geometry: AtomRgbGeometry? = null
    private val votes = AtomRgbTwoOfThree()
    private var rgbEpoch = -1L
    private var tofEpoch = -1L
    private var rgbPreviousSequence = -1L
    @Volatile private var height = 1.35
    @Volatile private var heightSource = "默认 1.35 m，建议先低头标定"
    private val calibrateRequested = AtomicBoolean()
    private var floor: MobileFloorHeight? = null
    private val skipped = AtomicLong()
    private val processed = AtomicLong()
    private val receipt = File(directory,"live-${System.currentTimeMillis()}.jsonl")
    private data class RgbJob(val frame: CnhCaptureFrame, val receiptNs: Long, val image: AtomCameraCaptureFrame?,
        val pitch: Double, val roll: Double, val tof: CnhStrongResult, val epoch: Long)

    fun start() {
        rgbWork.execute {
            try {
                camera = MobileFusionCamera(directory)
                geometry = AtomRgbGeometry(480,640,camera!!.calibration)
                tofEngine = CnhStrongEngine(File(directory,"cnh_strong_m3.onnx"),File(directory,"cnh_strong_calibration.bin"))
                depth = Dav2DepthRuntimeFactory.create(application,File(directory,"dav2s_686x518_fp32.dlc"))
                if (closed.get()) return@execute
                emit(MobileFusionView(message="模型已就绪，正在连接三路传感器"))
                openSources()
            } catch (error: Exception) {
                emit(MobileFusionView(message="实验不可用：${error.cause?.message ?: error.message}"))
                close()
            }
        }
    }

    fun calibrateHeight() { calibrateRequested.set(true) }
    fun accepts(value:MobileFusionView) = value.inputEpoch == epoch.get()

    private fun openSources() {
        val cs = AtomMjpegCaptureSource(atomHost)
        val ts = TcpCnhCaptureSource(tofHost)
        val ims = TcpImuCaptureSource(atomHost)
        cameraSource=cs; tofSource=ts; imuSource=ims
        if (closed.get()) { cs.close(); ts.close(); ims.close(); return }
        fun unavailable(label: String, status: CnhCaptureStatus) {
            if (status.state in listOf(CnhConnectionState.ERROR,CnhConnectionState.DISCONNECTED)) {
                epoch.incrementAndGet(); rgbPending.set(null)
                latestCamera.set(null)
                synchronized(imuLock) { imuHistory.clear() }
                emit(MobileFusionView(message="$label 暂无数据，请停止后重新连接"))
            }
        }
        cs.start({ if(!closed.get()) latestCamera.set(it) }, { unavailable("相机",it) })
        ims.start(onRecord={ r ->
            val frame=r.frame
            if(frame!=null && !closed.get()) synchronized(imuLock) {
                if(imuHistory.lastOrNull()?.frame?.bootId != frame.bootId && imuHistory.isNotEmpty()) {
                    imuHistory.clear(); epoch.incrementAndGet()
                }
                imuHistory.addLast(r)
                while(imuHistory.size>64) imuHistory.removeFirst()
            }
        },onStatus={ unavailable("IMU",it) })
        ts.start(onRecord={ r ->
            val f=r.frame ?: return@start
            if(closed.get()) return@start
            val im=synchronized(imuLock) { imuHistory.lastOrNull { it.hostReceivedMonotonicNs <= r.hostReceivedMonotonicNs } }
            val pose=im?.takeIf { r.hostReceivedMonotonicNs-it.hostReceivedMonotonicNs in 0..100_000_000L }
                ?.frame?.let { camera?.attitude(it) }
            if(pose==null) {
                epoch.incrementAndGet(); emit(MobileFusionView(message="等待匹配的 IMU 姿态")); return@start
            }
            val image=latestCamera.get()?.takeIf { r.hostReceivedMonotonicNs-it.receivedElapsedNs in 0..100_000_000L }
            val e=epoch.get()
            try { tofWork.execute { processTof(f,r.hostReceivedMonotonicNs,image,pose,e) } }
            catch (_:RejectedExecutionException) { skipped.incrementAndGet(); epoch.incrementAndGet() }
        },onStatus={ unavailable("ToF",it) })
    }

    private fun processTof(f:CnhCaptureFrame,t:Long,image:AtomCameraCaptureFrame?,pose:Pair<Double,Double>,e:Long) {
        if(closed.get() || e!=epoch.get()) return
        if(SystemClock.elapsedRealtimeNanos()-t>750_000_000L) { skipped.incrementAndGet(); epoch.incrementAndGet(); return }
        try {
            if(tofEpoch!=e) { tofEngine!!.reset(); floor=null; tofEpoch=e }
            if(calibrateRequested.getAndSet(false)) { floor=MobileFloorHeight(); heightSource="低头标定中，请看空地" }
            floor?.let { estimator ->
                val p=estimator.add(f.distanceMm.map(Int::toDouble).toDoubleArray(),f.targetStatus.toIntArray())
                heightSource="低头标定 ${p.frameCount}/${p.targetFrames} 帧"
                if(p.frameCount>=p.targetFrames) {
                    p.calibration?.let { height=it.heightM; heightSource="本次低头标定 %.2f m".format(height) }
                        ?: run { heightSource="未找到稳定地面，仍用 %.2f m；可重试".format(height) }
                    floor=null; epoch.incrementAndGet(); return
                }
            }
            val hist=FloatArray(1024) { i -> Math.scalb(f.histRaw[i/16][i%16].toDouble(),-f.histScaler[i/16][i%16]).toFloat() }
            val ambient=FloatArray(64) { i -> Math.scalb(f.ambientRaw[i].toDouble(),-f.ambientScaler[i]).toFloat() }
            val result=tofEngine!!.process(hist,ambient,f.sequence,f.deviceMs,pose.first,pose.second)
            if(!result.ready) {
                emit(MobileFusionView(message="积累 ToF 历史 ${result.historyLength}/8",heightM=height,heightSource=heightSource)); return
            }
            if(closed.get() || e!=epoch.get()) return
            if(rgbPending.getAndSet(RgbJob(f,t,image,pose.first,pose.second,result,e))!=null) skipped.incrementAndGet()
            scheduleRgb()
        } catch(error:Exception) {
            epoch.incrementAndGet(); emit(MobileFusionView(message="推理暂停：${error.message}"))
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
        if(closed.get() || job.epoch!=epoch.get()) return
        try {
            val g=geometry!!
            if(rgbEpoch!=job.epoch || (rgbPreviousSequence>=0 && job.frame.sequence!=rgbPreviousSequence+1)) {
                g.reset(); votes.reset(); rgbEpoch=job.epoch
            }
            rgbPreviousSequence=job.frame.sequence
            val poseOk=AtomRgbGeometry.poseInEnvelope(job.pitch,job.roll)
            var rgb:AtomRgbGeometry.Result?=null
            var image:Bitmap?=null
            if(job.image!=null && poseOk && SystemClock.elapsedRealtimeNanos()-job.receiptNs<=750_000_000L) {
                val raw=BitmapFactory.decodeByteArray(job.image.jpeg,0,job.image.jpeg.size)
                    ?: error("JPEG decode failed")
                val upright=try { camera!!.undistort(raw) } finally { raw.recycle() }
                image=upright.bitmap
                val disparity=depth!!.estimate(upright.bitmap)
                rgb=g.process(disparity,upright.valid,job.frame.distanceMm.map(Int::toDouble).toDoubleArray(),
                    job.frame.targetStatus.toIntArray(),job.pitch,job.roll,height)
            }
            if(closed.get() || job.epoch!=epoch.get()) return
            val tofHead=job.tof.headStrong && poseOk; val tofBody=job.tof.bodyStrong && poseOk
            val confirmed=votes.push(tofHead || rgb?.head?.alarm==true,tofBody || rgb?.body?.alarm==true)
            if(!poseOk) votes.reset()
            val elapsed=(SystemClock.elapsedRealtimeNanos()-job.receiptNs)/1_000_000
            if(elapsed>750) {
                votes.reset(); skipped.incrementAndGet()
                emit(MobileFusionView(message="处理超过 750 ms，当前结果已过期",latencyMs=elapsed,
                    heightM=height,heightSource=heightSource,skipped=skipped.get())); return
            }
            val count=processed.incrementAndGet()
            val result=MobileFusionView(message=when {
                !poseOk -> "姿态超出实验范围"
                job.image==null -> "没有 100 ms 内的配对画面，仅 ToF 强档"
                rgb?.fit==null -> "RGB 尺度暂不可用，仅 ToF 强档"
                else -> "实时实验 · 两帧确认"
            },ready=poseOk,head=confirmed.first && poseOk,body=confirmed.second && poseOk,
                tofHead=tofHead,tofBody=tofBody,rgbHead=rgb?.head?.alarm==true,rgbBody=rgb?.body?.alarm==true,
                anchor=rgb?.anchor?.name ?: "NONE",heightM=height,heightSource=heightSource,latencyMs=elapsed,
                receiptNs=job.receiptNs,sequence=job.frame.sequence,image=image,processed=count,skipped=skipped.get())
            receipt.appendText(JSONObject().put("sequence",job.frame.sequence).put("receipt_ns",job.receiptNs)
                .put("processing_receipt_age_ms",elapsed).put("head",result.head).put("body",result.body)
                .put("tof_head",tofHead).put("tof_body",tofBody).put("rgb_head",result.rgbHead).put("rgb_body",result.rgbBody)
                .put("anchor",result.anchor).put("pose_ok",poseOk).put("height_m",height).put("skipped",skipped.get()).toString()+"\n")
            emit(result,job.epoch)
        } catch(error:Exception) { votes.reset(); emit(MobileFusionView(message="RGB 推理暂停：${error.cause?.message ?: error.message}")) }
    }

    private fun emit(value:MobileFusionView, inputEpoch:Long=epoch.get()) {
        if(!closed.get()) publish(value.copy(inputEpoch=inputEpoch))
    }

    override fun close() {
        if(!closed.compareAndSet(false,true)) return
        epoch.incrementAndGet(); rgbPending.set(null); latestCamera.set(null)
        // Never block the Android lifecycle thread on socket or accelerator shutdown.
        Thread({
            try {
            listOf(tofSource,imuSource,cameraSource).forEach { runCatching { it?.close() } }
            tofWork.shutdown(); tofWork.awaitTermination(10,TimeUnit.SECONDS)
            rgbWork.shutdown(); rgbWork.awaitTermination(30,TimeUnit.SECONDS)
            // If a native call is still running, its serialized cleanup follows it.
            if(tofWork.isTerminated && rgbWork.isTerminated) {
                runCatching { tofEngine?.close() }; runCatching { depth?.close() }
            } else {
                while(!tofWork.isTerminated || !rgbWork.isTerminated) Thread.sleep(100)
                runCatching { tofEngine?.close() }; runCatching { depth?.close() }
            }
            } finally { MobileFusionLease.release(lease) }
        },"fusion-cleanup").start()
    }
}
