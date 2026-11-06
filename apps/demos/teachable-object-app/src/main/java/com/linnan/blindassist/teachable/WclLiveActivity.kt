package com.linnan.blindassist.teachable

import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CaptureRequest
import android.os.BatteryManager
import android.os.Bundle
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Gravity
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.ceil

/**
 * Live WCL chain: CameraX 30 fps, frames scored at [EXTRA_RATE] per second -> centre square -> 224 -> So400m on the HTP ->
 * cosine to the teaching prototypes -> causal confidence-weighted EMA with a 0.5 s half-life ([ConfidenceEma]) -> speak after ceil(rate) consecutive analysed frames agree (3 by default).
 * [EXTRA_CONF_POWER] 0 with [EXTRA_HALF_LIFE] 1 restores the old plain EMA, whose top-1 is always logged next to it.
 * [EXTRA_LIBRARY] uses saved personal teaching. Without it, the existing explicit ORBIT replay path is retained.
 * [EXTRA_SPATIAL] replaces the cosine with the prototype-guided spatial readout ([PfPipeline]) and logs the paired plain WCL top-1.
 * [stopAndDump] writes filesDir/wcl/live_<user>_<rate>[_pf].json with per-frame timings and thermal samples.
 */
class WclLiveActivity : ComponentActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private lateinit var label: TextView
    private lateinit var user: String
    private var library: String? = null
    private var rate = 3.0
    private var confPower = ConfidenceEma.DEFAULT_POWER
    private lateinit var smoother: ConfidenceEma
    private lateinit var plainSmoother: ConfidenceEma
    private var stableFrames = 3
    @Volatile private var pipeline: WclPipeline? = null
    @Volatile private var pf: PfPipeline? = null
    private var spatial = false
    private var protos = FloatArray(0)
    private var objects: List<String> = emptyList()
    private var tts: TextToSpeech? = null
    @Volatile private var ttsReady = false
    @Volatile private var speechWarning: String? = null
    private var provider: ProcessCameraProvider? = null
    private var analysisUseCase: ImageAnalysis? = null
    private var previewUseCase: Preview? = null
    @Volatile private var stopping = false
    private val t0 = SystemClock.elapsedRealtimeNanos()
    private var nextNs = 0L
    private var seen = 0L

    private val frameTs = ArrayList<Long>(); private val doneTs = ArrayList<Long>()
    private val prepMs = ArrayList<Double>(); private val embedMs = ArrayList<Double>()
    private val smoothTop = ArrayList<Int>()
    private val readMs = ArrayList<Double>(); private val wclTop = ArrayList<Int>()
    private val plainEmaTop = ArrayList<Int>(); private val frameWeight = ArrayList<Double>()
    private val thermal = JSONArray(); private val announcements = JSONArray()
    private var runTop = -1; private var runLen = 0; private var lastSpoken = -1
    private var loadMs = 0.0; private var teachMs = 0.0; private var teachFrames = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT >= 27) { setShowWhenLocked(true); setTurnScreenOn(true) }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        library = intent.getStringExtra(EXTRA_LIBRARY)
        user = library?.let { "library_$it" } ?: (intent.getStringExtra(EXTRA_USER) ?: "P189")
        rate = intent.getDoubleExtra(EXTRA_RATE, 3.0)
        spatial = intent.getBooleanExtra(EXTRA_SPATIAL, false)
        confPower = intent.getIntExtra(EXTRA_CONF_POWER, ConfidenceEma.DEFAULT_POWER)
        val halfLife = intent.getDoubleExtra(EXTRA_HALF_LIFE, ConfidenceEma.DEFAULT_HALF_LIFE_S)
        smoother = ConfidenceEma(ConfidenceEma.alphaOf(rate, halfLife), confPower); plainSmoother = ConfidenceEma(ConfidenceEma.alphaOf(rate, 1.0), 0)
        stableFrames = ceil(rate).toInt()
        val preview = PreviewView(this)
        label = TextView(this).apply { text = "正在准备识别，请稍候。"; textSize = 22f; setTextColor(0xFFFFFFFF.toInt()); setBackgroundColor(0x88000000.toInt()); setPadding(24, 16, 24, 16) }
        setContentView(FrameLayout(this).apply {
            addView(preview)
            addView(label, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM))
            addView(Button(this@WclLiveActivity).apply {
                text = "结束识别，返回教学"
                setOnClickListener { finish() }
            }, FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP or Gravity.CENTER_HORIZONTAL))
        })
        tts = TextToSpeech(this) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (!ttsReady) speechWarning = "语音服务不可用，当前仅显示识别结果。"
        }
        worker.execute {
            try { setup(); main.post { if (!isDestroyed) label.text = "准备好了，请把已教物品放到镜头前。" } }
            catch (error: Exception) { Log.e(TAG, "setup failed", error); main.post { if (!isDestroyed) label.text = error.message ?: "无法开始识别，请返回教学页。" } }
        }
        sampleThermal()
        startCamera(preview)
    }

    private fun setup() {
        var t = System.nanoTime()
        val personal = library?.let {
            check(!spatial) { "个人教学当前使用整帧识别，请关闭空间读取。" }
            WclTeachingStore(WclTeachingStore.directory(filesDir, it)).load(WclTeachingStore.modelIdentity(File(filesDir, "wcl/wcl_cached.dlc")))
        }
        val w = if (spatial) null else WclPipeline(File(filesDir, "wcl/wcl_cached.dlc"), applicationInfo.nativeLibraryDir)
        val s = if (spatial) PfPipeline(File(filesDir, "wcl/pf_body_cached.dlc"), File(filesDir, "wcl/pf_head_m20_cached.dlc"), applicationInfo.nativeLibraryDir) else null
        loadMs = (System.nanoTime() - t) / 1e6
        t = System.nanoTime()
        try {
        if (personal != null) {
            objects = personal.names; protos = personal.prototypes; teachFrames = personal.counts.sum()
        } else {
        val dir = File(filesDir, "m0/$user")
        val meta = JSONObject(File(dir, "meta.json").readText())
        objects = List(meta.getJSONArray("objects").length()) { meta.getJSONArray("objects").getString(it) }
        val teach = meta.getJSONArray("teach")
        val sums = List(objects.size) { FloatArray(WclPipeline.D) }
        for (o in objects.indices) {
            val frames = teach.getJSONArray(o)
            for (i in 0 until frames.length()) {
                val x = WclPipeline.imageOf(File(dir, "img/" + frames.getString(i)))
                val e = w?.embed(x) ?: s!!.embed(x)
                for (d in 0 until WclPipeline.D) sums[o][d] += e[d]
                teachFrames++
            }
        }
        protos = WclPipeline.prototypesOf(sums)
        }
        s?.setPrototypes(protos)
        teachMs = (System.nanoTime() - t) / 1e6
        pipeline = w; pf = s
        Log.i(TAG, "ready: $user ${objects.size} objects, load ${"%.0f".format(loadMs)} ms, teach $teachFrames frames ${"%.0f".format(teachMs)} ms")
        } catch (error: Throwable) { w?.close(); s?.close(); throw error }
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun startCamera(view: PreviewView) {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (isDestroyed) return@addListener
            val cp = future.get(); provider = cp
            val fps = Range(30, 30)
            val previewBuilder = Preview.Builder()
            Camera2Interop.Extender(previewBuilder).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
            val preview = previewBuilder.build().also { it.surfaceProvider = view.surfaceProvider }
            previewUseCase = preview
            val analysisBuilder = ImageAnalysis.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            Camera2Interop.Extender(analysisBuilder).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
            val analysis = analysisBuilder.build().also { it.setAnalyzer(worker, ::analyze) }
            analysisUseCase = analysis
            cp.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        val now = SystemClock.elapsedRealtimeNanos()
        val w = pipeline; val spatialChain = pf
        seen++
        if (stopping || (w == null && spatialChain == null) || now < nextNs) { image.close(); return }
        nextNs = maxOf(nextNs + (1e9 / rate).toLong(), now)
        val ts = image.imageInfo.timestamp
        var t = System.nanoTime()
        val bmp = image.toBitmap(); val rot = image.imageInfo.rotationDegrees
        image.close()
        val side = minOf(bmp.width, bmp.height)
        val s = WclPipeline.SIZE.toFloat() / side
        val crop = Bitmap.createBitmap(bmp, (bmp.width - side) / 2, (bmp.height - side) / 2, side, side,
            Matrix().apply { postScale(s, s); postRotate(rot.toFloat()) }, true)
        val x = WclPipeline.imageOf(crop); crop.recycle(); bmp.recycle()
        val prep = (System.nanoTime() - t) / 1e6; t = System.nanoTime()
        var read = 0.0; var plainTop = -1
        val score: FloatArray
        if (spatialChain != null) {
            val tok = spatialChain.tokens(x)
            val tr = System.nanoTime()
            val r = spatialChain.read(tok)
            read = (System.nanoTime() - tr) / 1e6
            score = r.scores
            val plain = WclPipeline.cosine(r.embedding, protos); plainTop = plain.indices.maxBy { plain[it] }
        } else {
            score = WclPipeline.cosine(w!!.embed(x), protos)
        }
        val em = (System.nanoTime() - t) / 1e6
        val top = smoother.update(score); val plainEma = plainSmoother.update(score); val weight = ConfidenceEma.weight(score, confPower)
        if (top == runTop) runLen++ else { runTop = top; runLen = 1 }
        val done = SystemClock.elapsedRealtimeNanos()
        synchronized(this) { frameTs += ts; doneTs += done; prepMs += prep; embedMs += em; smoothTop += top; readMs += read; wclTop += plainTop; plainEmaTop += plainEma; frameWeight += weight }
        if (runLen == stableFrames && top != lastSpoken && ttsReady) {
            val name = objects[top].replace('-', ' ')
            val chinese = name.any { Character.UnicodeScript.of(it.code) == Character.UnicodeScript.HAN }
            val available = tts?.setLanguage(if (chinese) Locale.SIMPLIFIED_CHINESE else Locale.US) ?: TextToSpeech.LANG_NOT_SUPPORTED
            if (available >= TextToSpeech.LANG_AVAILABLE) {
                speechWarning = null; lastSpoken = top
                announcements.put(JSONObject().put("frame", frameTs.size - 1).put("object", objects[top]))
                if (tts?.speak(name, TextToSpeech.QUEUE_FLUSH, null, "a${announcements.length()}") == TextToSpeech.ERROR)
                    speechWarning = "语音播放失败，当前仅显示识别结果。"
            } else {
                speechWarning = if (chinese) "此设备尚无可用的中文语音数据，当前仅显示识别结果。" else "此设备尚无可用的英语语音数据，当前仅显示识别结果。"
            }
        }
        main.post {
            if (!isDestroyed) {
                val result = if (library != null) objects[top] else "${objects[top]}  ${"%.0f".format(em)} ms"
                label.text = result + (speechWarning?.let { "\n$it" } ?: "")
            }
        }
    }

    private fun rssKb(): Long = File("/proc/self/status").readLines().firstOrNull { it.startsWith("VmRSS") }?.filter { it.isDigit() }?.toLongOrNull() ?: -1

    internal fun analyzedFrameCount(): Int = synchronized(this) { frameTs.size }

    private fun sampleThermal() {
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val pm = getSystemService(PowerManager::class.java)
        thermal.put(JSONObject().put("t_s", (SystemClock.elapsedRealtimeNanos() - t0) / 1e9)
            .put("battery_c", (battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1) / 10.0)
            .put("level", battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1)
            .put("thermal_status", if (Build.VERSION.SDK_INT >= 29) pm.currentThermalStatus else -1)
            .put("headroom_10s", if (Build.VERSION.SDK_INT >= 30) pm.getThermalHeadroom(10).toDouble().let { if (it.isNaN()) -1.0 else it } else -1.0)
            .put("frames", frameTs.size).put("rss_kb", rssKb()))
        main.postDelayed(::sampleThermal, 15000)
    }

    fun stopAndDump(): File {
        main.removeCallbacksAndMessages(null)
        analysisUseCase?.let { provider?.unbind(it) }; previewUseCase?.let { provider?.unbind(it) }
        sampleThermal(); main.removeCallbacksAndMessages(null)
        val out = File(filesDir, "wcl").apply { mkdirs() }.let { File(it, "live_${user}_${rate}${if (spatial) "_pf" else ""}.json") }
        synchronized(this) {
            out.writeText(JSONObject().put("user", user).put("personal_library", library ?: JSONObject.NULL).put("rate", rate).put("objects", JSONArray(objects)).put("load_ms", loadMs).put("teach_ms", teachMs).put("teach_frames", teachFrames)
                .put("camera_frames_seen", seen).put("frame_ts_ns", JSONArray(frameTs)).put("done_ns", JSONArray(doneTs)).put("prep_ms", JSONArray(prepMs)).put("embed_ms", JSONArray(embedMs)).put("spatial", spatial).put("read_ms", JSONArray(readMs)).put("wcl_top", JSONArray(wclTop))
                .put("smooth_top", JSONArray(smoothTop)).put("conf_power", confPower).put("plain_ema_top", JSONArray(plainEmaTop)).put("frame_weight", JSONArray(frameWeight)).put("thermal", thermal).put("announcements", announcements).toString())
        }
        return out
    }

    override fun onDestroy() {
        main.removeCallbacksAndMessages(null)
        stopping = true
        analysisUseCase?.clearAnalyzer()
        analysisUseCase?.let { provider?.unbind(it) }; previewUseCase?.let { provider?.unbind(it) }
        tts?.shutdown()
        // Close after an in-flight load/inference finishes, on the same serial executor.
        worker.execute { pipeline?.close(); pf?.close(); pipeline = null; pf = null }
        worker.shutdown()
        super.onDestroy()
    }

    companion object {
        const val EXTRA_USER = "user"
        const val EXTRA_LIBRARY = "personalLibrary"
        const val EXTRA_RATE = "rate"
        const val EXTRA_SPATIAL = "spatial"
        const val EXTRA_CONF_POWER = "confPower"
        const val EXTRA_HALF_LIFE = "halfLife"
        const val TAG = "TeachableWcl"
    }
}
