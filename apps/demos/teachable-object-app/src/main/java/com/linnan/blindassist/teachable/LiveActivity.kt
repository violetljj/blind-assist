package com.linnan.blindassist.teachable

import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.os.BatteryManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Gravity
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
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
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.pow

/**
 * M1 live chain: CameraX 30 fps -> centre square -> 224 -> identify (main readout) -> causal EMA (half-life 30 frames,
 * bias-corrected, as the thesis EMA30 readout) -> speak the object name once the smoothed top-1 has been the same for
 * STABLE_FRAMES consecutive analysed frames and differs from the last announcement. Memory: the PC-built memory of one
 * ORBIT validation user (filesDir/m0/<user>/pc_parts.raw); CLS prototypes from that user's teaching frames on the phone.
 * Logs per frame and thermal samples; [stopAndDump] writes filesDir/m1/live_<user>.json.
 */
class LiveActivity : ComponentActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private lateinit var label: TextView
    private lateinit var user: String
    @Volatile private var pipeline: TeachablePipeline? = null
    private var memory: TeachablePipeline.Memory? = null
    private var objects: List<String> = emptyList()
    private var tts: TextToSpeech? = null
    @Volatile private var ttsReady = false
    private var provider: ProcessCameraProvider? = null
    private val t0 = SystemClock.elapsedRealtimeNanos()

    // per analysed frame
    private val frameTs = ArrayList<Long>(); private val entryTs = ArrayList<Long>(); private val doneTs = ArrayList<Long>()
    private val prepMs = ArrayList<Double>(); private val backboneMs = ArrayList<Double>(); private val identifyMs = ArrayList<Double>()
    private val looked = ArrayList<Boolean>(); private val pred = ArrayList<Int>(); private val smoothTop = ArrayList<Int>()
    private val thermal = JSONArray(); private val announcements = JSONArray()
    private val pendingSpeech = HashMap<String, JSONObject>()
    private var ema: DoubleArray? = null; private var emaW = 0.0
    private var runTop = -1; private var runLen = 0; private var lastSpoken = -1
    private var timestampSource = "UNKNOWN"; private var setupMs = 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setShowWhenLocked(true); setTurnScreenOn(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        user = intent.getStringExtra(EXTRA_USER) ?: "P189"
        val preview = PreviewView(this)
        label = TextView(this).apply { textSize = 22f; setTextColor(0xFFFFFFFF.toInt()); setBackgroundColor(0x88000000.toInt()); setPadding(24, 16, 24, 16) }
        setContentView(FrameLayout(this).apply {
            addView(preview)
            addView(label, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM))
        })
        tts = TextToSpeech(this) { status ->
            ttsReady = status == TextToSpeech.SUCCESS && tts?.setLanguage(Locale.US)?.let { it >= 0 } == true
        }.apply {
            setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(id: String) { synchronized(pendingSpeech) { pendingSpeech[id]?.put("tts_start_ns", SystemClock.elapsedRealtimeNanos()) } }
                override fun onDone(id: String) { synchronized(pendingSpeech) { pendingSpeech[id]?.put("tts_done_ns", SystemClock.elapsedRealtimeNanos()) } }
                @Deprecated("Deprecated in Java") override fun onError(id: String) {}
            })
        }
        worker.execute { setup() }
        sampleThermal()
        startCamera(preview)
    }

    private fun setup() {
        val t = System.nanoTime()
        val root = File(filesDir, "m0"); val dir = File(root, user)
        val meta = JSONObject(File(dir, "meta.json").readText())
        objects = List(meta.getJSONArray("objects").length()) { meta.getJSONArray("objects").getString(it) }
        val p = TeachablePipeline(File(root, "backbone_fp32.dlc"), File(root, "head_dyn.dlc"), applicationInfo.nativeLibraryDir)
        val teach = meta.getJSONArray("teach"); val cls = List(objects.size) { FloatArray(TeachablePipeline.D) }
        for (o in objects.indices) {
            val frames = teach.getJSONArray(o)
            for (i in 0 until frames.length()) {
                val tok = p.tokens(p.image(File(dir, "img/" + frames.getString(i))))
                for (d in 0 until TeachablePipeline.D) cls[o][d] += tok[d]
            }
        }
        val raw = File(dir, "pc_parts.raw").readBytes()
        val parts = FloatArray(raw.size / 4).also { ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
        memory = p.memory(objects.size, parts, cls)
        setupMs = (System.nanoTime() - t) / 1e6
        pipeline = p
        Log.i(TAG, "ready: $user ${objects.size} objects, setup ${"%.0f".format(setupMs)} ms")
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun startCamera(view: PreviewView) {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val cp = future.get(); provider = cp
            val fps = Range(30, 30)
            val previewBuilder = Preview.Builder()
            Camera2Interop.Extender(previewBuilder).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
            val preview = previewBuilder.build().also { it.surfaceProvider = view.surfaceProvider }
            val analysisBuilder = ImageAnalysis.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            Camera2Interop.Extender(analysisBuilder).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
            val analysis = analysisBuilder.build().also { it.setAnalyzer(worker, ::analyze) }
            cp.unbindAll()
            val camera = cp.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            timestampSource = when (Camera2CameraInfo.from(camera.cameraInfo).getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)) {
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME -> "REALTIME"
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN -> "UNKNOWN_MONOTONIC"
                else -> "OTHER"
            }
        }, mainExecutor)
    }

    private fun analyze(image: ImageProxy) {
        val entry = SystemClock.elapsedRealtimeNanos()
        val p = pipeline; val m = memory
        if (p == null || m == null) { image.close(); return }
        val ts = image.imageInfo.timestamp
        var t = System.nanoTime()
        val bmp = image.toBitmap(); val rot = image.imageInfo.rotationDegrees
        image.close()
        val side = minOf(bmp.width, bmp.height)
        val s = TeachablePipeline.SIZE.toFloat() / side
        val crop = Bitmap.createBitmap(bmp, (bmp.width - side) / 2, (bmp.height - side) / 2, side, side,
            Matrix().apply { postScale(s, s); postRotate(rot.toFloat()) }, true)
        val x = p.image(crop); crop.recycle(); bmp.recycle()
        val prep = (System.nanoTime() - t) / 1e6; t = System.nanoTime()
        val tok = p.tokens(x)
        val bb = (System.nanoTime() - t) / 1e6; t = System.nanoTime()
        val r = p.identify(x, tok, m)
        val id = (System.nanoTime() - t) / 1e6
        // causal EMA over z-scores, half-life 30 analysed frames, bias corrected
        val e = ema ?: DoubleArray(r.scores.size).also { ema = it }
        for (i in e.indices) e[i] = ALPHA * e[i] + (1 - ALPHA) * r.scores[i]
        emaW = ALPHA * emaW + (1 - ALPHA)
        val top = e.indices.maxBy { e[it] }
        if (top == runTop) runLen++ else { runTop = top; runLen = 1 }
        val done = SystemClock.elapsedRealtimeNanos()
        synchronized(this) {
            frameTs += ts; entryTs += entry; doneTs += done; prepMs += prep; backboneMs += bb; identifyMs += id
            looked += r.looked; pred += r.pred; smoothTop += top
        }
        if (runLen == STABLE_FRAMES && top != lastSpoken && ttsReady) {
            lastSpoken = top
            val uid = "a${announcements.length()}"
            val rec = JSONObject().put("frame", frameTs.size - 1).put("object", objects[top]).put("frame_ts_ns", ts)
                .put("decided_ns", done).put("smoothed_margin", e.sortedDescending().let { (it[0] - it[1]) / emaW })
            synchronized(pendingSpeech) { pendingSpeech[uid] = rec }
            announcements.put(rec)
            tts?.speak(objects[top].replace('-', ' '), TextToSpeech.QUEUE_FLUSH, null, uid)
        }
        main.post { label.text = "${objects[top]}  ${"%.1f".format(bb + id + prep)} ms" }
    }

    private fun sampleThermal() {
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val pm = getSystemService(PowerManager::class.java)
        thermal.put(JSONObject().put("t_s", (SystemClock.elapsedRealtimeNanos() - t0) / 1e9)
            .put("battery_c", (battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1) / 10.0)
            .put("thermal_status", pm.currentThermalStatus).put("headroom_10s", pm.getThermalHeadroom(10).toDouble())
            .put("frames", frameTs.size))
        main.postDelayed(::sampleThermal, 5000)
    }

    /** Stops the camera and writes the log; called by the device test on the main thread. */
    fun stopAndDump(): File {
        main.removeCallbacksAndMessages(null)
        provider?.unbindAll()
        sampleThermal(); main.removeCallbacksAndMessages(null)
        val out = File(filesDir, "m1").apply { mkdirs() }.let { File(it, "live_$user.json") }
        synchronized(this) {
            val j = JSONObject().put("user", user).put("objects", JSONArray(objects)).put("setup_ms", setupMs)
                .put("timestamp_source", timestampSource).put("stable_frames", STABLE_FRAMES).put("ema_half_life", HALF_LIFE)
                .put("frame_ts_ns", JSONArray(frameTs)).put("entry_ns", JSONArray(entryTs)).put("done_ns", JSONArray(doneTs))
                .put("prep_ms", JSONArray(prepMs)).put("backbone_ms", JSONArray(backboneMs)).put("identify_ms", JSONArray(identifyMs))
                .put("looked", JSONArray(looked)).put("pred", JSONArray(pred)).put("smooth_top", JSONArray(smoothTop))
                .put("thermal", thermal).put("announcements", synchronized(pendingSpeech) { announcements })
            out.writeText(j.toString())
        }
        return out
    }

    override fun onDestroy() {
        tts?.shutdown(); worker.shutdown()
        worker.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS)
        pipeline?.close()
        super.onDestroy()
    }

    companion object {
        const val EXTRA_USER = "user"
        const val TAG = "TeachableM1"
        const val HALF_LIFE = 30
        const val STABLE_FRAMES = 15
        val ALPHA = 0.5.pow(1.0 / HALF_LIFE)
    }
}
