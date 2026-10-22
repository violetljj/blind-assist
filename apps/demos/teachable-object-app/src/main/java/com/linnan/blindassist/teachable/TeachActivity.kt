package com.linnan.blindassist.teachable

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Range
import android.util.Size
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
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
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.pow

/**
 * M2: teach your own objects and test live recognition.
 *  - Record: type a name, hold the object in front of the camera for RECORD_S seconds; every 10th analysed frame
 *    (centre square, 224) is saved -> about 30 teaching frames per object.
 *  - Build: K frames per object from the local <= 1.5T budget (as the thesis budget variant), backbone on the HTP,
 *    discriminative local memory built on the phone.
 *  - Live: identify -> EMA30 -> speak once the smoothed top-1 is stable for STABLE_FRAMES frames.
 *  - Test: prompted cover-and-reveal trials. Cover the lens (dark frames), then show the prompted object. The first
 *    non-dark frame after >= DARK_FRAMES dark frames is the reveal; latency = TTS start - reveal frame exposure time.
 *    Dark frames neither update the EMA nor trigger speech; each cover resets the EMA, stability run and last announcement.
 * Data: filesDir/m2/objects/<n>_<name>/ (JPEG frames), memory.json + memory.raw, trials_<time>.json.
 */
class TeachActivity : ComponentActivity() {
    private enum class Mode { IDLE, RECORD, LIVE, TEST }

    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val root by lazy { File(filesDir, "m2").apply { mkdirs() } }
    private val objectsDir by lazy { File(root, "objects").apply { mkdirs() } }
    private lateinit var status: TextView
    private lateinit var nameField: EditText
    private var pipeline: TeachablePipeline? = null
    private var simmax: SnpeEngine? = null                  // HTP other-object max for the memory build; CPU if absent
    private var simtop: SnpeEngine? = null                  // HTP own top-6 for the memory build; CPU if absent
    private var memory: TeachablePipeline.Memory? = null
    private var names: List<String> = emptyList()
    private var tts: TextToSpeech? = null
    private val ttsStarts = HashMap<String, Long>()
    @Volatile private var mode = Mode.IDLE

    private var recordDir: File? = null; private var recordUntil = 0L; private var recordCount = 0; private var frameIndex = 0
    private var ema: DoubleArray? = null; private var emaW = 0.0
    private var runTop = -1; private var runLen = 0; private var lastSpoken = -1
    // test state
    private val trials = JSONArray(); private var prompts: List<Int> = emptyList(); private var promptIndex = 0
    private var darkRun = 0; private var armed = false; private var revealTs = 0L; private var trial: JSONObject? = null
    private val frames = JSONArray(); private var testFile: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val preview = PreviewView(this)
        status = TextView(this).apply { textSize = 18f; setPadding(16, 8, 16, 8) }
        nameField = EditText(this).apply { hint = "物品名字"; setSingleLine() }
        fun button(text: String, onClick: () -> Unit) = Button(this).apply { this.text = text; setOnClickListener { onClick() } }
        val row1 = LinearLayout(this).apply {
            addView(nameField, LinearLayout.LayoutParams(0, WRAP_CONTENT, 2f))
            addView(button("录制 ${RECORD_S} 秒") { startRecord() }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        }
        val row2 = LinearLayout(this).apply {
            for ((t, f) in listOf("建记忆" to ::build, "识别" to { setMode(Mode.LIVE) }, "测试" to ::startTest,
                                  "停止" to { setMode(Mode.IDLE) }, "清空" to ::clearAll))
                addView(button(t, f), LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(preview, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
            addView(status); addView(row1); addView(row2)
        })
        tts = TextToSpeech(this) {}.apply {
            setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(id: String) { synchronized(ttsStarts) { ttsStarts[id] = SystemClock.elapsedRealtimeNanos() } }
                override fun onDone(id: String) {}
                @Deprecated("Deprecated in Java") override fun onError(id: String) {}
            })
        }
        worker.execute {
            val m0 = File(filesDir, "m0")
            pipeline = TeachablePipeline(File(m0, "backbone_fp32.dlc"), File(m0, "head_dyn.dlc"), applicationInfo.nativeLibraryDir)
            File(m0, "simmax.dlc").takeIf { it.isFile }?.let { simmax = SnpeEngine(it.path, true, applicationInfo.nativeLibraryDir, listOf("mx")) }
            File(m0, "simtop.dlc").takeIf { it.isFile }?.let { simtop = SnpeEngine(it.path, true, applicationInfo.nativeLibraryDir, listOf("tv")) }
            loadMemory()
            show(summary())
        }
        startCamera(preview)
    }

    private fun show(text: String) = main.post { status.text = text }
    private fun taught() = objectsDir.listFiles { f -> f.isDirectory }!!.sortedBy { it.name.substringBefore('_').toInt() }
    private fun summary() = "已教 ${taught().size} 件：" + taught().joinToString { "${it.name.substringAfter('_')}(${it.list()!!.size})" } +
        if (memory != null) "\n记忆已建好（${names.size} 件），可以识别/测试" else "\n还没有建记忆"

    private fun setMode(m: Mode) {
        if ((m == Mode.LIVE || m == Mode.TEST) && memory == null) { show("先建记忆"); return }
        worker.execute { ema = null; emaW = 0.0; runTop = -1; runLen = 0; lastSpoken = -1; mode = m }
        if (m == Mode.IDLE) show(summary())
    }

    private fun startRecord() {
        val name = nameField.text.toString().trim().replace(Regex("[^\\p{L}\\p{N} -]"), "")
        if (name.isEmpty()) { show("先输入物品名字"); return }
        val dir = File(objectsDir, "${taught().size}_$name").apply { mkdirs() }
        worker.execute { recordDir = dir; recordCount = 0; recordUntil = SystemClock.elapsedRealtime() + RECORD_S * 1000; mode = Mode.RECORD }
        nameField.setText("")
        show("录制中：$name，请把它放在镜头前，慢慢转动")
    }

    private fun clearAll() = worker.execute {
        mode = Mode.IDLE; objectsDir.deleteRecursively(); objectsDir.mkdirs()
        File(root, "memory.json").delete(); File(root, "memory.raw").delete(); memory = null; names = emptyList()
        show(summary())
    }

    /** Thesis budget_k: largest frames-per-object K whose analytic personalisation cost fits 1.5 TMACs. */
    private fun budgetK(nObj: Int): Int {
        var best = 1
        for (k in 1 until 400) {
            val n = k * 256.0; val oth = minOf(30000.0, (nObj - 1) * n); val own = minOf(30000.0, n)
            val mem = nObj * (n * oth * 384 + n * own * 384 + 15 * (0.15 * n + 1).toInt() * 32 * 384.0)
            if (nObj * k * 6.123561984e9 + mem > 1.5e12) break
            best = k
        }
        return best
    }

    private fun build() = worker.execute {
        mode = Mode.IDLE
        val p = pipeline ?: return@execute
        val dirs = taught()
        if (dirs.size < 2 || dirs.size > TeachablePipeline.OMAX) { show("需要 2–${TeachablePipeline.OMAX} 件物品，现在 ${dirs.size} 件"); return@execute }
        val k = budgetK(dirs.size); val t0 = SystemClock.elapsedRealtimeNanos()
        val chunks = ArrayList<FloatArray>(); val counts = IntArray(dirs.size); val cls = List(dirs.size) { FloatArray(TeachablePipeline.D) }
        for ((o, d) in dirs.withIndex()) {
            val all = d.listFiles()!!.sortedBy { it.name }
            val pick = if (all.size <= k) all else List(k) { all[Math.round(it * (all.size - 1.0) / (k - 1).coerceAtLeast(1)).toInt()] }.distinct()
            for (f in pick) {
                val tok = p.tokens(p.image(f))
                for (i in 0 until TeachablePipeline.D) cls[o][i] += tok[i]
                chunks += tok.copyOfRange(TeachablePipeline.D, tok.size); counts[o] += 256
            }
        }
        val teachMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
        show("建记忆中（${dirs.size} 件，每件 ≤$k 帧）…")
        val patches = FloatArray(chunks.sumOf { it.size }).also { a -> var i = 0; for (c in chunks) { c.copyInto(a, i); i += c.size } }
        val t1 = SystemClock.elapsedRealtimeNanos()
        val parts = Native.build(patches, counts, TeachablePipeline.KEEP, TeachablePipeline.PARTS, simmax, simtop).parts
        val buildMs = (SystemClock.elapsedRealtimeNanos() - t1) / 1e6
        names = dirs.map { it.name.substringAfter('_') }
        memory = p.memory(dirs.size, parts, cls)
        File(root, "memory.raw").writeBytes(java.nio.ByteBuffer.allocate(parts.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .also { b -> b.asFloatBuffer().put(parts) }.array())
        val protos = JSONArray(cls.map { JSONArray(it.toList()) })
        File(root, "memory.json").writeText(JSONObject().put("names", JSONArray(names)).put("K", k).put("counts", JSONArray(counts.toList()))
            .put("teach_ms", teachMs).put("build_ms", buildMs).put("cls_sum", protos).toString())
        show("记忆已建好：${names.size} 件，每件 ${counts.map { it / 256 }} 帧；前向 ${"%.1f".format(teachMs / 1000)} s，建记忆 ${"%.1f".format(buildMs / 1000)} s")
    }

    private fun loadMemory() {
        val j = File(root, "memory.json"); val raw = File(root, "memory.raw")
        if (!j.isFile || !raw.isFile) return
        val meta = JSONObject(j.readText()); val p = pipeline ?: return
        names = List(meta.getJSONArray("names").length()) { meta.getJSONArray("names").getString(it) }
        val b = raw.readBytes(); val parts = FloatArray(b.size / 4)
        java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(parts)
        val cs = meta.getJSONArray("cls_sum")
        memory = p.memory(names.size, parts, List(names.size) { o -> FloatArray(TeachablePipeline.D) { cs.getJSONArray(o).getDouble(it).toFloat() } })
    }

    private fun startTest() {
        if (memory == null) { show("先建记忆"); return }
        worker.execute {
            prompts = (0 until TRIALS_PER_OBJECT).flatMap { names.indices.shuffled() }
            promptIndex = 0
            while (trials.length() > 0) trials.remove(0)
            while (frames.length() > 0) frames.remove(0)
            testFile = File(root, "trials_${System.currentTimeMillis()}.json"); trial = null; darkRun = 0; armed = false
            ema = null; emaW = 0.0; runTop = -1; runLen = 0; lastSpoken = -1; mode = Mode.TEST
            prompt()
        }
    }

    private fun prompt() {
        if (promptIndex >= prompts.size) { mode = Mode.IDLE; saveTest(); show("测试完成：${trials.length()} 次，已保存 ${testFile?.name}"); return }
        show("第 ${promptIndex + 1}/${prompts.size} 次：先用手遮住镜头，再展示【${names[prompts[promptIndex]]}】")
    }

    private fun saveTest() {
        testFile?.writeText(JSONObject().put("names", JSONArray(names)).put("stable_frames", STABLE_FRAMES).put("ema_half_life", HALF_LIFE)
            .put("dark_threshold", DARK).put("dark_frames", DARK_FRAMES)
            .put("trials", trials).put("frames", frames).toString())
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun startCamera(view: PreviewView) {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val cp = future.get(); val fps = Range(30, 30)
            val pb = Preview.Builder(); Camera2Interop.Extender(pb).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
            val preview = pb.build().also { it.surfaceProvider = view.surfaceProvider }
            val ab = ImageAnalysis.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            Camera2Interop.Extender(ab).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
            val analysis = ab.build().also { it.setAnalyzer(worker, ::analyze) }
            cp.unbindAll(); cp.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, mainExecutor)
    }

    private fun crop(image: ImageProxy): Bitmap {
        val bmp = image.toBitmap(); val rot = image.imageInfo.rotationDegrees
        val side = minOf(bmp.width, bmp.height); val s = TeachablePipeline.SIZE.toFloat() / side
        return Bitmap.createBitmap(bmp, (bmp.width - side) / 2, (bmp.height - side) / 2, side, side,
            Matrix().apply { postScale(s, s); postRotate(rot.toFloat()) }, true).also { bmp.recycle() }
    }

    private fun luminance(b: Bitmap): Double {
        val px = IntArray(TeachablePipeline.SIZE * TeachablePipeline.SIZE)
        b.getPixels(px, 0, TeachablePipeline.SIZE, 0, 0, TeachablePipeline.SIZE, TeachablePipeline.SIZE)
        return px.sumOf { (0.299 * ((it shr 16) and 255) + 0.587 * ((it shr 8) and 255) + 0.114 * (it and 255)) } / px.size
    }

    private fun analyze(image: ImageProxy) {
        val m = mode; val p = pipeline
        if (m == Mode.IDLE || p == null) { image.close(); return }
        val ts = image.imageInfo.timestamp
        val bmp = crop(image); image.close(); frameIndex++
        if (m == Mode.RECORD) {
            if (frameIndex % 10 == 0) File(recordDir, "f%03d.jpg".format(recordCount++)).outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            bmp.recycle()
            if (SystemClock.elapsedRealtime() > recordUntil) { mode = Mode.IDLE; show("录好了 ${recordCount} 帧。" + summary()) }
            return
        }
        val mem = memory ?: run { bmp.recycle(); return }
        val lum = luminance(bmp)
        val dark = lum < DARK
        if (m == Mode.TEST && trial == null && promptIndex < prompts.size) {
            // armed after DARK_FRAMES consecutive dark frames; the reveal is the first non-dark frame after arming
            // (earliest partial view, so the latency is not understated)
            if (dark) {
                darkRun++
                // covering the lens is a scene break: restart smoothing and stability, so the next announcement needs
                // STABLE_FRAMES new frames (without this, the first uncovered frame re-announced the previous object)
                if (darkRun == DARK_FRAMES && !armed) {
                    armed = true; ema = null; emaW = 0.0; runTop = -1; runLen = 0; lastSpoken = -1
                    show("已遮住，现在展示【${names[prompts[promptIndex]]}】")
                }
            } else {
                darkRun = 0
                if (armed) {
                    armed = false; revealTs = ts
                    trial = JSONObject().put("target", names[prompts[promptIndex]]).put("reveal_ts_ns", ts).put("reveal_lum", lum)
                }
            }
        }
        if (dark) { bmp.recycle(); logFrame(m, ts, lum, -1, -1, 0.0); return }
        val t = System.nanoTime()
        val x = p.image(bmp); bmp.recycle()
        val r = p.identify(x, p.tokens(x), mem)
        val ms = (System.nanoTime() - t) / 1e6
        val e = ema ?: DoubleArray(r.scores.size).also { ema = it }
        for (i in e.indices) e[i] = ALPHA * e[i] + (1 - ALPHA) * r.scores[i]
        emaW = ALPHA * emaW + (1 - ALPHA)
        val top = e.indices.maxBy { e[it] }
        if (top == runTop) runLen++ else { runTop = top; runLen = 1 }
        logFrame(m, ts, lum, r.pred, top, ms)
        if (runLen >= STABLE_FRAMES && top != lastSpoken) {
            lastSpoken = top
            val uid = "u${SystemClock.elapsedRealtimeNanos()}"
            val name = names[top]
            tts?.language = if (name.any { Character.UnicodeScript.of(it.code) == Character.UnicodeScript.HAN }) Locale.SIMPLIFIED_CHINESE else Locale.US
            tts?.speak(name, TextToSpeech.QUEUE_FLUSH, null, uid)
            trial?.let { tr ->
                if (!tr.has("announced")) {
                    tr.put("announced", name).put("announce_frame_ts_ns", ts).put("decided_ns", SystemClock.elapsedRealtimeNanos()).put("uid", uid)
                    main.postDelayed({ finishTrial() }, 1500)
                }
            }
        }
        trial?.let { tr -> if (!tr.has("announced") && ts - revealTs > TIMEOUT_NS) { tr.put("announced", JSONObject.NULL); finishTrial() } }
        main.post { if (m == Mode.LIVE) status.text = "${names[top]}  ${"%.0f".format(ms)} ms" }
    }

    private fun logFrame(m: Mode, ts: Long, lum: Double, pred: Int, top: Int, ms: Double) {
        if (m != Mode.TEST) return
        frames.put(JSONArray().put(ts).put(Math.round(lum * 10) / 10.0).put(pred).put(top).put(Math.round(ms * 100) / 100.0))
    }

    private fun finishTrial() = worker.execute {
        val tr = trial ?: return@execute
        if (tr.has("uid")) synchronized(ttsStarts) { ttsStarts[tr.getString("uid")]?.let { tr.put("tts_start_ns", it) } }
        tr.put("correct", tr.opt("announced") == tr.getString("target"))
        trials.put(tr); trial = null; promptIndex++; saveTest()
        if (mode == Mode.TEST) prompt()
    }

    override fun onDestroy() {
        tts?.shutdown(); worker.shutdown(); worker.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS); simmax?.close(); simtop?.close(); pipeline?.close()
        super.onDestroy()
    }

    companion object {
        const val RECORD_S = 10
        const val HALF_LIFE = 30
        const val STABLE_FRAMES = 15
        const val DARK = 25.0
        const val DARK_FRAMES = 15
        const val TRIALS_PER_OBJECT = 3
        const val TIMEOUT_NS = 6_000_000_000L
        val ALPHA = 0.5.pow(1.0 / HALF_LIFE)
    }
}
