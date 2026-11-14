package com.linnan.blindassist.teachable

import android.Manifest
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.Executors

/** User-operated WCL teaching: explicit capture, persistent prototypes, then the existing live chain. */
class WclTeachActivity : ComponentActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val testLibrary by lazy {
        intent.getStringExtra(EXTRA_TEST_LIBRARY)?.takeIf {
            applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0 && it.startsWith("bui_")
        }
    }
    private val library by lazy { testLibrary ?: WclTeachingStore.PERSONAL }
    private val store by lazy { WclTeachingStore(WclTeachingStore.directory(filesDir, library)) }
    private val settings by lazy { getSharedPreferences("personal_teaching", MODE_PRIVATE) }
    private lateinit var preview: PreviewView
    private lateinit var status: TextView
    private lateinit var objectList: TextView
    private lateinit var name: EditText
    private lateinit var record: Button
    private lateinit var save: Button
    private lateinit var resume: Button
    private lateinit var trial: Button
    private lateinit var undo: Button
    private lateinit var useB: CheckBox
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var previewUseCase: Preview? = null
    @Volatile private var busy = false
    @Volatile private var recording = false
    private var recordDir: File? = null
    private var recordStart = 0L
    private var nextFrame = 0L
    private var frames = 0
    private var announcedSecond = -1
    private var recordedName = ""

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else show("需要相机权限才能录制教学。允许权限后重新打开此页。")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (testLibrary != null && android.os.Build.VERSION.SDK_INT >= 27) {
            setShowWhenLocked(true); setTurnScreenOn(true)
        }
        preview = PreviewView(this).apply { importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO }
        status = TextView(this).apply { textSize = 18f; setTextColor(Color.BLACK); accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE }
        objectList = TextView(this).apply { textSize = 18f; setTextColor(Color.BLACK) }
        name = EditText(this).apply { hint = "物品名字，例如我的钥匙"; contentDescription = "物品名字"; setSingleLine(); setTextColor(Color.BLACK); setHintTextColor(Color.DKGRAY) }
        record = Button(this).apply { text = "录制 10 秒"; setOnClickListener { if (recording) finishRecording() else startRecording() } }
        save = Button(this).apply { id = SAVE_BUTTON_ID; text = "保存教学并开始识别"; setOnClickListener { buildAndRecognize() } }
        resume = Button(this).apply { text = "继续识别已保存的物品"; setOnClickListener { openLive() } }
        trial = Button(this).apply { text = "打开呈现对照试验"; setOnClickListener { openLive(presentationTrials = true) } }
        undo = Button(this).apply { id = UNDO_BUTTON_ID; text = "撤销指定教学片段"; setOnClickListener { chooseUndo() } }
        useB = CheckBox(this).apply {
            id = COMPLETE_RECOGNITION_ID
            text = "使用完整识别（B＋WCL）"
            isChecked = testLibrary != null || settings.getBoolean("bwcl", false)
            setOnCheckedChangeListener { _, checked ->
                if (testLibrary == null) settings.edit().putBoolean("bwcl", checked).apply()
            }
        }
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            val pad = (16 * resources.displayMetrics.density).toInt(); setPadding(pad, pad, pad, pad)
            addView(TextView(this@WclTeachActivity).apply {
                text = "教我的物品\n给物品起名字，再单独放在镜头前慢慢转动。每段录制10秒，可随时结束。输入已有名称可补教。\n目前只在教过的物品中选择；没有这些物品时也可能误报。"; textSize = 20f; setTextColor(Color.BLACK)
            })
            addView(objectList); addView(name); addView(record); addView(status); addView(useB); addView(save); addView(resume); addView(trial)
            addView(undo)
            addView(Button(this@WclTeachActivity).apply {
                text = "打开轻量识别演示"
                setOnClickListener { if (!busy && !recording) startActivity(Intent(this@WclTeachActivity, TeachActivity::class.java)) }
            })
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(preview, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
            addView(ScrollView(this@WclTeachActivity).apply { addView(controls) }, LinearLayout.LayoutParams(MATCH_PARENT, 0, 2f))
        })
        refresh()
        show("教学图片保存在这部手机上。新录片段经保存后才会进入识别库。")
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) cameraPermission.launch(Manifest.permission.CAMERA)
    }

    override fun onResume() {
        super.onResume()
        if (::preview.isInitialized && checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        if (::resume.isInitialized) refresh()
    }

    private fun show(message: String) = runOnUiThread { if (!isDestroyed) status.text = message }
    private fun refresh() {
        val lists = runCatching { store.savedClips() to store.pendingClips() }
        val (saved, pending) = lists.getOrDefault(emptyList<WclTeachingStore.ObjectFrames>() to emptyList())
        objectList.text = if (lists.isFailure) "无法读取教学库" else buildString {
            append("已保存 ${saved.size} 段，待保存 ${pending.size} 段。")
            if (saved.isNotEmpty()) append("\n已保存物品：").append(saved.map { it.name }.distinct().joinToString("、"))
            if (pending.isNotEmpty()) append("\n待保存名称：").append(pending.map { it.name }.distinct().joinToString("、"))
        }
        record.isEnabled = !busy; name.isEnabled = !busy && !recording
        useB.isEnabled = !busy && !recording && testLibrary == null
        save.isEnabled = !busy && !recording && (saved.isNotEmpty() || pending.isNotEmpty())
        resume.isEnabled = !busy && !recording && saved.isNotEmpty()
        undo.isEnabled = save.isEnabled
        trial.isEnabled = resume.isEnabled
        record.text = if (recording) "结束本次录制" else "录制 10 秒"
    }

    private fun startRecording() {
        if (busy || recording) return
        if (provider == null) { show("相机尚未准备好，请稍候"); return }
        val enteredName = name.text.toString().trim()
        try {
            recordDir = store.newClip(enteredName); recordedName = enteredName
            recordStart = SystemClock.elapsedRealtime(); nextFrame = 0L; frames = 0; announcedSecond = -1
            recording = true; name.setText(""); refresh()
            show("正在录制：$enteredName。请保持物品在画面中央，慢慢转动。")
        } catch (error: Exception) { show(error.message ?: "无法开始录制") }
    }

    private fun finishRecording() {
        if (!recording) return
        recording = false; busy = true
        val directory = requireNotNull(recordDir); val capturedName = recordedName
        worker.execute {
            val count = frames
            val complete = store.finishObject(directory)
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                busy = false; refresh()
                show("${capturedName}录制结束，共${count}帧。" + if (complete) "可以添加下一件，或保存教学开始识别。" else "时间不足，未加入教学库，请重新录制。")
            }
        }
    }

    private fun buildAndRecognize() {
        val saveNs = SystemClock.elapsedRealtimeNanos()
        if (busy || recording) return
        val bwcl = useB.isChecked
        if (bwcl && store.objects().size !in 2..TeachablePipeline.OMAX) {
            show("完整识别需要2到${TeachablePipeline.OMAX}件已教物品。")
            return
        }
        if (bwcl && !listOf("backbone_b_cached.dlc", "bhead_fp32.dlc").all { File(filesDir, "bwcl/$it").isFile }) {
            show("完整识别模型尚未安装。可以取消勾选后保存。")
            return
        }
        busy = true; refresh(); show("正在检查待保存片段，请稍候。")
        worker.execute {
            try {
                val prepared = prepareTeaching()
                if (prepared.warnings.isEmpty()) commitTeaching(prepared, saveNs)
                else runOnUiThread { if (!isDestroyed) warnBeforeSave(prepared, saveNs) }
            } catch (error: Exception) { finishEdit(error.message ?: "未能保存教学。") }
        }
    }

    private fun prepareTeaching(removing: WclTeachingStore.ObjectFrames? = null): WclTeachingStore.Prepared {
        val model = File(filesDir, WclPipeline.MODEL)
        val identity = WclTeachingStore.modelIdentity(model)
        val reuse = testLibrary == null || !intent.getBooleanExtra(EXTRA_TEST_FORCE_FULL_WCL, false)
        return WclPipeline(model, applicationInfo.nativeLibraryDir).use {
            store.prepare(it, identity, reuse, removing = removing, progress = ::show)
        }
    }

    private fun warnBeforeSave(prepared: WclTeachingStore.Prepared, saveNs: Long) {
        val warnings = prepared.warnings.joinToString("\n") {
            "拟存为「${it.clip.name}」，片段也可能是「${it.match.suggested}」。"
        }
        AlertDialog.Builder(this).setTitle("请核对教学名称")
            .setMessage("$warnings\n暂缓保存后，可以继续使用已保存的物品，或撤销待保存的片段。")
            .setNegativeButton("暂缓保存") { _, _ -> finishEdit("已暂缓保存，可继续识别已保存的物品。") }
            .setPositiveButton("仍然保存") { _, _ -> worker.execute { commitTeaching(prepared, saveNs) } }
            .setOnCancelListener { finishEdit("已暂缓保存。") }
            .show()
    }

    private fun commitTeaching(prepared: WclTeachingStore.Prepared, saveNs: Long) {
        try {
            val built = store.commit(prepared, ::show)
            val savePhaseMs = (SystemClock.elapsedRealtimeNanos() - saveNs) / 1e6
            finishEdit("教学已保存。") {
                if (lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.STARTED)) {
                    openLive(saveNs = saveNs, built = built, wclSavePhaseMs = savePhaseMs)
                }
            }
        } catch (error: Exception) { finishEdit(error.message ?: "未能保存教学。") }
    }

    private fun chooseUndo() {
        if (busy || recording) return
        val pending = store.pendingClips()
        val clips = (store.savedClips() + pending).asReversed()
        if (clips.isEmpty()) return
        val date = java.text.DateFormat.getDateTimeInstance(java.text.DateFormat.SHORT, java.text.DateFormat.MEDIUM)
        val choices = clips.map { clip ->
            val time = clip.directory.name.substringBefore('_').toLongOrNull()?.let { date.format(java.util.Date(it)) }
            "${clip.name} · ${clip.frames.size}帧 · ${time ?: clip.directory.name}" + if (clip in pending) " · 待保存" else ""
        }.toTypedArray()
        AlertDialog.Builder(this).setTitle("撤销指定教学片段").setItems(choices) { _, index ->
            AlertDialog.Builder(this).setTitle("确认撤销").setMessage(choices[index])
                .setNegativeButton("取消", null)
                .setPositiveButton("撤销此片段") { _, _ -> undoClip(clips[index], clips[index] in pending) }
                .show()
        }.setNegativeButton("取消", null).show()
    }

    private fun undoClip(clip: WclTeachingStore.ObjectFrames, pending: Boolean) {
        busy = true; refresh(); show("正在撤销「${clip.name}」的教学片段。")
        worker.execute {
            try {
                if (pending) store.discardDraft(clip) else store.commit(prepareTeaching(removing = clip), ::show)
                finishEdit("已撤销「${clip.name}」的教学片段。")
            } catch (error: Exception) { finishEdit(error.message ?: "未能撤销教学片段。") }
        }
    }

    private fun finishEdit(message: String, next: () -> Unit = {}) = runOnUiThread {
        if (!isDestroyed) {
            busy = false
            refresh()
            show(message)
            next()
        }
    }

    private fun openLive(
        presentationTrials: Boolean = false, saveNs: Long = 0L,
        built: WclTeachingStore.BuildResult? = null, wclSavePhaseMs: Double = 0.0,
    ) {
        if (busy || recording) return
        val bwcl = !presentationTrials && useB.isChecked && store.savedObjects().size in 2..TeachablePipeline.OMAX
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            cameraPermission.launch(Manifest.permission.CAMERA); return
        }
        startActivity(Intent(this, WclLiveActivity::class.java)
            .putExtra(WclLiveActivity.EXTRA_LIBRARY, library)
            .putExtra(WclLiveActivity.EXTRA_BWCL, bwcl)
            .putExtra(WclLiveActivity.EXTRA_PRESENTATION_TRIALS, presentationTrials).apply {
                if (testLibrary != null && saveNs != 0L && built != null) {
                    putExtra(EXTRA_TEST_SAVE_NS, saveNs)
                    putExtra(EXTRA_TEST_FORCE_FULL_B, intent.getBooleanExtra(EXTRA_TEST_FORCE_FULL_B, false))
                    putExtra(EXTRA_TEST_WCL_ENCODED, built.encodedFrames)
                    putExtra(EXTRA_TEST_WCL_REUSED, built.reusedFrames)
                    putExtra(EXTRA_TEST_WCL_BUILD_MS, built.buildMs)
                    putExtra(EXTRA_TEST_WCL_SAVE_PHASE_MS, wclSavePhaseMs)
                    putStringArrayListExtra(EXTRA_TEST_FIXED_QUERY, intent.getStringArrayListExtra(EXTRA_TEST_FIXED_QUERY))
                }
            })
    }

    internal fun isCameraReady(): Boolean = analysis?.let { provider?.isBound(it) == true } == true

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (isDestroyed || !lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.STARTED)) return@addListener
            try {
                val cp = future.get(); provider = cp
                previewUseCase?.let { cp.unbind(it) }; analysis?.let { cp.unbind(it) }
                val p = Preview.Builder().build().also { it.surfaceProvider = preview.surfaceProvider }; previewUseCase = p
                val a = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build().also { it.setAnalyzer(worker, ::capture) }
                analysis = a; cp.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, p, a)
            } catch (error: Exception) { show("相机暂时不可用：${error.message}") }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun capture(image: ImageProxy) {
        try {
            val now = SystemClock.elapsedRealtime()
            if (!recording || now < nextFrame) return
            if (now - recordStart >= 10_000) { runOnUiThread { finishRecording() }; return }
            nextFrame = now + 333
            val bitmap = image.toBitmap()
            val side = minOf(bitmap.width, bitmap.height); val scale = WclPipeline.SIZE.toFloat() / side
            val crop = Bitmap.createBitmap(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side,
                Matrix().apply { postScale(scale, scale); postRotate(image.imageInfo.rotationDegrees.toFloat()) }, true)
            try {
                File(requireNotNull(recordDir), "frame_%03d.jpg".format(frames)).outputStream().use { output ->
                    check(crop.compress(Bitmap.CompressFormat.JPEG, 95, output)) { "教学图片保存失败" }
                }
                frames++
            } finally { crop.recycle(); if (crop !== bitmap) bitmap.recycle() }
            val second = ((now - recordStart) / 1000).toInt()
            if (second >= announcedSecond + 3) { announcedSecond = second; show("正在录制$recordedName，还剩${10 - second}秒。") }
        } catch (error: Exception) { recording = false; runOnUiThread { refresh(); show(error.message ?: "录制失败") } }
        finally { image.close() }
    }

    override fun onStop() {
        if (recording) finishRecording()
        analysis?.clearAnalyzer()
        previewUseCase?.let { provider?.unbind(it) }; analysis?.let { provider?.unbind(it) }
        super.onStop()
    }

    override fun onDestroy() {
        worker.shutdown()
        super.onDestroy()
    }

    companion object {
        internal val SAVE_BUTTON_ID = View.generateViewId()
        internal val UNDO_BUTTON_ID = View.generateViewId()
        internal val COMPLETE_RECOGNITION_ID = View.generateViewId()
        internal const val EXTRA_TEST_LIBRARY = "testPersonalLibrary"
        internal const val EXTRA_TEST_FORCE_FULL_B = "testForceFullB"
        internal const val EXTRA_TEST_FORCE_FULL_WCL = "testForceFullWcl"
        internal const val EXTRA_TEST_SAVE_NS = "testSaveNs"
        internal const val EXTRA_TEST_WCL_ENCODED = "testWclEncoded"
        internal const val EXTRA_TEST_WCL_REUSED = "testWclReused"
        internal const val EXTRA_TEST_WCL_BUILD_MS = "testWclBuildMs"
        internal const val EXTRA_TEST_WCL_SAVE_PHASE_MS = "testWclSavePhaseMs"
        internal const val EXTRA_TEST_FIXED_QUERY = "testFixedQuery"
    }
}
