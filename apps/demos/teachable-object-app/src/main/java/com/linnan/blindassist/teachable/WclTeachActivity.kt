package com.linnan.blindassist.teachable

import android.Manifest
import android.content.Intent
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
    private val store by lazy { WclTeachingStore(WclTeachingStore.directory(filesDir, WclTeachingStore.PERSONAL)) }
    private lateinit var preview: PreviewView
    private lateinit var status: TextView
    private lateinit var objectList: TextView
    private lateinit var name: EditText
    private lateinit var record: Button
    private lateinit var save: Button
    private lateinit var resume: Button
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
        preview = PreviewView(this).apply { importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO }
        status = TextView(this).apply { textSize = 18f; setTextColor(Color.BLACK); accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE }
        objectList = TextView(this).apply { textSize = 18f; setTextColor(Color.BLACK) }
        name = EditText(this).apply { hint = "物品名字，例如我的钥匙"; contentDescription = "物品名字"; setSingleLine(); setTextColor(Color.BLACK); setHintTextColor(Color.DKGRAY) }
        record = Button(this).apply { text = "录制 10 秒"; setOnClickListener { if (recording) finishRecording() else startRecording() } }
        save = Button(this).apply { text = "保存教学并开始识别"; setOnClickListener { buildAndRecognize() } }
        resume = Button(this).apply { text = "继续识别已保存的物品"; setOnClickListener { openLive() } }
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            val pad = (16 * resources.displayMetrics.density).toInt(); setPadding(pad, pad, pad, pad)
            addView(TextView(this@WclTeachActivity).apply {
                text = "教我的物品\n给物品起名字，再单独放在镜头前慢慢转动。每件录制10秒，可随时结束。\n目前只在教过的物品中选择；没有这些物品时也可能误报。"; textSize = 20f; setTextColor(Color.BLACK)
            })
            addView(objectList); addView(name); addView(record); addView(status); addView(save); addView(resume)
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
        show("教学图片保存在这部手机上。新录物品不会覆盖已教物品；保存成功后可继续识别。")
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) cameraPermission.launch(Manifest.permission.CAMERA)
    }

    override fun onResume() {
        super.onResume()
        if (::preview.isInitialized && checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        if (::resume.isInitialized) refresh()
    }

    private fun show(message: String) = runOnUiThread { if (!isDestroyed) status.text = message }
    private fun refresh() {
        val objects = store.objects()
        objectList.text = if (objects.isEmpty()) "还没有教学物品" else "已录制 ${objects.size} 件：" + objects.joinToString { "${it.name}，${it.frames.size}帧" }
        record.isEnabled = !busy; name.isEnabled = !busy && !recording
        save.isEnabled = !busy && !recording && objects.isNotEmpty()
        resume.isEnabled = !busy && !recording && store.hasSavedLibrary()
        record.text = if (recording) "结束本次录制" else "录制 10 秒"
    }

    private fun startRecording() {
        if (busy || recording) return
        if (provider == null) { show("相机尚未准备好，请稍候"); return }
        val enteredName = name.text.toString().trim()
        try {
            recordDir = store.newObject(enteredName); recordedName = enteredName
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
        if (busy || recording) return
        busy = true; refresh(); show("正在准备识别模型并保存教学，请稍候。")
        worker.execute {
            try {
                val model = File(filesDir, "wcl/wcl_cached.dlc")
                val identity = WclTeachingStore.modelIdentity(model)
                WclPipeline(model, applicationInfo.nativeLibraryDir).use { store.build(it, identity, ::show) }
                runOnUiThread {
                    if (isDestroyed) return@runOnUiThread
                    busy = false; refresh(); show("教学已保存。")
                    if (lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.STARTED)) openLive()
                }
            } catch (error: Exception) {
                runOnUiThread { if (!isDestroyed) { busy = false; refresh(); show(error.message ?: "保存失败，已有教学记录仍保留。") } }
            }
        }
    }

    private fun openLive() {
        if (busy || recording) return
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            cameraPermission.launch(Manifest.permission.CAMERA); return
        }
        startActivity(Intent(this, WclLiveActivity::class.java).putExtra(WclLiveActivity.EXTRA_LIBRARY, WclTeachingStore.PERSONAL))
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
}
