package com.linnan.blindassist

import android.os.Bundle
import android.os.SystemClock
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.lifecycle.lifecycleScope
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** Fusion demo page: opt-in research preview, separate from the default assist and capture paths. */
class MobileFusionActivity : ComponentActivity() {
    private var session:MobileFusionSession?=null
    private val generation=AtomicLong()
    private var view by mutableStateOf(MobileFusionView(message="连接前，请先结束其他采集或辅助"))
    private var active by mutableStateOf(false)
    private var now by mutableLongStateOf(0L)
    private var leaseBusy by mutableStateOf(MobileFusionLease.busy)
    private var speech by mutableStateOf(true)
    private var vibration by mutableStateOf(true)
    private var demoRules by mutableStateOf(true)
    private var lastSpoken by mutableStateOf<String?>(null)
    private var announcer:MobileFusionAnnouncer?=null
    // Display-only preview state. Compose reads the image only while drawing, so new camera frames
    // redraw one Canvas instead of recomposing the page (same approach as the capture page).
    private val liveFrame = mutableStateOf<ImageBitmap?>(null)
    @Volatile private var liveFrameNs = 0L
    private val pendingFrame = AtomicReference<ImageBitmap?>()
    private val framePostQueued = AtomicBoolean()
    internal fun diagnosticSnapshot()=view
    internal fun isExperimentActive()=active

    override fun onCreate(savedInstanceState:Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge(statusBarStyle=SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
            navigationBarStyle=SystemBarStyle.dark(android.graphics.Color.TRANSPARENT))
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val preferences=getSharedPreferences("fusion_demo",MODE_PRIVATE)
        speech=preferences.getBoolean("speech",true)
        vibration=preferences.getBoolean("vibration",true)
        demoRules=preferences.getBoolean("demo_rules",true)
        announcer=MobileFusionAnnouncer(this)
        setContent { BlindAssistTheme {
            Surface(Modifier.fillMaxSize(),color=FxNight) {
                FusionScreen(
                    view=view,active=active,now=now,leaseBusy=leaseBusy,
                    preview={ liveFrame.value },
                    speech=speech,vibration=vibration,demoRules=demoRules,lastSpoken=lastSpoken,
                    onStart=::startExperiment,onStop=::stopExperiment,
                    onCalibrate={ session?.calibrateHeight() },
                    onSpeech={ speech=it; preferences.edit().putBoolean("speech",it).apply(); if(!it) announcer?.silence() },
                    onVibration={ vibration=it; preferences.edit().putBoolean("vibration",it).apply() },
                    onDemoRules={ demoRules=it; preferences.edit().putBoolean("demo_rules",it).apply() },
                    onBack={ finish() },
                )
            }
        } }
        lifecycleScope.launch { while(true) {
            now=SystemClock.elapsedRealtimeNanos()
            // Observe lease changes explicitly so asynchronous cleanup re-enables the button.
            leaseBusy=MobileFusionLease.busy
            // Drop the preview only after the camera has been silent for 1.5 s.
            if(liveFrame.value!=null && (!active || now-liveFrameNs>1_500_000_000L)) liveFrame.value=null
            val fresh=active && view.receiptNs>0 && now-view.receiptNs in 0..750_000_000L && view.ready
            announcer?.let { a ->
                a.speech=speech; a.vibration=vibration
                a.update(now/1_000_000,fresh && view.head,fresh && view.body,view.nearestM)?.let { lastSpoken=it }
            }
            delay(100)
        } }
    }

    internal fun startExperiment() {
        if(active || MobileFusionLease.busy) return
        val root=File(noBackupFilesDir,"fusion-experiment")
        val required=listOf("camera.json","atom_remap.bin","cnh_strong_m3.onnx","cnh_strong_calibration.bin","dav2s_686x518_fp32.dlc")
        if(required.any { !File(root,it).isFile }) {
            view=MobileFusionView(message="本机尚未安装实验模型包，请先完成模型部署。"); return
        }
        val preferences=getSharedPreferences("phone_capture",MODE_PRIVATE)
        val atom=preferences.getString("atom_host",preferences.getString("imu_host","192.168.5.9"))!!
        val tof=preferences.getString("host","192.168.5.10")!!
        val token=generation.incrementAndGet()
        active=true; view=MobileFusionView(demoRules=demoRules); lastSpoken=null; liveFrame.value=null
        session=MobileFusionSession(application,root,atom,tof,demoRules,publishFrame={ image, receiptNs ->
            pendingFrame.set(image); liveFrameNs=receiptNs
            if(framePostQueued.compareAndSet(false,true)) runOnUiThread {
                framePostQueued.set(false)
                val next=pendingFrame.getAndSet(null)
                if(next!=null && active && generation.get()==token) liveFrame.value=next
            }
        }) { value ->
            runOnUiThread {
                if(active && generation.get()==token && session?.accepts(value)==true) {
                    view=value
                    if(value.terminal) { active=false; session=null }
                }
            }
        }.also { it.start() }
    }

    internal fun stopExperiment() {
        generation.incrementAndGet(); active=false
        session?.close(); session=null
        announcer?.silence(); liveFrame.value=null
        view=MobileFusionView(message="已停止，连接与推理正在释放",demoRules=demoRules)
    }

    override fun onStop() { stopExperiment(); super.onStop() }
    override fun onDestroy() { announcer?.shutdown(); announcer=null; super.onDestroy() }
}
