package com.linnan.blindassist

import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
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
    private var leaseBusy by mutableStateOf(MobileFusionLease.busy)
    private var speech by mutableStateOf(true)
    private var vibration by mutableStateOf(true)
    private var demoRules by mutableStateOf(true)
    private var lastSpoken by mutableStateOf<String?>(null)
    private var announcer:MobileFusionAnnouncer?=null
    // Time-dependent display state is computed in the 100 ms loop below and only written when it
    // changes, so the page recomposes on real transitions rather than on every tick.
    private var fresh by mutableStateOf(false)
    private var shownHead by mutableStateOf(false)
    private var shownBody by mutableStateOf(false)
    private var headSeenNs=0L
    private var bodySeenNs=0L
    private var modelsReady by mutableStateOf(MobileFusionModels.ready)
    private var thermal by mutableIntStateOf(PowerManager.THERMAL_STATUS_NONE)
    private var audioWarning by mutableStateOf<String?>(null)
    private var thermalListener:Any?=null
    // Display-only preview state. Compose reads the image only while drawing, so new camera frames
    // redraw one Canvas instead of recomposing the page (same approach as the capture page).
    private val liveFrame = mutableStateOf<ImageBitmap?>(null)
    @Volatile private var liveFrameNs = 0L
    private val pendingFrame = AtomicReference<ImageBitmap?>()
    private val framePostQueued = AtomicBoolean()
    internal fun diagnosticSnapshot()=view
    internal fun isExperimentActive()=active

    private val root get()=File(noBackupFilesDir,"fusion-experiment")
    private val requiredFiles=listOf("camera.json","atom_remap.bin","cnh_strong_m3.onnx","cnh_strong_calibration.bin","dav2s_686x518_fp32.dlc")

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
        // Load DAV2/ToF models while the presenter is still on the idle screen.
        if(requiredFiles.all { File(root,it).isFile }) MobileFusionModels.preload(application,root)
        setContent { BlindAssistTheme {
            Surface(Modifier.fillMaxSize(),color=FxNight) {
                FusionScreen(
                    view=view,active=active,fresh=fresh,shownHead=shownHead,shownBody=shownBody,leaseBusy=leaseBusy,
                    preview={ liveFrame.value },modelsReady=modelsReady,thermal=thermal,audioWarning=audioWarning,
                    speech=speech,vibration=vibration,demoRules=demoRules,lastSpoken=lastSpoken,
                    onStart=::startExperiment,onStop=::stopExperiment,
                    onCalibrate={ session?.calibrateHeight() },
                    onSpeech={ speech=it; preferences.edit().putBoolean("speech",it).apply(); if(!it) announcer?.silence(); checkAudio() },
                    onVibration={ vibration=it; preferences.edit().putBoolean("vibration",it).apply() },
                    onDemoRules={ demoRules=it; preferences.edit().putBoolean("demo_rules",it).apply() },
                    onRaiseVolume=::raiseVolume,
                    onBack={ finish() },
                )
            }
        } }
        lifecycleScope.launch {
            var tick=0
            while(true) {
                val now=SystemClock.elapsedRealtimeNanos()
                // Observe lease changes explicitly so asynchronous cleanup re-enables the button.
                leaseBusy=MobileFusionLease.busy
                modelsReady=MobileFusionModels.ready
                // Drop the preview only after the camera has been silent for 1.5 s.
                if(liveFrame.value!=null && (!active || now-liveFrameNs>1_500_000_000L)) liveFrame.value=null
                // Display smoothing (speech below keeps the strict 750 ms rule): results stay 1.5 s,
                // an alarm is held 0.7 s, so a single late or dropped frame does not flicker.
                val shownFresh=active && view.receiptNs>0 && now-view.receiptNs in 0..1_500_000_000L
                if(shownFresh && view.ready && view.head) headSeenNs=now
                if(shownFresh && view.ready && view.body) bodySeenNs=now
                fresh=shownFresh
                shownHead=shownFresh && view.ready && now-headSeenNs in 0..700_000_000L
                shownBody=shownFresh && view.ready && now-bodySeenNs in 0..700_000_000L
                val strict=active && view.receiptNs>0 && now-view.receiptNs in 0..750_000_000L && view.ready
                announcer?.let { a ->
                    a.speech=speech; a.vibration=vibration
                    a.update(now/1_000_000,strict && view.head,strict && view.body,view.nearestM)?.let { lastSpoken=it }
                }
                if(tick++%10==0) checkAudio()
                delay(100)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.Q) getSystemService(PowerManager::class.java)?.let { power ->
            thermal=power.currentThermalStatus
            val listener=PowerManager.OnThermalStatusChangedListener { status -> thermal=status; applyThermal() }
            thermalListener=listener; power.addThermalStatusListener(listener)
        }
        checkAudio()
    }

    override fun onPause() {
        if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.Q) (thermalListener as? PowerManager.OnThermalStatusChangedListener)?.let {
            getSystemService(PowerManager::class.java)?.removeThermalStatusListener(it)
        }
        thermalListener=null
        super.onPause()
    }

    /** Preview slows down only when the phone is clearly hot; judgement work is never throttled. */
    private fun applyThermal() {
        session?.previewIntervalNs=when {
            thermal>=PowerManager.THERMAL_STATUS_CRITICAL -> 250_000_000L
            thermal>=PowerManager.THERMAL_STATUS_SEVERE -> 125_000_000L
            else -> 66_000_000L
        }
    }

    private fun checkAudio() {
        val audio=getSystemService(AudioManager::class.java)
        val max=audio?.getStreamMaxVolume(AudioManager.STREAM_MUSIC) ?: 0
        val volume=audio?.getStreamVolume(AudioManager.STREAM_MUSIC) ?: 0
        audioWarning=when {
            !speech -> null
            announcer?.speechAvailable==false -> "中文语音引擎不可用，只能看屏幕和震动"
            max>0 && volume*10<max*3 -> "媒体音量偏低（$volume/$max），播报可能听不见"
            else -> null
        }
    }

    private fun raiseVolume() {
        val audio=getSystemService(AudioManager::class.java) ?: return
        val max=audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        audio.setStreamVolume(AudioManager.STREAM_MUSIC,(max*7+9)/10,AudioManager.FLAG_SHOW_UI)
        checkAudio()
    }

    internal fun startExperiment() {
        if(active || MobileFusionLease.busy) return
        if(requiredFiles.any { !File(root,it).isFile }) {
            view=MobileFusionView(message="本机尚未安装实验模型包，请先完成模型部署。"); return
        }
        val preferences=getSharedPreferences("phone_capture",MODE_PRIVATE)
        val atom=preferences.getString("atom_host",preferences.getString("imu_host","192.168.5.9"))!!
        val tof=preferences.getString("host","192.168.5.10")!!
        val token=generation.incrementAndGet()
        active=true; view=MobileFusionView(message=if(MobileFusionModels.ready) "模型已就绪，正在连接三路传感器" else "准备模型…",
            demoRules=demoRules); lastSpoken=null; liveFrame.value=null; headSeenNs=0; bodySeenNs=0
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
        }.also { applyThermalTo(it); it.start() }
    }

    private fun applyThermalTo(target:MobileFusionSession) { session=target; applyThermal() }

    internal fun stopExperiment() {
        generation.incrementAndGet(); active=false
        session?.close(); session=null
        announcer?.silence(); liveFrame.value=null
        view=MobileFusionView(message="已停止，连接与推理正在释放",demoRules=demoRules)
    }

    override fun onStop() { stopExperiment(); super.onStop() }
    override fun onDestroy() {
        announcer?.shutdown(); announcer=null
        if(isFinishing) MobileFusionModels.releaseWhenIdle()
        super.onDestroy()
    }
}
