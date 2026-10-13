package com.linnan.blindassist

import android.os.Bundle
import android.os.SystemClock
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.atomic.AtomicLong

/** Explicit research preview, separate from the user's default assist and capture paths. */
class MobileFusionActivity : ComponentActivity() {
    private var session:MobileFusionSession?=null
    private val generation=AtomicLong()
    private var view by mutableStateOf(MobileFusionView(message="连接前，请先结束其他采集或辅助"))
    private var active by mutableStateOf(false)
    private var now by mutableLongStateOf(0L)
    private var leaseBusy by mutableStateOf(MobileFusionLease.busy)
    internal fun diagnosticSnapshot()=view
    internal fun isExperimentActive()=active

    override fun onCreate(savedInstanceState:Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent { BlindAssistTheme {
            Scaffold { insets ->
                Column(Modifier.fillMaxSize().padding(insets).verticalScroll(rememberScrollState()).padding(20.dp),
                    verticalArrangement=Arrangement.spacedBy(14.dp)) {
                    Text("融合实验",style=MaterialTheme.typography.headlineSmall)
                    Text("ToF 强档 + RGB · 当前为画面验证，语音和震动关闭。实验结果不表示道路可通行。",
                        style=MaterialTheme.typography.bodyMedium)
                    val fresh=active && view.receiptNs>0 && now-view.receiptNs in 0..750_000_000L
                    Card(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(16.dp),verticalArrangement=Arrangement.spacedBy(8.dp)) {
                            Text(if(!fresh || !view.ready) "等待有效结果" else if(view.head || view.body) "检测区内有障碍提示" else "当前未触发提示",
                                style=MaterialTheme.typography.titleLarge,modifier=Modifier.testTag("fusion_result"))
                            Text("头部：${if(fresh && view.ready) if(view.head) "提示" else "未触发" else "未知"}    身体：${if(fresh && view.ready) if(view.body) "提示" else "未触发" else "未知"}")
                            Text(if(active && view.receiptNs>0 && !fresh) "数据已过期，等待新帧" else view.message,
                                modifier=Modifier.testTag("fusion_status"))
                        }
                    }
                    if(!active) Button(onClick=::startExperiment,enabled=!leaseBusy,
                        modifier=Modifier.fillMaxWidth().heightIn(min=56.dp).testTag("fusion_start")) {
                        Text(if(leaseBusy) "正在释放上一实验…" else "开始实验")
                    } else OutlinedButton(onClick=::stopExperiment,modifier=Modifier.fillMaxWidth().heightIn(min=56.dp).testTag("fusion_stop")) {
                        Text("停止实验")
                    }
                    OutlinedButton(onClick={session?.calibrateHeight()},enabled=active,
                        modifier=Modifier.fillMaxWidth().heightIn(min=48.dp).testTag("fusion_calibrate")) { Text("低头看空地，标定安装高度") }
                    Text(view.heightSource)
                    if(fresh) view.image?.let { Image(it.asImageBitmap(),"去畸变相机画面",Modifier.fillMaxWidth().heightIn(max=320.dp)) }
                    Text("诊断：ToF ${view.tofHead}/${view.tofBody} · RGB ${view.rgbHead}/${view.rgbBody}\n"+
                        "尺度来源 ${view.anchor} · 接收到结果 ${view.latencyMs} ms\n处理 ${view.processed} 帧 · 跳过 ${view.skipped} 帧",
                        style=MaterialTheme.typography.bodySmall)
                    Text("离开本页即停止。新录制的正式评价仍按冻结脚本进行。",style=MaterialTheme.typography.bodySmall)
                    TextButton(onClick={finish()},modifier=Modifier.fillMaxWidth().heightIn(min=48.dp)) { Text("返回") }
                }
            }
        } }
        lifecycleScope.launch { while(true) {
            now=SystemClock.elapsedRealtimeNanos()
            // Inactive UI no longer reads `now` because the freshness condition short-circuits.
            // Observe lease changes explicitly so asynchronous cleanup re-enables the button.
            leaseBusy=MobileFusionLease.busy
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
        active=true; view=MobileFusionView()
        session=MobileFusionSession(application,root,atom,tof) { value ->
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
        view=MobileFusionView(message="已停止，连接与推理正在释放")
    }

    override fun onStop() { stopExperiment(); super.onStop() }
}
