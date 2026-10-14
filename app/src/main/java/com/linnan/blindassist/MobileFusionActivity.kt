package com.linnan.blindassist

import android.os.Bundle
import android.os.SystemClock
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.Crossfade
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.Height
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material.icons.rounded.Vibration
import androidx.compose.material.icons.rounded.VolumeUp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.atomic.AtomicLong

private val FxNight = Color(0xFF0A1115)
private val FxPanel = Color(0xFF121D22)
private val FxPanelRaised = Color(0xFF1A2A31)
private val FxHairline = Color(0xFF2A3B42)
private val FxText = Color(0xFFF2EEE6)
private val FxMuted = Color(0xFF9DB0B7)
private val FxMint = Color(0xFF7FE0B2)
private val FxSky = Color(0xFF86C3FF)
private val FxAmber = Color(0xFFFFC857)
private val FxDanger = Color(0xFFFF5C72)
private val FxIdle = Color(0xFF5E7079)

private enum class Verdict { WAITING, CLEAR, BODY, HEAD, BOTH }

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
        active=true; view=MobileFusionView(demoRules=demoRules); lastSpoken=null
        session=MobileFusionSession(application,root,atom,tof,demoRules) { value ->
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
        announcer?.silence()
        view=MobileFusionView(message="已停止，连接与推理正在释放",demoRules=demoRules)
    }

    override fun onStop() { stopExperiment(); super.onStop() }
    override fun onDestroy() { announcer?.shutdown(); announcer=null; super.onDestroy() }
}

@Composable
private fun FusionScreen(
    view:MobileFusionView, active:Boolean, now:Long, leaseBusy:Boolean,
    speech:Boolean, vibration:Boolean, demoRules:Boolean, lastSpoken:String?,
    onStart:()->Unit, onStop:()->Unit, onCalibrate:()->Unit,
    onSpeech:(Boolean)->Unit, onVibration:(Boolean)->Unit, onDemoRules:(Boolean)->Unit, onBack:()->Unit,
) {
    val fresh=active && view.receiptNs>0 && now-view.receiptNs in 0..750_000_000L
    val verdict=when {
        !fresh || !view.ready -> Verdict.WAITING
        view.head && view.body -> Verdict.BOTH
        view.head -> Verdict.HEAD
        view.body -> Verdict.BODY
        else -> Verdict.CLEAR
    }
    var showDetails by rememberSaveable { mutableStateOf(false) }
    Column(
        Modifier.fillMaxSize().background(Brush.verticalGradient(listOf(FxNight,Color(0xFF0D171C),FxNight)))
            .systemBarsPadding().verticalScroll(rememberScrollState()).padding(horizontal=18.dp,vertical=12.dp),
        verticalArrangement=Arrangement.spacedBy(14.dp)
    ) {
        Header(active=active,fresh=fresh,onBack=onBack)
        VerdictCard(view,verdict,fresh,active)
        PrimaryControl(active,leaseBusy,onStart,onStop)
        CameraPanel(view,fresh,verdict)
        AnimatedVisibility(visible=lastSpoken!=null && active,enter=fadeIn()+expandVertically(),exit=fadeOut()+shrinkVertically()) {
            Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(FxPanel).padding(horizontal=14.dp,vertical=10.dp),
                verticalAlignment=Alignment.CenterVertically) {
                Icon(Icons.Rounded.VolumeUp,null,tint=FxSky,modifier=Modifier.size(18.dp))
                Spacer(Modifier.width(10.dp))
                Text("最近播报：${lastSpoken ?: ""}",color=FxMuted,fontSize=13.sp)
            }
        }
        SettingsPanel(active,view,speech,vibration,demoRules,onCalibrate,onSpeech,onVibration,onDemoRules)
        DetailsPanel(view,fresh,active,showDetails) { showDetails=!showDetails }
        Text("研究原型演示：提示只表示头部 / 身体检测区内有障碍响应，不表示道路可通行。演示规则（外参俯仰修正与走/停距离）来自开发数据，尚未经新冻结检验。",
            color=FxMuted.copy(alpha=.75f),fontSize=11.sp,lineHeight=16.sp)
        Spacer(Modifier.height(8.dp))
    }
}

@Composable
private fun Header(active:Boolean, fresh:Boolean, onBack:()->Unit) {
    Row(Modifier.fillMaxWidth(),verticalAlignment=Alignment.CenterVertically) {
        IconButton(onClick=onBack) { Icon(Icons.AutoMirrored.Rounded.ArrowBack,"返回",tint=FxText) }
        Column(Modifier.weight(1f)) {
            Text("融合演示",color=FxText,fontSize=22.sp,fontWeight=FontWeight.Bold)
            Text("RGB 深度 + 8×8 ToF · 头部 / 身体两区",color=FxMuted,fontSize=12.sp)
        }
        val (label,color)=when {
            !active -> "未开始" to FxIdle
            fresh -> "实时" to FxMint
            else -> "连接中" to FxAmber
        }
        StatusChip(label,color,pulse=active)
    }
}

@Composable
private fun StatusChip(label:String, color:Color, pulse:Boolean) {
    val transition=rememberInfiniteTransition(label="chip")
    val alpha by transition.animateFloat(.35f,1f,infiniteRepeatable(tween(900,easing=FastOutSlowInEasing),RepeatMode.Reverse),label="chip-alpha")
    val animated by animateColorAsState(color,tween(300),label="chip-color")
    Row(Modifier.clip(CircleShape).background(animated.copy(alpha=.14f)).padding(horizontal=12.dp,vertical=6.dp),
        verticalAlignment=Alignment.CenterVertically) {
        Box(Modifier.size(8.dp).graphicsLayer { this.alpha=if(pulse) alpha else 1f }.clip(CircleShape).background(animated))
        Spacer(Modifier.width(7.dp))
        Text(label,color=animated,fontSize=13.sp,fontWeight=FontWeight.SemiBold)
    }
}

@Composable
private fun VerdictCard(view:MobileFusionView, verdict:Verdict, fresh:Boolean, active:Boolean) {
    val accent by animateColorAsState(when(verdict) {
        Verdict.WAITING -> FxIdle; Verdict.CLEAR -> FxMint; Verdict.BODY -> FxAmber
        Verdict.HEAD, Verdict.BOTH -> FxDanger
    },tween(260),label="verdict-accent")
    val alarm=verdict==Verdict.HEAD || verdict==Verdict.BODY || verdict==Verdict.BOTH
    val transition=rememberInfiniteTransition(label="alarm")
    val pulse by transition.animateFloat(0f,1f,infiniteRepeatable(tween(1100,easing=FastOutSlowInEasing),RepeatMode.Restart),label="alarm-pulse")
    val headline=when(verdict) {
        Verdict.WAITING -> if(active) "等待有效结果" else "点击下方开始演示"
        Verdict.CLEAR -> "检测区未触发"
        Verdict.BODY -> "前方有障碍"
        Verdict.HEAD -> "注意头部前方"
        Verdict.BOTH -> "前方障碍 · 注意头部"
    }
    val distance=view.nearestM.takeIf { alarm && it.isFinite() }
    val shape=RoundedCornerShape(26.dp)
    Box(
        Modifier.fillMaxWidth().clip(shape)
            .background(Brush.linearGradient(listOf(lerp(FxPanel,accent,.22f),FxPanel)))
            .border(BorderStroke(1.dp,accent.copy(alpha=.45f)),shape)
            .semantics(mergeDescendants=true) { liveRegion=LiveRegionMode.Polite }
            .padding(20.dp)
    ) {
        Row(verticalAlignment=Alignment.CenterVertically) {
            Box(Modifier.size(64.dp),contentAlignment=Alignment.Center) {
                if(alarm) Canvas(Modifier.fillMaxSize()) {
                    drawCircle(accent.copy(alpha=(1f-pulse)*.55f),radius=size.minDimension/2*(.55f+.45f*pulse))
                }
                Box(Modifier.size(30.dp).clip(CircleShape).background(accent))
            }
            Spacer(Modifier.width(16.dp))
            Column(Modifier.weight(1f)) {
                AnimatedContent(headline,transitionSpec={
                    (fadeIn(tween(200))).togetherWith(fadeOut(tween(120)))
                },label="headline") { text ->
                    Text(text,color=FxText,fontSize=26.sp,lineHeight=32.sp,fontWeight=FontWeight.Bold,
                        modifier=Modifier.testTag("fusion_result"))
                }
                Spacer(Modifier.height(4.dp))
                AnimatedContent(distance?.let { "约 %.1f 米".format(it) } ?: "",label="distance") { text ->
                    if(text.isNotEmpty()) Text(text,color=accent,fontSize=20.sp,fontWeight=FontWeight.SemiBold)
                }
                Text(if(active && view.receiptNs>0 && !fresh) "数据已过期，等待新帧" else view.message,
                    color=FxMuted,fontSize=13.sp,modifier=Modifier.testTag("fusion_status"))
            }
        }
    }
    Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(10.dp)) {
        val known=fresh && view.ready
        ZonePill("头部区",known,view.head,view.tofHead,view.rgbHead,view.headDistanceM,Modifier.weight(1f))
        ZonePill("身体区",known,view.body,view.tofBody,view.rgbBody,view.bodyDistanceM,Modifier.weight(1f))
    }
    AnimatedVisibility(visible=fresh && view.demoRules && view.limitM.isFinite(),enter=fadeIn()+expandVertically(),exit=fadeOut()+shrinkVertically()) {
        Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(FxPanel).padding(horizontal=14.dp,vertical=10.dp),
            verticalAlignment=Alignment.CenterVertically) {
            val walkColor by animateColorAsState(if(view.walking) FxSky else FxMuted,label="walk")
            Text(if(view.walking) "行走中" else "站立",color=walkColor,fontWeight=FontWeight.Bold,fontSize=14.sp)
            Spacer(Modifier.width(10.dp))
            Text("提醒距离 %.1f m".format(view.limitM),color=FxMuted,fontSize=13.sp)
            Spacer(Modifier.weight(1f))
            Text(if(view.walking) "按 2 秒步态判断" else "站着只报 0.8 m 内",color=FxMuted.copy(alpha=.7f),fontSize=12.sp)
        }
    }
}

@Composable
private fun ZonePill(label:String, known:Boolean, alarm:Boolean, tof:Boolean, rgb:Boolean, distanceM:Double, modifier:Modifier) {
    val color by animateColorAsState(when { !known -> FxIdle; alarm -> FxDanger; else -> FxMint },tween(240),label="pill-$label")
    Column(modifier.clip(RoundedCornerShape(18.dp)).background(FxPanel)
        .border(1.dp,color.copy(alpha=.35f),RoundedCornerShape(18.dp)).padding(14.dp)
        .semantics(mergeDescendants=true) {}) {
        Row(verticalAlignment=Alignment.CenterVertically) {
            Box(Modifier.size(10.dp).clip(CircleShape).background(color))
            Spacer(Modifier.width(8.dp))
            Text(label,color=FxText,fontWeight=FontWeight.Bold,fontSize=15.sp)
            Spacer(Modifier.weight(1f))
            Text(when { !known -> "未知"; alarm -> "提示"; else -> "未触发" },color=color,fontSize=13.sp,fontWeight=FontWeight.SemiBold)
        }
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement=Arrangement.spacedBy(6.dp),verticalAlignment=Alignment.CenterVertically) {
            SourceTag("ToF",known && tof)
            SourceTag("RGB",known && rgb)
            if(known && distanceM.isFinite()) { Spacer(Modifier.weight(1f)); Text("%.1f m".format(distanceM),color=FxMuted,fontSize=12.sp) }
        }
    }
}

@Composable
private fun SourceTag(label:String, on:Boolean) {
    val background by animateColorAsState(if(on) FxSky.copy(alpha=.22f) else FxPanelRaised,label="tag-bg-$label")
    val foreground by animateColorAsState(if(on) FxSky else FxMuted.copy(alpha=.6f),label="tag-fg-$label")
    Text(label,color=foreground,fontSize=11.sp,fontWeight=FontWeight.Bold,
        modifier=Modifier.clip(RoundedCornerShape(6.dp)).background(background).padding(horizontal=7.dp,vertical=3.dp))
}

@Composable
private fun CameraPanel(view:MobileFusionView, fresh:Boolean, verdict:Verdict) {
    val shape=RoundedCornerShape(22.dp)
    Box(Modifier.fillMaxWidth().aspectRatio(3f/4f).clip(shape).background(FxPanel).border(1.dp,FxHairline,shape)) {
        val image=if(fresh) view.image else null
        Crossfade(image!=null,animationSpec=tween(300),label="camera") { has ->
            if(has && image!=null) {
                Box(Modifier.fillMaxSize()) {
                    Image(image.asImageBitmap(),"去畸变相机画面",Modifier.fillMaxSize(),contentScale=ContentScale.FillBounds)
                    BoxOverlay(view,verdict,image.width,image.height)
                }
            } else Box(Modifier.fillMaxSize(),contentAlignment=Alignment.Center) {
                Text(if(fresh) "画面暂不可用，仅 ToF" else "相机画面将在这里显示",color=FxMuted,fontSize=14.sp)
            }
        }
        view.tofGrid?.takeIf { fresh }?.let { grid ->
            TofHeatmap(grid,Modifier.align(Alignment.BottomEnd).padding(12.dp).size(108.dp))
        }
        Row(Modifier.align(Alignment.TopStart).padding(12.dp),horizontalArrangement=Arrangement.spacedBy(6.dp)) {
            LegendDot("头部区",FxSky); LegendDot("身体区",FxAmber)
        }
    }
}

@Composable
private fun LegendDot(label:String, color:Color) {
    Row(Modifier.clip(CircleShape).background(Color.Black.copy(alpha=.45f)).padding(horizontal=9.dp,vertical=4.dp),
        verticalAlignment=Alignment.CenterVertically) {
        Box(Modifier.size(7.dp).clip(CircleShape).background(color))
        Spacer(Modifier.width(5.dp))
        Text(label,color=Color.White,fontSize=11.sp)
    }
}

@Composable
private fun BoxOverlay(view:MobileFusionView, verdict:Verdict, imageWidth:Int, imageHeight:Int) {
    val headColor by animateColorAsState(if(view.head) FxDanger else FxSky,tween(200),label="head-box")
    val bodyColor by animateColorAsState(if(view.body) FxDanger else FxAmber,tween(200),label="body-box")
    val headFill by animateFloatAsStateCompat(if(view.head) .22f else 0f)
    val bodyFill by animateFloatAsStateCompat(if(view.body) .22f else 0f)
    Canvas(Modifier.fillMaxSize()) {
        if(view.boxes.size!=4) return@Canvas
        val sx=size.width/imageWidth; val sy=size.height/imageHeight
        fun quad(points:List<Pair<Float,Float>?>):Path? {
            if(points.any { it==null }) return null
            return Path().apply {
                points.forEachIndexed { i,p -> val o=Offset(p!!.first*sx,p.second*sy); if(i==0) moveTo(o.x,o.y) else lineTo(o.x,o.y) }
                close()
            }
        }
        listOf(Triple(0,headColor,headFill),Triple(2,bodyColor,bodyFill)).forEach { (start,color,fill) ->
            val near=quad(view.boxes[start]); val far=quad(view.boxes[start+1])
            far?.let { drawPath(it,color.copy(alpha=.55f),style=Stroke(2.dp.toPx())) }
            near?.let {
                if(fill>0f) drawPath(it,color.copy(alpha=fill))
                drawPath(it,color,style=Stroke(3.dp.toPx()))
            }
            // Edges between the 1 m and 2 m faces give the box its depth.
            val a=view.boxes[start]; val b=view.boxes[start+1]
            for(i in 0..3) { val p=a[i]; val q=b[i]
                if(p!=null && q!=null) drawLine(color.copy(alpha=.4f),Offset(p.first*sx,p.second*sy),Offset(q.first*sx,q.second*sy),1.5.dp.toPx())
            }
        }
    }
}

@Composable
private fun animateFloatAsStateCompat(target:Float) =
    androidx.compose.animation.core.animateFloatAsState(target,tween(220),label="fill")

@Composable
private fun TofHeatmap(grid:FloatArray, modifier:Modifier) {
    Column(modifier.clip(RoundedCornerShape(12.dp)).background(Color.Black.copy(alpha=.55f)).padding(6.dp)) {
        Text("ToF 8×8",color=Color.White,fontSize=10.sp,fontWeight=FontWeight.Bold)
        Spacer(Modifier.height(4.dp))
        Canvas(Modifier.fillMaxSize()) {
            val cell=size.minDimension/8
            for(i in 0 until 64) {
                val d=grid[i]
                val color=if(d.isNaN()) Color(0xFF2B3940) else {
                    val t=((d-.3f)/2.7f).coerceIn(0f,1f)   // 0.3 m red -> 3 m blue
                    if(t<.5f) lerp(FxDanger,FxAmber,t*2) else lerp(FxAmber,FxSky,(t-.5f)*2)
                }
                drawRect(color,Offset((i%8)*cell+.75f,(i/8)*cell+.75f),Size(cell-1.5f,cell-1.5f))
            }
        }
    }
}

@Composable
private fun PrimaryControl(active:Boolean, leaseBusy:Boolean, onStart:()->Unit, onStop:()->Unit) {
    val color by animateColorAsState(if(active) FxPanelRaised else FxMint,tween(240),label="primary")
    val content by animateColorAsState(if(active) FxText else FxNight,tween(240),label="primary-content")
    Button(onClick=if(active) onStop else onStart,enabled=active || !leaseBusy,
        colors=ButtonDefaults.buttonColors(containerColor=color,contentColor=content,
            disabledContainerColor=FxPanelRaised,disabledContentColor=FxMuted),
        shape=RoundedCornerShape(20.dp),
        modifier=Modifier.fillMaxWidth().heightIn(min=60.dp).testTag(if(active) "fusion_stop" else "fusion_start")) {
        AnimatedContent(active,label="primary-label") { running ->
            Row(verticalAlignment=Alignment.CenterVertically) {
                Icon(if(running) Icons.Rounded.Stop else Icons.Rounded.PlayArrow,null)
                Spacer(Modifier.width(8.dp))
                Text(when { running -> "停止演示"; leaseBusy -> "正在释放上一实验…"; else -> "开始演示" },
                    fontSize=18.sp,fontWeight=FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun SettingsPanel(active:Boolean, view:MobileFusionView, speech:Boolean, vibration:Boolean, demoRules:Boolean,
                          onCalibrate:()->Unit, onSpeech:(Boolean)->Unit, onVibration:(Boolean)->Unit, onDemoRules:(Boolean)->Unit) {
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(FxPanel).padding(vertical=6.dp)) {
        ToggleRow(Icons.Rounded.VolumeUp,"语音提醒","首次立即播报，持续时每 6 秒重复",speech,onSpeech,true)
        ToggleRow(Icons.Rounded.Vibration,"震动提醒","头部区三连短震，身体区一次长震",vibration,onVibration,true)
        ToggleRow(Icons.Rounded.Height,"演示规则","外参俯仰修正 + 走/停提醒距离；关闭即原冻结 v2，停止后可切换",
            demoRules,onDemoRules,!active)
        HorizontalDivider(color=FxHairline,modifier=Modifier.padding(horizontal=16.dp))
        Row(Modifier.fillMaxWidth().clickable(enabled=active,role=Role.Button,onClick=onCalibrate)
                .testTag("fusion_calibrate").heightIn(min=56.dp).padding(horizontal=16.dp,vertical=10.dp),
            verticalAlignment=Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("低头看空地，标定安装高度",color=if(active) FxText else FxMuted,fontSize=15.sp,fontWeight=FontWeight.SemiBold)
                Text(view.heightSource,color=FxMuted,fontSize=12.sp)
            }
            Text("标定",color=if(active) FxSky else FxMuted,fontWeight=FontWeight.Bold)
        }
    }
}

@Composable
private fun ToggleRow(icon:ImageVector, title:String, body:String, checked:Boolean, onChange:(Boolean)->Unit, enabled:Boolean) {
    Row(Modifier.fillMaxWidth().heightIn(min=60.dp).clickable(enabled=enabled,role=Role.Switch) { onChange(!checked) }
            .padding(horizontal=16.dp,vertical=8.dp),
        verticalAlignment=Alignment.CenterVertically) {
        Icon(icon,null,tint=if(checked) FxSky else FxMuted,modifier=Modifier.size(22.dp))
        Spacer(Modifier.width(14.dp))
        Column(Modifier.weight(1f)) {
            Text(title,color=if(enabled) FxText else FxMuted,fontSize=15.sp,fontWeight=FontWeight.SemiBold)
            Text(body,color=FxMuted,fontSize=12.sp,lineHeight=16.sp)
        }
        Switch(checked,onCheckedChange=null,enabled=enabled,colors=SwitchDefaults.colors(
            checkedTrackColor=FxMint,checkedThumbColor=FxNight,uncheckedTrackColor=FxPanelRaised,
            uncheckedThumbColor=FxMuted,uncheckedBorderColor=FxHairline))
    }
}

@Composable
private fun DetailsPanel(view:MobileFusionView, fresh:Boolean, active:Boolean, expanded:Boolean, onToggle:()->Unit) {
    val rotation by androidx.compose.animation.core.animateFloatAsState(if(expanded) 180f else 0f,tween(220),label="chevron")
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(FxPanel)) {
        Row(Modifier.fillMaxWidth().clickable(role=Role.Button,onClick=onToggle).heightIn(min=52.dp).padding(horizontal=16.dp)
                .semantics { contentDescription=if(expanded) "收起开发者详情" else "展开开发者详情" },
            verticalAlignment=Alignment.CenterVertically) {
            Text("开发者详情",color=FxText,fontSize=15.sp,fontWeight=FontWeight.SemiBold,modifier=Modifier.weight(1f))
            Icon(Icons.Rounded.ExpandMore,null,tint=FxMuted,modifier=Modifier.rotate(rotation))
        }
        AnimatedVisibility(expanded,enter=fadeIn()+expandVertically(),exit=fadeOut()+shrinkVertically()) {
            fun m(v:Double)=if(v.isFinite()) "%.2f m".format(v) else "—"
            Text(
                "原始：ToF ${view.tofHead}/${view.tofBody} · RGB ${view.rgbHead}/${view.rgbBody}（头/身，门控后）\n"+
                "距离：头 ${m(view.headDistanceM)} · 身 ${m(view.bodyDistanceM)} · 上限 ${m(view.limitM)}\n"+
                "尺度来源 ${view.anchor} · 姿态俯仰 ${if(view.pitchDeg.isFinite()) "%.1f°".format(view.pitchDeg) else "—"}"+
                " · 外参俯仰 ${if(view.extrinsicPitchDeg.isFinite()) "%.1f°".format(view.extrinsicPitchDeg) else "—"}\n"+
                "接收到结果 ${view.latencyMs} ms · 处理 ${view.processed} 帧 · 跳过 ${view.skipped} 帧\n"+
                "规则：${if(view.demoRules) "演示（俯仰 0° + v7w 走/停）" else "冻结 v2"} · ${if(active && !fresh) "等待新帧" else "两帧确认（最近 3 帧中 2 帧）"}\n"+
                "离开本页即停止。新录制的正式评价仍按冻结脚本进行。",
                color=FxMuted,fontSize=12.sp,lineHeight=18.sp,modifier=Modifier.padding(start=16.dp,end=16.dp,bottom=14.dp))
        }
    }
}
