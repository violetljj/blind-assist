package com.linnan.blindassist

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.Crossfade
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.automirrored.rounded.DirectionsWalk
import androidx.compose.material.icons.automirrored.rounded.VolumeOff
import androidx.compose.material.icons.automirrored.rounded.VolumeUp
import androidx.compose.material.icons.rounded.Accessibility
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.Straighten
import androidx.compose.material.icons.rounded.Vibration
import androidx.compose.material.icons.rounded.MobileOff
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.withTransform
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
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// Night palette: one cool neutral ramp plus three state colours (clear / body / head).
internal val FxNight = Color(0xFF070B0E)
private val FxSurface = Color(0xFF10171B)
private val FxSurfaceHigh = Color(0xFF172126)
private val FxLine = Color(0xFF243138)
private val FxText = Color(0xFFF3F1EC)
private val FxMuted = Color(0xFF8FA1A8)
private val FxFaint = Color(0xFF55666D)
private val FxClear = Color(0xFF6EE7B7)
private val FxBody = Color(0xFFFFC14D)
private val FxHead = Color(0xFFFF5E73)
private val FxInfo = Color(0xFF7CC4FF)

private enum class Verdict { WAITING, CLEAR, BODY, HEAD, BOTH }

private val Tabular = TextStyle(fontFeatureSettings = "tnum")

@Composable
internal fun FusionScreen(
    view:MobileFusionView, active:Boolean, fresh:Boolean, shownHead:Boolean, shownBody:Boolean, leaseBusy:Boolean,
    preview:()->androidx.compose.ui.graphics.ImageBitmap?, modelsReady:Boolean, thermal:Int, audioWarning:String?,
    speech:Boolean, vibration:Boolean, demoRules:Boolean, lastSpoken:String?,
    onStart:()->Unit, onStop:()->Unit, onCalibrate:()->Unit,
    onSpeech:(Boolean)->Unit, onVibration:(Boolean)->Unit, onDemoRules:(Boolean)->Unit, onRaiseVolume:()->Unit, onBack:()->Unit,
) {
    // Freshness and the short alarm hold are computed in the Activity loop (display smoothing only).
    val held=view.copy(head=shownHead,body=shownBody)
    val verdict=when {
        !fresh || !view.ready -> Verdict.WAITING
        held.head && held.body -> Verdict.BOTH
        held.head -> Verdict.HEAD
        held.body -> Verdict.BODY
        else -> Verdict.CLEAR
    }
    val accent by animateColorAsState(verdict.color(),tween(280),label="accent")
    Column(Modifier.fillMaxSize().background(FxNight).verticalScroll(rememberScrollState())) {
        Viewfinder(held,preview,fresh,active,verdict,accent,onBack)
        Column(Modifier.padding(horizontal=18.dp).padding(top=18.dp,bottom=12.dp),verticalArrangement=Arrangement.spacedBy(14.dp)) {
            ControlDock(active,leaseBusy,speech,vibration,onStart,onStop,onSpeech,onVibration)
            AnimatedVisibility(audioWarning!=null,enter=fadeIn()+expandVertically(),exit=fadeOut()+shrinkVertically()) {
                NoticeRow(audioWarning ?: "",if(audioWarning?.startsWith("媒体音量")==true) "调高音量" else null,onRaiseVolume)
            }
            AnimatedVisibility(!active && !modelsReady,enter=fadeIn(),exit=fadeOut()) {
                NoticeRow("正在后台预加载模型，完成后开始会更快",null) {}
            }
            AnimatedVisibility(thermal>=3,enter=fadeIn()+expandVertically(),exit=fadeOut()+shrinkVertically()) {
                NoticeRow("手机较热，画面已自动降帧；判决与提醒不受影响",null) {}
            }
            ZoneFigureCard(held,fresh && view.ready)
            AnimatedVisibility(active,enter=fadeIn()+expandVertically(),exit=fadeOut()+shrinkVertically()) {
                Column(verticalArrangement=Arrangement.spacedBy(14.dp)) {
                    MotionStrip(view,fresh && view.demoRules && view.limitM.isFinite())
                    SpokenLine(lastSpoken)
                }
            }
            RulesCard(view,active,demoRules,onDemoRules,onCalibrate)
            DetailsCard(view,fresh,active)
            Text("研究原型演示：提示只表示头部 / 身体检测区内有障碍响应，不表示道路可通行。演示规则来自开发数据，尚未经新冻结检验。",
                color=FxFaint,fontSize=11.sp,lineHeight=16.sp,modifier=Modifier.padding(top=4.dp))
            Spacer(Modifier.navigationBarsPadding())
        }
    }
}

private fun Verdict.color()=when(this) {
    Verdict.WAITING -> FxFaint; Verdict.CLEAR -> FxClear; Verdict.BODY -> FxBody; Verdict.HEAD, Verdict.BOTH -> FxHead
}

// ---------------------------------------------------------------- viewfinder

@Composable
private fun Viewfinder(view:MobileFusionView, preview:()->androidx.compose.ui.graphics.ImageBitmap?, fresh:Boolean, active:Boolean, verdict:Verdict, accent:Color, onBack:()->Unit) {
    val alarm=verdict==Verdict.BODY || verdict==Verdict.HEAD || verdict==Verdict.BOTH
    val glow by rememberInfiniteTransition(label="glow").animateFloat(.35f,1f,
        infiniteRepeatable(tween(900,easing=FastOutSlowInEasing),RepeatMode.Reverse),label="glow-alpha")
    val glowAlpha by animateFloatAsState(if(alarm) 1f else 0f,tween(260),label="glow-on")
    val shape=RoundedCornerShape(bottomStart=32.dp,bottomEnd=32.dp)
    Box(Modifier.fillMaxWidth().aspectRatio(3f/4f).clip(shape).background(Color(0xFF0B1216))) {
        // Availability changes composition; each new frame only invalidates the Canvas draw.
        val available by remember(preview) { derivedStateOf { preview()!=null } }
        Crossfade(available,animationSpec=tween(350),label="feed") { has ->
            if(has) Box(Modifier.fillMaxSize()) {
                Canvas(Modifier.fillMaxSize().semantics { contentDescription="去畸变相机画面" }) {
                    val frame=preview() ?: return@Canvas
                    // Raw Atom frames are landscape; the calibrated upright view is their transpose
                    // (rotate 90° plus mirror). Scale to cover the 3:4 viewfinder.
                    val upright=frame.width>frame.height
                    val w=if(upright) frame.height else frame.width; val h=if(upright) frame.width else frame.height
                    val fit=maxOf(size.width/w,size.height/h)
                    withTransform({
                        translate(size.width/2,size.height/2)
                        scale(fit,fit,pivot=Offset.Zero)
                        if(upright) { scale(-1f,1f,pivot=Offset.Zero); rotate(90f,pivot=Offset.Zero) }
                    }) { drawImage(frame,topLeft=Offset(-frame.width/2f,-frame.height/2f)) }
                }
                // Boxes are projected in the undistorted 480×640 view.
                if(fresh) BoxOverlay(view,480,640)
            } else IdleRadar(active)
        }
        // Alarm vignette: the whole frame warms up instead of a small badge changing colour.
        Box(Modifier.fillMaxSize().graphicsLayer { alpha=glowAlpha*(.55f+.45f*glow) }
            .background(Brush.radialGradient(listOf(Color.Transparent,accent.copy(alpha=.42f)),radius=1400f)))
        Box(Modifier.fillMaxWidth().height(150.dp).background(Brush.verticalGradient(listOf(Color.Black.copy(alpha=.72f),Color.Transparent))))
        Box(Modifier.align(Alignment.BottomCenter).fillMaxWidth().height(230.dp)
            .background(Brush.verticalGradient(listOf(Color.Transparent,FxNight.copy(alpha=.92f)))))
        TopBar(active,fresh,onBack,Modifier.align(Alignment.TopStart))
        if(active) TofHeatmap(view.tofGrid?.takeIf { fresh },Modifier.align(Alignment.TopEnd).statusBarsPadding().padding(top=64.dp,end=16.dp).width(86.dp))
        VerdictOverlay(view,verdict,fresh,active,accent,Modifier.align(Alignment.BottomStart))
        if(glowAlpha>0f) Canvas(Modifier.fillMaxSize()) {
            drawRoundRect(accent.copy(alpha=glowAlpha*glow),cornerRadius=CornerRadius(32.dp.toPx()),style=Stroke(4.dp.toPx()))
        }
    }
}

@Composable
private fun TopBar(active:Boolean, fresh:Boolean, onBack:()->Unit, modifier:Modifier) {
    Row(modifier.fillMaxWidth().statusBarsPadding().padding(horizontal=8.dp,vertical=6.dp),verticalAlignment=Alignment.CenterVertically) {
        IconButton(onClick=onBack) { Icon(Icons.AutoMirrored.Rounded.ArrowBack,"返回",tint=Color.White) }
        Column(Modifier.weight(1f)) {
            Text("融合演示",color=Color.White,fontSize=19.sp,fontWeight=FontWeight.Bold)
            Text("RGB 深度 × 8×8 ToF",color=Color.White.copy(alpha=.62f),fontSize=12.sp)
        }
        val (label,color)=when { !active -> "待机" to FxMuted; fresh -> "实时" to FxClear; else -> "连接中" to FxBody }
        LivePill(label,color,active)
        Spacer(Modifier.width(8.dp))
    }
}

@Composable
private fun LivePill(label:String, color:Color, pulse:Boolean) {
    val a by rememberInfiniteTransition(label="pill").animateFloat(.3f,1f,
        infiniteRepeatable(tween(800,easing=FastOutSlowInEasing),RepeatMode.Reverse),label="pill-a")
    val c by animateColorAsState(color,tween(300),label="pill-c")
    Row(Modifier.clip(CircleShape).background(Color.Black.copy(alpha=.45f)).border(1.dp,c.copy(alpha=.35f),CircleShape)
        .padding(horizontal=11.dp,vertical=6.dp),verticalAlignment=Alignment.CenterVertically) {
        Box(Modifier.size(7.dp).graphicsLayer { alpha=if(pulse) a else 1f }.clip(CircleShape).background(c))
        Spacer(Modifier.width(6.dp))
        Text(label,color=c,fontSize=12.sp,fontWeight=FontWeight.SemiBold)
    }
}

@Composable
private fun VerdictOverlay(view:MobileFusionView, verdict:Verdict, fresh:Boolean, active:Boolean, accent:Color, modifier:Modifier) {
    val alarm=verdict!=Verdict.WAITING && verdict!=Verdict.CLEAR
    val headline=when(verdict) {
        Verdict.WAITING -> if(active) "等待有效结果" else "准备就绪"
        Verdict.CLEAR -> "检测区未触发"
        Verdict.BODY -> "前方有障碍"
        Verdict.HEAD -> "注意头部前方"
        Verdict.BOTH -> "头部与身体区障碍"
    }
    val distance=view.nearestM.takeIf { alarm && it.isFinite() }
    Row(modifier.fillMaxWidth().padding(horizontal=22.dp,vertical=22.dp).semantics(mergeDescendants=true) { liveRegion=LiveRegionMode.Polite },
        verticalAlignment=Alignment.Bottom) {
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment=Alignment.CenterVertically) {
                Box(Modifier.size(9.dp).clip(CircleShape).background(accent))
                Spacer(Modifier.width(8.dp))
                Text(when(verdict) { Verdict.WAITING -> "UNKNOWN"; Verdict.CLEAR -> "CLEAR"; else -> "ALERT" },
                    color=accent,fontSize=11.sp,fontWeight=FontWeight.Bold,letterSpacing=2.sp)
            }
            Spacer(Modifier.height(6.dp))
            AnimatedContent(headline,transitionSpec={
                (fadeIn(tween(220))+slideInVertically(tween(260)) { it/3 }).togetherWith(fadeOut(tween(120))+slideOutVertically { -it/4 })
            },label="headline") { text ->
                Text(text,color=Color.White,fontSize=28.sp,lineHeight=34.sp,fontWeight=FontWeight.Bold,maxLines=1,
                    overflow=TextOverflow.Ellipsis,modifier=Modifier.testTag("fusion_result"))
            }
            Spacer(Modifier.height(4.dp))
            Text(if(active && view.receiptNs>0 && !fresh) "数据已过期，等待新帧" else view.message,
                color=Color.White.copy(alpha=.66f),fontSize=13.sp,maxLines=1,overflow=TextOverflow.Ellipsis,
                modifier=Modifier.testTag("fusion_status"))
            // Reserved line: the pose hint never pushes the layout around.
            Text(poseHint(view)?.takeIf { fresh } ?: "",color=FxBody,fontSize=13.sp,fontWeight=FontWeight.SemiBold,maxLines=1)
        }
        // Fixed-width slot; the last distance fades out instead of collapsing the column.
        val shown=remember { DoubleArray(1) { Double.NaN } }
        if(distance!=null) shown[0]=distance
        val distanceAlpha by animateFloatAsState(if(distance!=null) 1f else 0f,tween(220),label="distance-alpha")
        Column(Modifier.width(104.dp).graphicsLayer { alpha=distanceAlpha },horizontalAlignment=Alignment.End) {
            Text(if(shown[0].isFinite()) "%.1f".format(shown[0]) else "",color=accent,fontSize=52.sp,lineHeight=54.sp,
                fontWeight=FontWeight.Bold,style=Tabular,maxLines=1)
            Text("米",color=accent.copy(alpha=.8f),fontSize=13.sp,fontWeight=FontWeight.SemiBold)
        }
    }
}

@Composable
private fun IdleRadar(active:Boolean) {
    val t by rememberInfiniteTransition(label="radar").animateFloat(0f,1f,infiniteRepeatable(tween(2400,easing=LinearEasing)),label="radar-t")
    Box(Modifier.fillMaxSize(),contentAlignment=Alignment.Center) {
        Canvas(Modifier.fillMaxSize()) {
            val c=Offset(size.width/2,size.height*.36f); val max=size.minDimension*.3f
            for(i in 1..3) drawCircle(FxLine,radius=max*i/3,center=c,style=Stroke(1.dp.toPx()))
            for(k in 0..1) {
                val p=(t+k*.5f)%1f
                drawCircle(FxInfo.copy(alpha=(1f-p)*(if(active) .5f else .22f)),radius=max*p,center=c,style=Stroke(2.dp.toPx()))
            }
            drawLine(FxLine,Offset(c.x-max,c.y),Offset(c.x+max,c.y),1.dp.toPx())
            drawLine(FxLine,Offset(c.x,c.y-max),Offset(c.x,c.y+max),1.dp.toPx())
            drawCircle(if(active) FxInfo else FxFaint,radius=5.dp.toPx(),center=c)
        }
    }
}

@Composable
private fun BoxOverlay(view:MobileFusionView, imageWidth:Int, imageHeight:Int) {
    val headColor by animateColorAsState(if(view.head) FxHead else FxInfo,tween(220),label="head-box")
    val bodyColor by animateColorAsState(if(view.body) FxHead else FxBody,tween(220),label="body-box")
    val headFill by animateFloatAsState(if(view.head) .24f else 0f,tween(220),label="head-fill")
    val bodyFill by animateFloatAsState(if(view.body) .24f else 0f,tween(220),label="body-fill")
    Canvas(Modifier.fillMaxSize()) {
        if(view.boxes.size!=4) return@Canvas
        val sx=size.width/imageWidth; val sy=size.height/imageHeight
        fun o(p:Pair<Float,Float>)=Offset(p.first*sx,p.second*sy)
        fun quad(points:List<Pair<Float,Float>?>):Path?=if(points.any { it==null }) null else Path().apply {
            points.forEachIndexed { i,p -> val q=o(p!!); if(i==0) moveTo(q.x,q.y) else lineTo(q.x,q.y) }; close()
        }
        listOf(Triple(0,headColor,headFill),Triple(2,bodyColor,bodyFill)).forEach { (start,color,fill) ->
            val near=quad(view.boxes[start]); val far=quad(view.boxes[start+1])
            far?.let { drawPath(it,color.copy(alpha=.45f),style=Stroke(1.5.dp.toPx())) }
            near?.let { if(fill>0f) drawPath(it,color.copy(alpha=fill)); drawPath(it,color,style=Stroke(2.5.dp.toPx(),cap=StrokeCap.Round)) }
            val a=view.boxes[start]; val b=view.boxes[start+1]
            for(i in 0..3) { val p=a[i]; val q=b[i]; if(p!=null && q!=null) drawLine(color.copy(alpha=.35f),o(p),o(q),1.dp.toPx()) }
        }
    }
}

@Composable
private fun TofHeatmap(grid:FloatArray?, modifier:Modifier) {
    Column(modifier.clip(RoundedCornerShape(14.dp)).background(Color.Black.copy(alpha=.5f))
        .border(1.dp,Color.White.copy(alpha=.12f),RoundedCornerShape(14.dp)).padding(7.dp)) {
        Text("ToF",color=Color.White.copy(alpha=.8f),fontSize=9.sp,fontWeight=FontWeight.Bold,letterSpacing=1.sp)
        Spacer(Modifier.height(3.dp))
        Canvas(Modifier.fillMaxWidth().aspectRatio(1f)) {
            val cell=size.minDimension/8; val gap=1.2f
            for(i in 0 until 64) {
                val d=grid?.get(i) ?: Float.NaN
                val color=if(d.isNaN()) Color.White.copy(alpha=.06f) else {
                    val t=((d-.3f)/2.7f).coerceIn(0f,1f)   // 0.3 m warm -> 3 m cool
                    if(t<.5f) lerp(FxHead,FxBody,t*2) else lerp(FxBody,FxInfo,(t-.5f)*2).copy(alpha=1f-.5f*(t-.5f))
                }
                drawRoundRect(color,Offset((i%8)*cell+gap,(i/8)*cell+gap),Size(cell-2*gap,cell-2*gap),CornerRadius(2f))
            }
        }
    }
}

private fun poseHint(view:MobileFusionView):String? {
    if(view.poseOk || !view.pitchDeg.isFinite() || !view.rollDeg.isFinite()) return null
    return when {
        kotlin.math.abs(view.rollDeg)>25 -> "设备侧倾 %.0f°，请把头摆正".format(view.rollDeg)
        view.pitchDeg>10 -> "设备上仰 %.0f°，请平视或略低头".format(view.pitchDeg)
        else -> null
    }
}

// ---------------------------------------------------------------- zones

@Composable
private fun ZoneFigureCard(view:MobileFusionView, known:Boolean) {
    val headColor by animateColorAsState(when { !known -> FxFaint; view.head -> FxHead; else -> FxClear },tween(240),label="fig-head")
    val bodyColor by animateColorAsState(when { !known -> FxFaint; view.body -> FxHead; else -> FxClear },tween(240),label="fig-body")
    val pulse by rememberInfiniteTransition(label="fig").animateFloat(.55f,1f,infiniteRepeatable(tween(700),RepeatMode.Reverse),label="fig-p")
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(24.dp)).background(FxSurface).border(1.dp,FxLine,RoundedCornerShape(24.dp))
        .padding(horizontal=18.dp,vertical=16.dp),verticalAlignment=Alignment.CenterVertically) {
        // Silhouette with the two warning bands at true proportions (eye 1.62 m on a 1.75 m figure).
        Canvas(Modifier.width(74.dp).height(150.dp)) {
            val h=size.height; val m=h/1.75f; val cx=size.width/2; val eye=.13f*m
            fun band(top:Float,bottom:Float,color:Color,alarm:Boolean) {
                val y0=(eye+top*m).coerceAtLeast(0f); val y1=eye+bottom*m
                drawRoundRect(color.copy(alpha=if(alarm) .28f*pulse+.1f else .14f),Offset(0f,y0),Size(size.width,y1-y0),CornerRadius(10f))
                drawLine(color.copy(alpha=.7f),Offset(0f,y1),Offset(size.width,y1),1.5f)
            }
            band(-.20f,.42f,headColor,known && view.head)
            band(.42f,.90f,bodyColor,known && view.body)
            val ink=Color(0xFFCBD5D9)
            drawCircle(ink,radius=.105f*m,center=Offset(cx,.12f*m))
            drawRoundRect(ink,Offset(cx-.19f*m,.27f*m),Size(.38f*m,.62f*m),CornerRadius(.09f*m))
            drawRoundRect(ink,Offset(cx-.17f*m,.86f*m),Size(.15f*m,.89f*m),CornerRadius(.06f*m))
            drawRoundRect(ink,Offset(cx+.02f*m,.86f*m),Size(.15f*m,.89f*m),CornerRadius(.06f*m))
        }
        Spacer(Modifier.width(18.dp))
        Column(Modifier.weight(1f),verticalArrangement=Arrangement.spacedBy(12.dp)) {
            ZoneRow("头部区","眼下 0.2 m 到胸口",known,view.head,view.tofHead,view.rgbHead,view.headDistanceM,headColor)
            HorizontalDivider(color=FxLine)
            ZoneRow("身体区","胸口到腰下 0.9 m",known,view.body,view.tofBody,view.rgbBody,view.bodyDistanceM,bodyColor)
        }
    }
}

@Composable
private fun ZoneRow(label:String, hint:String, known:Boolean, alarm:Boolean, tof:Boolean, rgb:Boolean, distanceM:Double, color:Color) {
    Column(Modifier.semantics(mergeDescendants=true) {}) {
        Row(verticalAlignment=Alignment.CenterVertically) {
            Text(label,color=FxText,fontSize=16.sp,fontWeight=FontWeight.Bold)
            Spacer(Modifier.width(8.dp))
            Text(when { !known -> "未知"; alarm -> "提示"; else -> "未触发" },color=color,fontSize=13.sp,fontWeight=FontWeight.SemiBold)
            Spacer(Modifier.weight(1f))
            AnimatedContent(if(known && alarm && distanceM.isFinite()) "%.1f m".format(distanceM) else "",label="zone-d-$label") { t ->
                Text(t,color=color,fontSize=15.sp,fontWeight=FontWeight.Bold,style=Tabular)
            }
        }
        Spacer(Modifier.height(6.dp))
        Row(verticalAlignment=Alignment.CenterVertically,horizontalArrangement=Arrangement.spacedBy(6.dp)) {
            SourceTag("ToF",known && tof); SourceTag("RGB",known && rgb)
            Spacer(Modifier.width(4.dp))
            Text(hint,color=FxFaint,fontSize=11.sp)
        }
    }
}

@Composable
private fun SourceTag(label:String, on:Boolean) {
    val bg by animateColorAsState(if(on) FxHead.copy(alpha=.18f) else FxSurfaceHigh,label="tag-bg-$label")
    val fg by animateColorAsState(if(on) FxHead else FxFaint,label="tag-fg-$label")
    Text(label,color=fg,fontSize=10.sp,fontWeight=FontWeight.Bold,letterSpacing=.5.sp,
        modifier=Modifier.clip(RoundedCornerShape(6.dp)).background(bg).padding(horizontal=7.dp,vertical=3.dp))
}

@Composable
private fun MotionStrip(view:MobileFusionView, known:Boolean) {
    val c by animateColorAsState(if(known && view.walking) FxInfo else FxMuted,label="walk")
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(FxSurface).padding(horizontal=16.dp,vertical=12.dp),
        verticalAlignment=Alignment.CenterVertically) {
        Icon(if(view.walking) Icons.AutoMirrored.Rounded.DirectionsWalk else Icons.Rounded.Accessibility,null,tint=c,modifier=Modifier.size(22.dp))
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(when { !known -> "步态"; view.walking -> "行走中"; else -> "站立" },color=FxText,fontSize=15.sp,fontWeight=FontWeight.SemiBold)
            Text(when { !known -> "等待 IMU 与结果"; view.walking -> "IMU 步态判断 · 提前提醒"; else -> "站着时只提醒近处障碍" },
                color=FxMuted,fontSize=12.sp,maxLines=1)
        }
        // Range bar: how far ahead the alert reaches, out of 2 m.
        val reach by animateFloatAsState(if(known) (view.limitM/2.0).toFloat().coerceIn(0f,1f) else 0f,spring(),label="reach")
        Column(horizontalAlignment=Alignment.End) {
            Text(if(known) "%.1f m".format(view.limitM) else "— m",color=c,fontSize=18.sp,fontWeight=FontWeight.Bold,style=Tabular)
            Box(Modifier.width(72.dp).height(4.dp).clip(CircleShape).background(FxSurfaceHigh)) {
                Box(Modifier.fillMaxHeight().fillMaxWidth(reach).clip(CircleShape).background(c))
            }
        }
    }
}

// ---------------------------------------------------------------- controls

@Composable
private fun ControlDock(active:Boolean, leaseBusy:Boolean, speech:Boolean, vibration:Boolean,
                        onStart:()->Unit, onStop:()->Unit,
                        onSpeech:(Boolean)->Unit, onVibration:(Boolean)->Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(28.dp)).background(FxSurface).border(1.dp,FxLine,RoundedCornerShape(28.dp))
        .padding(horizontal=10.dp,vertical=16.dp),
        horizontalArrangement=Arrangement.SpaceEvenly,verticalAlignment=Alignment.CenterVertically) {
        DockToggle(if(speech) Icons.AutoMirrored.Rounded.VolumeUp else Icons.AutoMirrored.Rounded.VolumeOff,"语音",speech,true) { onSpeech(!speech) }
        ShutterButton(active,leaseBusy,onStart,onStop)
        DockToggle(if(vibration) Icons.Rounded.Vibration else Icons.Rounded.MobileOff,"震动",vibration,true) { onVibration(!vibration) }
    }
}

@Composable
private fun ShutterButton(active:Boolean, leaseBusy:Boolean, onStart:()->Unit, onStop:()->Unit) {
    val source=remember { MutableInteractionSource() }
    val pressed by source.collectIsPressedAsState()
    val scale by animateFloatAsState(if(pressed) .92f else 1f,spring(dampingRatio=.5f),label="shutter-scale")
    val ring by animateColorAsState(if(active) FxHead else FxClear,tween(260),label="shutter-ring")
    val corner by animateFloatAsState(if(active) .28f else .5f,tween(260),label="shutter-corner")
    val enabled=active || !leaseBusy
    Column(horizontalAlignment=Alignment.CenterHorizontally) {
        Box(Modifier.size(78.dp).graphicsLayer { scaleX=scale; scaleY=scale }.clip(CircleShape)
            .border(3.dp,ring.copy(alpha=if(enabled) 1f else .3f),CircleShape)
            .clickable(source,null,enabled=enabled,role=Role.Button,onClick=if(active) onStop else onStart)
            .testTag(if(active) "fusion_stop" else "fusion_start")
            .semantics { contentDescription=if(active) "停止演示" else "开始演示" },
            contentAlignment=Alignment.Center) {
            val inner by animateFloatAsState(if(active) 30f else 58f,spring(dampingRatio=.6f),label="shutter-inner")
            Box(Modifier.size(inner.dp).drawBehind {
                drawRoundRect(ring.copy(alpha=if(enabled) 1f else .3f),cornerRadius=CornerRadius(size.minDimension*corner))
            })
        }
        Spacer(Modifier.height(6.dp))
        Text(when { active -> "停止"; leaseBusy -> "释放中…"; else -> "开始演示" },color=FxText,fontSize=12.sp,fontWeight=FontWeight.SemiBold)
    }
}

@Composable
private fun DockToggle(icon:ImageVector, label:String, on:Boolean, enabled:Boolean, modifier:Modifier=Modifier, onClick:()->Unit) {
    val bg by animateColorAsState(if(on) FxInfo.copy(alpha=.16f) else FxSurfaceHigh,label="dock-bg-$label")
    val fg by animateColorAsState(when { !enabled -> FxFaint; on -> FxInfo; else -> FxMuted },label="dock-fg-$label")
    Column(modifier.clip(RoundedCornerShape(16.dp)).clickable(enabled=enabled,role=Role.Button,onClick=onClick).padding(6.dp)
        .semantics(mergeDescendants=true) { contentDescription=label+if(on) "，已开启" else "" },
        horizontalAlignment=Alignment.CenterHorizontally) {
        Box(Modifier.size(50.dp).clip(CircleShape).background(bg),contentAlignment=Alignment.Center) {
            Icon(icon,null,tint=fg,modifier=Modifier.size(24.dp))
        }
        Spacer(Modifier.height(6.dp))
        Text(label,color=fg,fontSize=12.sp)
    }
}

@Composable
private fun NoticeRow(text:String, action:String?, onAction:()->Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(FxBody.copy(alpha=.10f))
        .border(1.dp,FxBody.copy(alpha=.25f),RoundedCornerShape(16.dp)).padding(start=14.dp,end=6.dp,top=6.dp,bottom=6.dp)
        .heightIn(min=44.dp),verticalAlignment=Alignment.CenterVertically) {
        Text(text,color=FxText,fontSize=13.sp,lineHeight=18.sp,modifier=Modifier.weight(1f))
        if(action!=null) Text(action,color=FxBody,fontSize=13.sp,fontWeight=FontWeight.Bold,
            modifier=Modifier.clip(RoundedCornerShape(12.dp)).clickable(role=Role.Button,onClick=onAction).padding(horizontal=12.dp,vertical=10.dp))
    }
}

@Composable
private fun SpokenLine(text:String?) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(FxInfo.copy(alpha=.08f)).padding(horizontal=14.dp,vertical=10.dp),
        verticalAlignment=Alignment.CenterVertically) {
        Icon(Icons.AutoMirrored.Rounded.VolumeUp,null,tint=FxInfo,modifier=Modifier.size(18.dp))
        Spacer(Modifier.width(10.dp))
        Crossfade(text,label="spoken") { t ->
            Text(if(t==null) "尚无播报 · 报警确认后会在这里显示" else "“$t”",color=if(t==null) FxFaint else FxText,fontSize=14.sp,maxLines=1,
                overflow=TextOverflow.Ellipsis)
        }
    }
}

@Composable
private fun RulesCard(view:MobileFusionView, active:Boolean, demoRules:Boolean, onDemoRules:(Boolean)->Unit, onCalibrate:()->Unit) {
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(FxSurface)) {
        Row(Modifier.fillMaxWidth().clickable(enabled=!active,role=Role.Switch) { onDemoRules(!demoRules) }
            .padding(horizontal=16.dp,vertical=14.dp),verticalAlignment=Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("演示规则",color=if(active) FxMuted else FxText,fontSize=15.sp,fontWeight=FontWeight.SemiBold)
                Text(if(demoRules) "外参俯仰修正 + 走/停提醒距离" else "原冻结 v2 规则",color=FxMuted,fontSize=12.sp)
                if(active) Text("停止后可切换",color=FxFaint,fontSize=11.sp)
            }
            Switch(demoRules,onCheckedChange=null,enabled=!active,colors=SwitchDefaults.colors(
                checkedTrackColor=FxClear,checkedThumbColor=FxNight,uncheckedTrackColor=FxSurfaceHigh,
                uncheckedThumbColor=FxMuted,uncheckedBorderColor=FxLine,disabledCheckedTrackColor=FxClear.copy(alpha=.35f)))
        }
        HorizontalDivider(color=FxLine,modifier=Modifier.padding(horizontal=16.dp))
        Row(Modifier.fillMaxWidth().clickable(enabled=active,role=Role.Button,onClick=onCalibrate).testTag("fusion_calibrate")
            .heightIn(min=56.dp).padding(horizontal=16.dp,vertical=10.dp),verticalAlignment=Alignment.CenterVertically) {
            Icon(Icons.Rounded.Straighten,null,tint=if(active) FxInfo else FxFaint,modifier=Modifier.size(20.dp))
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("低头看空地，标定安装高度",color=if(active) FxText else FxMuted,fontSize=15.sp,fontWeight=FontWeight.SemiBold)
                Text(view.heightSource,color=FxMuted,fontSize=12.sp,maxLines=1)
            }
            Text("标定",color=if(active) FxInfo else FxFaint,fontSize=14.sp,fontWeight=FontWeight.Bold)
        }
    }
}

@Composable
private fun DetailsCard(view:MobileFusionView, fresh:Boolean, active:Boolean) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    val rotation by animateFloatAsState(if(expanded) 180f else 0f,tween(220),label="chevron")
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(FxSurface)) {
        Row(Modifier.fillMaxWidth().clickable(role=Role.Button) { expanded=!expanded }.heightIn(min=50.dp).padding(horizontal=16.dp)
            .semantics { contentDescription=if(expanded) "收起开发者详情" else "展开开发者详情" },verticalAlignment=Alignment.CenterVertically) {
            Text("开发者详情",color=FxMuted,fontSize=14.sp,fontWeight=FontWeight.SemiBold,modifier=Modifier.weight(1f))
            Icon(Icons.Rounded.ExpandMore,null,tint=FxFaint,modifier=Modifier.rotate(rotation))
        }
        AnimatedVisibility(expanded,enter=fadeIn()+expandVertically(),exit=fadeOut()+shrinkVertically()) {
            fun m(v:Double)=if(v.isFinite()) "%.2f m".format(v) else "—"
            fun d(v:Double)=if(v.isFinite()) "%.1f°".format(v) else "—"
            Text("门控后 ToF ${view.tofHead}/${view.tofBody} · RGB ${view.rgbHead}/${view.rgbBody}（头/身）\n"+
                "距离 头 ${m(view.headDistanceM)} · 身 ${m(view.bodyDistanceM)} · 上限 ${m(view.limitM)}\n"+
                "尺度来源 ${view.anchor} · 俯仰 ${d(view.pitchDeg)} / 横滚 ${d(view.rollDeg)} · 外参俯仰 ${d(view.extrinsicPitchDeg)}\n"+
                "接收到结果 ${view.latencyMs} ms · 处理 ${view.processed} 帧 · 跳过 ${view.skipped} 帧\n"+
                "规则 ${if(view.demoRules) "演示（俯仰 0° + v7w）" else "冻结 v2"} · ${if(active && !fresh) "等待新帧" else "最近 3 帧中 2 帧确认"}\n"+
                "离开本页即停止；正式评价仍按冻结脚本。",
                color=FxMuted,fontSize=12.sp,lineHeight=18.sp,style=Tabular,modifier=Modifier.padding(start=16.dp,end=16.dp,bottom=14.dp))
        }
    }
}
