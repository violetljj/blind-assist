package com.linnan.blindassist

import android.content.Intent
import android.graphics.Bitmap
import android.os.SystemClock
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.vision.MobileFusionCamera
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class MobileFusionDeviceTest {
    @Test fun calibratedRemapMatchesOpenCvFixedPointOracle() {
        val context=InstrumentationRegistry.getInstrumentation().targetContext
        val root=File(context.noBackupFilesDir,"fusion-experiment")
        val input=File(root,"remap_input.rgb").readBytes()
        val expected=File(root,"remap_expected.rgb").readBytes()
        val pixels=IntArray(640*480) { i -> -0x1000000 or ((input[i*3].toInt() and 255) shl 16) or
            ((input[i*3+1].toInt() and 255) shl 8) or (input[i*3+2].toInt() and 255) }
        val bitmap=Bitmap.createBitmap(pixels,640,480,Bitmap.Config.ARGB_8888)
        val result=MobileFusionCamera(root).undistort(bitmap)
        val actual=IntArray(480*640)
        result.bitmap.getPixels(actual,0,480,0,0,480,640)
        for(i in actual.indices) for(channel in 0..2) {
            assertEquals("pixel=$i channel=$channel",expected[i*3+channel].toInt() and 255,
                (actual[i] ushr ((2-channel)*8)) and 255)
        }
        result.bitmap.recycle(); bitmap.recycle()
    }

    @Test fun foregroundLiveFusionProducesResultsAndReleasesWorkers() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("mobileFusionLive")=="true")
        val instrumentation=InstrumentationRegistry.getInstrumentation()
        val context=instrumentation.targetContext
        val samples=mutableListOf<MobileFusionView>()
        val scenario=ActivityScenario.launch<MobileFusionActivity>(Intent(context,MobileFusionActivity::class.java))
        try {
            scenario.onActivity { it.startExperiment() }
            val deadline=SystemClock.elapsedRealtime()+55_000
            var last=-1L
            fun poseGated()=samples.size>=8 && samples.all { it.message=="姿态超出实验范围" }
            fun rgbExercised()=samples.any { it.ready && it.anchor!="NONE" && it.image!=null }
            while(SystemClock.elapsedRealtime()<deadline) {
                scenario.onActivity { activity ->
                    val v=activity.diagnosticSnapshot()
                    if(v.sequence>=0 && v.sequence!=last) { last=v.sequence; samples+=v }
                }
                if(samples.size>=8 && (poseGated() || rgbExercised())) break
                SystemClock.sleep(100)
            }
            val final= arrayOfNulls<MobileFusionView>(1)
            scenario.onActivity { final[0]=it.diagnosticSnapshot() }
            val root=File(context.noBackupFilesDir,"fusion-experiment")
            val evidenceTime=System.currentTimeMillis()
            val receipt=File(root,"live-device-check.json")
            if(receipt.exists()) receipt.copyTo(File(root,"live-device-check-before-$evidenceTime.json"))
            receipt.writeText(JSONObject()
                .put("samples",samples.size).put("last_message",final[0]?.message)
                .put("live_status",when {
                    poseGated() -> "POSE_GATED_RGB_NOT_EXERCISED"
                    samples.size>=8 && rgbExercised() -> "RGB_ANCHOR_AND_IMAGE_EXERCISED"
                    else -> "INCOMPLETE"
                })
                .put("messages",org.json.JSONArray(samples.map { it.message }))
                .put("ready",org.json.JSONArray(samples.map { it.ready }))
                .put("head",org.json.JSONArray(samples.map { it.head }))
                .put("body",org.json.JSONArray(samples.map { it.body }))
                .put("receipt_to_result_ms",org.json.JSONArray(samples.map { it.latencyMs }))
                .put("anchors",org.json.JSONArray(samples.map { it.anchor })).put("skipped",final[0]?.skipped)
                .put("scope","stationary engineering runtime, not obstacle accuracy or exposure latency").toString(2))
            val screenshot=instrumentation.uiAutomation.takeScreenshot()
            assertNotNull("Live evidence screenshot unavailable",screenshot)
            if(screenshot!=null) {
                val screen=File(root,"live-screen.png")
                if(screen.exists()) screen.copyTo(File(root,"live-screen-before-$evidenceTime.png"))
                try { screen.outputStream().use { screenshot.compress(Bitmap.CompressFormat.PNG,100,it) } }
                finally { screenshot.recycle() }
            }
            assertTrue("Too few live outputs: ${final[0]?.message}",samples.size>=8)
            if(poseGated()) {
                assertTrue("Pose-gated output must remain UNKNOWN without alerts",
                    samples.all { !it.ready && !it.head && !it.body })
            } else {
                assertTrue("No RGB anchoring/inference in valid-pose live pipeline",rgbExercised())
            }
        } finally {
            try {
                scenario.moveToState(Lifecycle.State.CREATED)
                val stopDeadline=SystemClock.elapsedRealtime()+12_000
                while(SystemClock.elapsedRealtime()<stopDeadline && fusionWorkers().isNotEmpty()) SystemClock.sleep(100)
                assertTrue("Task workers still alive: ${fusionWorkers()}",fusionWorkers().isEmpty())
            } finally { scenario.close() }
        }
    }

    private fun fusionWorkers()=Thread.getAllStackTraces().keys.filter {
        it.isAlive && it.name in setOf("fusion-tof","fusion-rgb","fusion-cleanup")
    }.map { it.name }
}
