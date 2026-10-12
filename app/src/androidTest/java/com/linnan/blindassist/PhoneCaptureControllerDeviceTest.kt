package com.linnan.blindassist

import android.content.Context
import android.content.ContextWrapper
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.capture.*
import java.io.File
import java.util.UUID
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/** Synthetic callback/time tests of the real controller and store; no sensors or network. */
@RunWith(AndroidJUnit4::class)
class PhoneCaptureControllerDeviceTest {
    private lateinit var directory: File
    private lateinit var controller: PhoneCaptureController
    private val camera = Camera()
    private val tof = Tof()
    private var now = 1_000_000_000L

    @Before fun prepare() {
        val base = InstrumentationRegistry.getInstrumentation().targetContext
        directory = File(base.cacheDir, "capture-controller-test-${UUID.randomUUID()}").apply { mkdirs() }
        val isolated = object : ContextWrapper(base) {
            override fun getApplicationContext(): Context = this
            override fun getNoBackupFilesDir(): File = directory
        }
        controller = PhoneCaptureController(isolated, monotonicClock = { now })
        controller.setImuEnabled(false)
        controller.connectCamera(camera)
        controller.connect(tof)
        camera.emit(now)
        tof.emit(now)
    }

    @After fun close() {
        controller.shutdown()
        directory.deleteRecursively()
    }

    @Test fun shortStallKeepsRecordingAndRecordsEachEpisodeOnce() {
        controller.startRecording("synthetic short stall")
        now += 3_000_000_000L
        camera.emit(now)
        val warning = controller.snapshot()
        assertTrue(warning.recording)
        assertFalse(warning.tofReady)
        assertTrue(warning.message.contains("ToF 暂无新数据（3000 毫秒）"))
        repeat(3) { controller.snapshot() }
        tof.emit(now)
        val resumed = controller.snapshot()
        assertTrue(resumed.recording)
        assertTrue(resumed.tofReady)
        assertFalse(resumed.message.contains("暂无新数据"))
        repeat(3) { controller.snapshot() }
        now += 3_000_000_000L
        camera.emit(now)
        controller.snapshot()
        tof.emit(now)
        controller.snapshot()
        val saved = controller.stopRecording()!!
        val events = File(saved.directory, "events.jsonl").readLines().map(::JSONObject)
        assertEquals(2, events.count { it.getString("kind") == "tof_stalled" })
        assertEquals(2, events.count { it.getString("kind") == "tof_recovered" })
        assertTrue(events.filter { it.getString("kind") == "tof_recovered" }
            .all { it.getString("message").contains("receipt_gap_ms=3000") })
        assertEquals("COMPLETED", saved.status)
    }

    @Test fun eightSecondLimitEndsOnceAndRetainsTofCauseThroughStoppedCallbacks() {
        controller.startRecording("synthetic timeout")
        now += 8_000_000_000L
        camera.emit(now)
        assertTrue(controller.snapshot().recording)
        now += 1_000_000L
        camera.emit(now)
        val stopped = controller.snapshot()
        assertFalse(stopped.recording)
        assertTrue(stopped.message.contains("ToF 超过 8 秒无新数据（8001 毫秒）"))
        assertFalse(stopped.message.contains("STOPPED"))
        assertEquals(stopped.message, controller.snapshot().message)
        val manifest = JSONObject(File(sessionDirectory(), "manifest.json").readText())
        assertEquals("INTERRUPTED", manifest.getString("status"))
        assertEquals("input_stale_tof", manifest.getString("finish_reason"))
        assertEquals(1, events().count { it.getString("kind") == "input_timeout" })
        assertEquals(1, camera.closes)
        assertEquals(1, tof.closes)
    }

    @Test fun explicitDisconnectRetainsFirstCauseRatherThanOtherSourceStopMessage() {
        controller.startRecording("synthetic eof")
        camera.emit(now)
        tof.emit(now)
        tof.status(CnhCaptureStatus(CnhConnectionState.DISCONNECTED, "peer EOF"))
        val stopped = controller.snapshot()
        assertFalse(stopped.recording)
        assertTrue(stopped.message.contains("ToF 连接中断：peer EOF"))
        assertEquals("tof_disconnected", JSONObject(File(sessionDirectory(), "manifest.json").readText())
            .getString("finish_reason"))
        controller.stopForeground()
        assertEquals(stopped.message, controller.snapshot().message)
    }

    @Test fun stalePreviewCannotStartRecordingDuringGracePeriod() {
        now += 3_000_000_000L
        assertFalse(controller.snapshot().cameraReady)
        assertNull(controller.peekCameraFrame())
        try {
            controller.startRecording("must reject stale input")
            fail("stale input must not start recording")
        } catch (_: IllegalStateException) { }
        assertFalse(controller.snapshot().recording)
    }

    private fun sessionDirectory() = File(directory, "phone-capture").listFiles()!!.single()
    private fun events() = File(sessionDirectory(), "events.jsonl").readLines().map(::JSONObject)

    private class Camera : AtomCameraCaptureSource {
        override val transport = "fake-camera"
        lateinit var frame: (AtomCameraCaptureFrame) -> Unit
        lateinit var status: (CnhCaptureStatus) -> Unit
        var closes = 0
        override fun start(onFrame: (AtomCameraCaptureFrame) -> Unit, onStatus: (CnhCaptureStatus) -> Unit) {
            frame = onFrame
            status = onStatus
            status(CnhCaptureStatus(CnhConnectionState.CONNECTED, "camera connected"))
        }
        fun emit(now: Long) = frame(AtomCameraCaptureFrame(byteArrayOf(-1, -40, -1, -39), now,
            1, 1, now, "fake", now / 1000, now / 1000, emptyMap(), "fake-camera"))
        override fun close() {
            closes++
            status(CnhCaptureStatus(CnhConnectionState.STOPPED, "camera STOPPED"))
        }
    }

    private class Tof : CnhCaptureSource {
        override val transport = "fake-tof"
        lateinit var record: (CnhCaptureRecord) -> Unit
        lateinit var status: (CnhCaptureStatus) -> Unit
        var closes = 0
        override fun start(onRecord: (CnhCaptureRecord) -> Unit, onStatus: (CnhCaptureStatus) -> Unit,
                           onRawChunk: ((ByteArray, Long) -> Unit)?) {
            record = onRecord
            status = onStatus
            status(CnhCaptureStatus(CnhConnectionState.CONNECTED, "tof connected"))
        }
        fun emit(now: Long) = record(CnhCaptureRecord(now, "{}\n".toByteArray(), now, "cnh_frame",
            CnhCaptureFrame(now, now / 1_000_000, 0, 0, List(64) { 1000 }, List(64) { 5 },
                List(64) { 1 }, List(64) { List(16) { 0 } }, List(64) { List(16) { 0 } },
                List(64) { 0 }, List(64) { 0 }, emptyList()), emptyList()))
        override fun close() {
            closes++
            status(CnhCaptureStatus(CnhConnectionState.STOPPED, "tof STOPPED"))
        }
    }
}
