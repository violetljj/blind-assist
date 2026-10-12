package com.linnan.blindassist

import android.content.Intent
import android.net.Uri
import android.os.SystemClock
import android.provider.DocumentsContract
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.capture.PhoneCaptureStore
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.UUID
import java.util.Random
import java.util.zip.ZipInputStream

class CaptureExportServiceTest {
    @Test
    fun realExportOutlivesActivityAndCancelledBlockedWorkerPreventsOverlappingRetry() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val resolver = context.contentResolver
        val tasks = CaptureExportTasks.get(context)
        assertFalse("Do not interfere with a user export", tasks.state.value?.isActive == true || tasks.hasExecution())
        val receipt = File(context.noBackupFilesDir, "capture-export-task.json")
        val originalReceipt = receipt.takeIf { it.exists() }?.readBytes()
        val session = PhoneCaptureStore.create(context, sceneNote = "自动化导出测试（合成数据）")
        val jpeg = ByteArray(3 * 1024 * 1024).also { Random(481L).nextBytes(it) }
        val rgbPath = session.recordRgb(jpeg, 1, 2, 640, 480, 0)
        val saved = session.finish()
        val ownedUris = mutableListOf<Uri>()
        var activity: ActivityScenario<PhoneCaptureActivity>? = null
        fun document(id: String): Uri = DocumentsContract.buildDocumentUri(AUTHORITY, id).also {
            ownedUris += it
        }
        fun exists(uri: Uri): Boolean = resolver.query(uri, null, null, null, null)?.use { it.moveToFirst() } == true
        fun release(id: String) { resolver.openOutputStream(document("release-$id"), "wt")!!.close() }
        try {
            activity = ActivityScenario.launch(PhoneCaptureActivity::class.java)
            val firstId = "gated-${UUID.randomUUID()}"
            val first = document(firstId)
            activity.onActivity { assertTrue(tasks.start(saved.id, first)) }
            await { exists(first) && tasks.state.value?.status == CaptureExportStatus.RUNNING }
            activity.close()
            activity = null
            assertTrue("Leaving Activity must not stop export", tasks.state.value!!.isActive)
            release(firstId)
            await { tasks.state.value?.status == CaptureExportStatus.COMPLETED && !tasks.hasExecution() }
            var restoredJpeg: ByteArray? = null
            await {
                runCatching {
                    ZipInputStream(resolver.openInputStream(first)!!).use { zip ->
                        while (true) {
                            val entry = zip.nextEntry ?: break
                            if (entry.name == "${saved.id}/$rgbPath") restoredJpeg = zip.readBytes()
                        }
                    }
                    restoredJpeg?.contentEquals(jpeg) == true
                }.getOrDefault(false)
            }
            assertArrayEquals(jpeg, restoredJpeg)

            activity = ActivityScenario.launch(PhoneCaptureActivity::class.java)
            val secondId = "gated-${UUID.randomUUID()}"
            val second = document(secondId)
            activity.onActivity { assertTrue(tasks.start(saved.id, second)) }
            await { exists(second) && tasks.state.value?.status == CaptureExportStatus.RUNNING }
            tasks.cancel()
            // Same ordering as an FGS timeout: service stops while provider I/O remains blocked.
            context.stopService(Intent(context, CaptureExportService::class.java))
            instrumentation.waitForIdleSync()
            assertTrue(tasks.hasExecution())
            assertFalse("Old worker must retain its lease", tasks.start(saved.id, first))
            release(secondId)
            await { tasks.state.value?.status == CaptureExportStatus.CANCELLED && !tasks.hasExecution() }
            assertFalse("Cancelled destination should be deleted", exists(second))
            assertArrayEquals(jpeg, File(saved.directory, rgbPath).readBytes())

            val retry = document("retry-${UUID.randomUUID()}")
            activity.onActivity { assertTrue(tasks.start(saved.id, retry)) }
            await { tasks.state.value?.status == CaptureExportStatus.COMPLETED && !tasks.hasExecution() }
            assertEquals(retry.toString(), tasks.state.value!!.destination)
        } finally {
            // Release every possible gate before cancellation cleanup, including assertion failures.
            ownedUris.toList().map(DocumentsContract::getDocumentId).filter { it.startsWith("gated-") }
                .forEach { runCatching { release(it) } }
            tasks.cancel()
            runCatching { await { !tasks.hasExecution() } }
            activity?.close()
            if (!tasks.hasExecution()) {
                ownedUris.forEach { uri ->
                    runCatching { DocumentsContract.deleteDocument(resolver, uri) }
                }
                saved.directory.deleteRecursively() // Only the synthetic session created by this test.
                if (originalReceipt == null) receipt.delete() else receipt.writeBytes(originalReceipt)
                // All test Activities and service have ended; reload the user's original receipt next time.
                CaptureExportTasks::class.java.getDeclaredField("instance").apply { isAccessible = true }.set(null, null)
            }
        }
    }

    private fun await(condition: () -> Boolean) {
        val deadline = SystemClock.elapsedRealtime() + 20_000
        while (SystemClock.elapsedRealtime() < deadline) {
            if (condition()) return
            Thread.sleep(25)
        }
        assertTrue("Timed out waiting for export state", condition())
    }

    companion object { private const val AUTHORITY = "com.linnan.blindassist.export.testprovider" }
}
