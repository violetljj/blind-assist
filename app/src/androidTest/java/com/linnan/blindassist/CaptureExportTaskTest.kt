package com.linnan.blindassist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CaptureExportTaskTest {
    @Test
    fun processLossNeverRestoresActiveExportAsSuccessfulOrAutomaticallyRunning() {
        listOf(CaptureExportStatus.QUEUED, CaptureExportStatus.RUNNING, CaptureExportStatus.CANCELLING).forEach { phase ->
            val before = CaptureExportTask("task", "session", "content://provider/document/created", phase,
                bytesWritten = 100, totalBytes = 1000, startedAtMs = 1)
            val restored = CaptureExportTask.restore(before.json())
            assertEquals(CaptureExportStatus.INTERRUPTED, restored.status)
            assertFalse(restored.isActive)
            assertEquals(100L, restored.bytesWritten)
            assertEquals(before.destination, restored.destination)
            assertTrue(restored.message.contains("可能不完整"))
            assertTrue(restored.finishedAtMs != null)
        }
    }

    @Test
    fun terminalReceiptRoundTripsWithoutLosingFailureOrCompletionDetails() {
        listOf(CaptureExportStatus.COMPLETED, CaptureExportStatus.FAILED, CaptureExportStatus.CANCELLED,
            CaptureExportStatus.INTERRUPTED).forEach { phase ->
            val before = CaptureExportTask("task", "session", "content://provider/document/created", phase,
                bytesWritten = 100, totalBytes = 1000, startedAtMs = 1, finishedAtMs = 2,
                message = "provider result")
            assertEquals(before, CaptureExportTask.restore(before.json()))
        }
    }
}
