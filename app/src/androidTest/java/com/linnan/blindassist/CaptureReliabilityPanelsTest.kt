package com.linnan.blindassist

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.linnan.blindassist.capture.*
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import androidx.test.platform.app.InstrumentationRegistry
import android.graphics.Bitmap
import java.io.File

class CaptureReliabilityPanelsTest {
    @get:Rule val compose = createComposeRule()

    @Test fun exportCancelAndRetryRemainReachableAtLargeFont() {
        val status = mutableStateOf(CaptureExportStatus.RUNNING)
        var cancelled = 0
        var retried = 0
        compose.setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(LocalDensity provides Density(density.density, 2f)) {
                MaterialTheme {
                    Column(Modifier.width(300.dp).verticalScroll(rememberScrollState())) {
                        CaptureExportTaskCard(CaptureExportTask("task", "session", "content://test/new",
                            status.value, 50, 100, 1, message = "正在导出"), { cancelled++ }, { retried++ }, true)
                    }
                }
            }
        }
        compose.onNodeWithTag("capture_export_cancel").performScrollTo().assertIsDisplayed().performClick()
        assertEquals(1, cancelled)
        compose.runOnIdle { status.value = CaptureExportStatus.CANCELLING }
        compose.onNodeWithTag("capture_export_cancel").assertIsNotEnabled()
        compose.runOnIdle { status.value = CaptureExportStatus.FAILED }
        compose.onNodeWithTag("capture_export_retry").performScrollTo().assertIsDisplayed().performClick()
        assertEquals(1, retried)
        screenshot("capture-export-large-font.png")
    }

    @Test fun historyShowsObservedRecoveryAndPersistenceFailure() {
        val event = CaptureConnectionEvent("tof", CaptureConnectionReason.RECOVERED, 5000000,
            "数据已恢复", 3, 2100, 1000)
        compose.setContent {
            MaterialTheme {
                Column(Modifier.width(300.dp).verticalScroll(rememberScrollState())) {
                    CaptureConnectionCard(CaptureConnectionDiagnostics(history = listOf(event.copy(message = "较早的断连原因")) +
                        List(12) { event.copy(wallClockMs = 2000L + it) }, persistenceError = "记录保存失败"))
                }
            }
        }
        compose.onNodeWithText("记录保存失败").assertIsDisplayed()
        compose.onNodeWithText("数据已恢复").assertDoesNotExist()
        compose.onNodeWithTag("capture_connection_expand").performClick()
        compose.onAllNodesWithText("数据已恢复")[0].performScrollTo().assertIsDisplayed()
        compose.onAllNodesWithText("连接尝试 3 次 · 恢复用时 2.1 秒")[0].performScrollTo().assertIsDisplayed()
        compose.onNodeWithText("较早的断连原因").assertDoesNotExist()
        compose.onNodeWithTag("capture_connection_show_all").performScrollTo().performClick()
        compose.onNodeWithText("较早的断连原因").performScrollTo().assertIsDisplayed()
        screenshot("capture-connection-panel.png")
    }

    private fun screenshot(name: String) {
        compose.waitForIdle()
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val bitmap = requireNotNull(instrumentation.uiAutomation.takeScreenshot())
        try {
            File(instrumentation.targetContext.cacheDir, name).outputStream().use {
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
            }
        } finally { bitmap.recycle() }
    }
}
