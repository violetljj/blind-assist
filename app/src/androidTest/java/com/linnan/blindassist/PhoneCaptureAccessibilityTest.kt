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
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.capture.PhoneCaptureState
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** Layout-only regression: no hardware connection or phone preference changes. */
@RunWith(AndroidJUnit4::class)
class PhoneCaptureAccessibilityTest {
    @get:Rule val compose = createComposeRule()

    @Test fun largeFontPreviewLabelsFitWithoutOverlapping() = assertStackedPreviews(2f, 360.dp)

    @Test fun narrowPreviewLabelsFitWithoutOverlapping() = assertStackedPreviews(1f, 300.dp)

    private fun assertStackedPreviews(fontScale: Float, width: Dp) {
        compose.setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(LocalDensity provides Density(density.density, fontScale)) {
                MaterialTheme {
                    Column(Modifier.width(width).verticalScroll(rememberScrollState())) {
                        CapturePreviews(PhoneCaptureState(), preview = { null }, onPreviewDraw = {})
                    }
                }
            }
        }
        val labels = mapOf(
            "capture_camera_status" to "capture_preview",
            "capture_tof_title" to "capture_tof_preview",
            "capture_tof_freshness" to "capture_tof_preview",
            "capture_tof_no_reading_legend" to "capture_tof_preview",
            "capture_imu_freshness" to "capture_imu_preview",
            "capture_imu_angles" to "capture_imu_preview",
            "capture_imu_accel" to "capture_imu_preview",
            "capture_imu_gyro" to "capture_imu_preview",
            "capture_imu_native_axes" to "capture_imu_preview",
        )
        labels.forEach { (tag, parent) ->
            val node = compose.onNodeWithTag(tag).performScrollTo().assertIsDisplayed()
            val layouts = mutableListOf<TextLayoutResult>()
            node.performSemanticsAction(SemanticsActions.GetTextLayoutResult) { it(layouts) }
            assertTrue("$tag has no measured text", layouts.isNotEmpty())
            val layoutDiagnostics = layouts.joinToString { result ->
                "size=${result.size}, paragraph=${result.multiParagraph.width}x${result.multiParagraph.height}, " +
                    "lines=${result.lineCount}, widthOverflow=${result.didOverflowWidth}, heightOverflow=${result.didOverflowHeight}, " +
                    "constraints=${result.layoutInput.constraints}, text=${result.layoutInput.text.text}"
            }
            // On the device, GetTextLayoutResult retained the paragraph's available width
            // (1190px) while the one-line Text node correctly measured its content (779px).
            // hasVisualOverflow compares those container widths and reports a false positive.
            // Check every rendered line against the actual node size, including truncation.
            layouts.forEach { result ->
                assertFalse("$tag exceeds max lines: $layoutDiagnostics", result.multiParagraph.didExceedMaxLines)
                (0 until result.lineCount).forEach { line ->
                    val left = result.getLineLeft(line)
                    val right = result.getLineRight(line)
                    val bottom = result.getLineBottom(line)
                    val lineDiagnostics = "$layoutDiagnostics; line=$line, left=$left, right=$right, bottom=$bottom"
                    assertFalse("$tag is ellipsized: $lineDiagnostics", result.isLineEllipsized(line))
                    assertTrue("$tag line outside width: $lineDiagnostics", left >= -1f && right <= result.size.width + 1f)
                    assertTrue("$tag line outside height: $lineDiagnostics", bottom <= result.size.height + 1f)
                }
            }
            val text = node.fetchSemanticsNode()
            val card = compose.onNodeWithTag(parent).fetchSemanticsNode()
            assertTrue("$tag clipped below card", text.positionInRoot.y + text.size.height <= card.positionInRoot.y + card.size.height + 1f)
        }
        compose.onNodeWithTag("capture_tof_title").performScrollTo()
        val title = compose.onNodeWithTag("capture_tof_title").fetchSemanticsNode()
        val freshness = compose.onNodeWithTag("capture_tof_freshness").fetchSemanticsNode()
        assertTrue("ToF title overlaps freshness", title.positionInRoot.y + title.size.height <= freshness.positionInRoot.y)
        val tof = compose.onNodeWithTag("capture_tof_preview").fetchSemanticsNode()
        val imu = compose.onNodeWithTag("capture_imu_preview").fetchSemanticsNode()
        assertTrue("IMU overlaps ToF", tof.positionInRoot.y + tof.size.height <= imu.positionInRoot.y)
    }

    @Test fun imuCheckboxIsOneLabelledFullHeightAction() {
        val enabled = mutableStateOf(true)
        compose.setContent {
            MaterialTheme { CaptureImuToggle(enabled.value, true) { enabled.value = it } }
        }
        compose.onNodeWithTag("capture_imu_enabled")
            .assertTextContains("同时记录 IMU")
            .assertHasClickAction().assertHeightIsAtLeast(48.dp).assertIsOn()
            .performClick().assertIsOff()
        compose.onAllNodes(isToggleable()).assertCountEquals(1)
    }
}
