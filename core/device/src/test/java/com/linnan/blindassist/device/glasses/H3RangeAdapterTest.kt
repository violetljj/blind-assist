package com.linnan.blindassist.device.glasses

import com.linnan.blindassist.capture.CnhCaptureFrame
import com.linnan.blindassist.capture.CnhCaptureRecord
import com.linnan.blindassist.risk.DemoTofDecision
import com.linnan.blindassist.risk.HardwareLocalFeatures
import com.linnan.blindassist.risk.HardwareLocalModel
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream

class H3RangeAdapterTest {
    private fun boot(id: String) = CnhCaptureRecord(1,
        "{\"boot_id\":\"$id\"}".toByteArray(), 0, "boot", null, emptyList())
    private fun frame(seq: Long = 1, ms: Long = 10, receiptNs: Long = 1_000_000_000,
        issues: List<String> = emptyList()) = CnhCaptureRecord(2, byteArrayOf(), receiptNs, "cnh_frame",
        CnhCaptureFrame(seq, ms, 1000, 1, List(64) { 500 }, List(64) { 5 }, List(64) { 1 },
            emptyList(), emptyList(), emptyList(), emptyList(), emptyList()), issues)

    @Test fun requiresBootAndPreservesRawZoneOrderAndReceiptClock() {
        val adapter = H3RangeAdapter()
        assertNull(adapter.accept(frame(), 1_000_000_000))
        adapter.accept(boot("a"), 0)
        val sample = requireNotNull(adapter.accept(frame(), 1_000_000_000))
        assertEquals("a", sample.boot)
        assertEquals(1000L, sample.receivedMs)
        assertEquals((0 until 64).toList(), sample.cells.map { it.zone })
        assertEquals(500.0, sample.cells[0].rangeMm, 0.0)
    }

    @Test fun duplicatesAndReorderedDataCannotRefreshButGapsAndBootChangesCan() {
        val adapter = H3RangeAdapter().apply { accept(boot("a"), 0) }
        assertNotNull(adapter.accept(frame(), 1_000_000_000))
        assertNull(adapter.accept(frame(), 1_000_000_000))
        assertNull(adapter.accept(frame(2, 10), 1_000_000_000))
        assertNotNull(adapter.accept(frame(4, 40, issues = listOf("sequence_gap:2")), 1_000_000_000))
        assertNull(adapter.accept(frame(3, 30), 1_000_000_000))
        adapter.accept(boot("b"), 0)
        assertNotNull(adapter.accept(frame(), 1_000_000_000))
    }

    @Test fun rejectsStaleFutureAndMalformedRecordsAndAcceptsUint32Rollover() {
        val adapter = H3RangeAdapter().apply { accept(boot("a"), 0) }
        assertNull(adapter.accept(frame(), 1_750_000_001))
        assertNull(adapter.accept(frame(), 999_999_999))
        assertNull(adapter.accept(frame(issues = listOf("device_clock_discontinuity")), 1_000_000_000))
        assertNotNull(adapter.accept(frame(0xffffffffL, 0xffffffffL), 1_750_000_000))
        assertNotNull(adapter.accept(frame(0, 0), 1_000_000_000))
    }

    private fun input() = HardwareDemoSnapshot("live", "h3", 1, 1, null, emptyList(), 8, 8,
        DemoTofDecision(true, "ALERT_SUPPORTED", 500, listOf(0), 64), true, "H3",
        tofAgeMs = null, tofTimingBasis = "host_receipt_only", tofReceiptAgeMs = 740)

    @Test fun receiptFreshnessExpiresAfterDispatchWithoutInventingExposureAge() {
        val input = input()
        val boundary = HardwareLocalRuntime.afterProcessing(input, input, 10)
        assertTrue(boundary.tofUsable)
        assertNull(boundary.tofAgeMs)
        assertEquals(750L, boundary.tofReceiptAgeMs)
        val expired = HardwareLocalRuntime.afterProcessing(input, input, 11)
        assertFalse(expired.tofUsable)
        assertFalse(expired.decision.alert)
    }

    @Test fun unmappedH3RetainsBasicDecisionWithoutEvenCalibratedAReplacement() {
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { output ->
            output.writeInt(0x42414c31); output.writeInt(HardwareLocalFeatures.FEATURE_SIZE)
            output.writeDouble(0.5); output.writeDouble(0.0); output.writeInt(1); output.writeInt(1)
            output.writeDouble(0.0); output.writeInt(0); output.writeDouble(0.0)
            output.writeInt(0); output.writeInt(0); output.writeBoolean(false); output.writeBoolean(true)
        }
        val runtime = HardwareLocalRuntime(HardwareLocalModel.read(bytes.toByteArray().inputStream())) { 1000 }
        val input = input()
        val result = runtime.evaluate(input)
        assertEquals(input.decision, result.decision)
        assertNull(result.tofAgeMs)
        assertTrue(result.status.contains("基础测距"))
    }
}
