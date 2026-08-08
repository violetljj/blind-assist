package com.linnan.blindassist.hftf

import android.os.Bundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class Dav2Fp16DecodeParityDeviceTest {
    @Test
    fun everyHalfBitPatternMatchesAndroidHalf() {
        val input = ByteBuffer.allocateDirect(BIT_PATTERNS * 2).order(ByteOrder.nativeOrder())
        val shorts = input.asShortBuffer()
        for (bits in 0 until BIT_PATTERNS) shorts.put(bits, bits.toShort())
        input.position(0)
        input.limit(BIT_PATTERNS * 2)
        val actual = FloatArray(BIT_PATTERNS)
        var mismatches = 0
        var firstMismatch = -1
        var firstExpectedBits = 0
        var firstActualBits = 0
        var nanMismatches = 0
        var finiteMismatches = 0

        Dav2NativePreprocessor().use { native ->
            native.decodeFp16ToFloatStrict(input, actual)
        }
        for (bits in 0 until BIT_PATTERNS) {
            val expectedBits = java.lang.Float.floatToRawIntBits(android.util.Half.toFloat(bits.toShort()))
            val actualBits = java.lang.Float.floatToRawIntBits(actual[bits])
            if (actualBits != expectedBits) {
                mismatches++
                val isHalfNan = bits and 0x7c00 == 0x7c00 && bits and 0x03ff != 0
                if (isHalfNan) nanMismatches++ else finiteMismatches++
                if (firstMismatch < 0) {
                    firstMismatch = bits
                    firstExpectedBits = expectedBits
                    firstActualBits = actualBits
                }
            }
        }
        val report = JSONObject()
            .put("schema", "blindassist_dav2_fp16_native_decode_parity_r0")
            .put("bit_patterns", BIT_PATTERNS)
            .put("mismatches", mismatches)
            .put("first_mismatch", firstMismatch)
            .put("first_expected_float_bits", firstExpectedBits.toUInt().toString(16))
            .put("first_actual_float_bits", firstActualBits.toUInt().toString(16))
            .put("nan_mismatches", nanMismatches)
            .put("finite_mismatches", finiteMismatches)
            .put("pass", mismatches == 0)
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        File(instrumentation.targetContext.filesDir, REPORT_FILE).writeText(report.toString(2))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
        assertEquals(report.toString(), 0, mismatches)
    }

    private companion object {
        const val BIT_PATTERNS = 65_536
        const val REPORT_KEY = "dav2_fp16_native_decode_parity_r0_report"
        const val REPORT_FILE = "dav2-fp16-native-decode-parity-r0.json"
    }
}
