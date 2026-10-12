package com.linnan.blindassist.vision

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test

class CnhStrongPreprocessorTest {
    private fun calibration() = CnhStrongCalibration(DoubleArray(1024), doubleArrayOf(0.0, 0.0, 1.0), 1e-9, 8.768392178320116)

    @Test fun signedNormalizationDropsBinZeroAndRotatesZones() {
        val p = CnhStrongPreprocessor(calibration())
        val h = FloatArray(1024) { -1f }
        h[0] = 100f
        assertTrue(p.normalize(h, FloatArray(64)).all { it == 0f })
        h[1] = 2f
        val z = p.normalize(h, FloatArray(64))
        assertEquals(2f, z[63 * 16 + 1], 0f)
        assertEquals(-1f, z[2], 0f)
        assertTrue((0..63).all { z[it * 16] == 0f })
    }

    @Test fun frozenReadinessSmoothingAndGaps() {
        var calls = 0
        val model = object : CnhStrongModel {
            override fun score(input: FloatArray): FloatArray { calls++; return floatArrayOf(calls.toFloat() * 10f, -10f) }
            override fun close() = Unit
        }
        val engine = CnhStrongEngine(calibration(), model)
        val hist = FloatArray(1024) { 1f }
        for (i in 0..6) assertFalse(engine.process(hist, FloatArray(64), i.toLong(), i * 200L, -10.0, 0.0).ready)
        val result = engine.process(hist, FloatArray(64), 7, 1400, -10.0, 0.0)
        assertEquals(5, calls)
        assertTrue(result.ready && result.headStrong && !result.bodyStrong)
        assertEquals((10.0 + 40 + 120 + 320 + 800) / 31, result.headScore, 1e-10)
        val gap = engine.process(hist, FloatArray(64), 9, 1600, -10.0, 0.0)
        assertTrue(gap.historyReset)
        assertEquals(1, gap.historyLength)
        val timeGap = engine.process(hist, FloatArray(64), 10, 2201, -10.0, 0.0)
        assertEquals(1, timeGap.historyLength)
        val repeatedTime = engine.process(hist, FloatArray(64), 11, 2201, -10.0, 0.0)
        assertEquals(1, repeatedTime.historyLength)
        engine.close()
    }

    @Test fun absentNativeEvidenceCannotAlertEvenWithPositiveNetworkBias() {
        val model = object : CnhStrongModel {
            override fun score(input: FloatArray) = floatArrayOf(100f, 100f)
            override fun close() = Unit
        }
        CnhStrongEngine(calibration(), model).use { engine ->
            var result: CnhStrongResult? = null
            repeat(8) { result = engine.process(FloatArray(1024), FloatArray(64), it.toLong(), it * 200L, 0.0, 0.0) }
            assertTrue(result!!.ready)
            assertFalse(result!!.evidencePresent || result!!.headStrong || result!!.bodyStrong)
        }
    }

    @Test fun syntheticPythonProjectionParity() {
        val explicit = System.getProperty("cnhStrongFixtureDir")
        val out = if (explicit != null) File(explicit) else generateSequence(File(System.getProperty("user.dir"))) { it.parentFile }
            .map { File(it, "artifacts.local/work/mobile-fusion-integration-20261012") }.firstOrNull { it.isDirectory }
        assumeTrue("Generate local fixtures with export_cnh_strong_mobile.py", out != null && out.isDirectory)
        val folder = requireNotNull(out)
        fun floats(name: String): FloatArray {
            val b = ByteBuffer.wrap(File(folder, name).readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(b.remaining() / 4) { b.float }
        }
        val p = CnhStrongPreprocessor(CnhStrongCalibration.read(File(folder, "cnh_strong_calibration.bin")))
        val h = floats("fixture_histogram.bin"); val a = floats("fixture_ambient.bin")
        val expectedNative = floats("fixture_native.bin")
        val history = (0..7).map { frame -> p.normalize(h.copyOfRange(frame * 1024, (frame + 1) * 1024), a.copyOfRange(frame * 64, (frame + 1) * 64)) }
        history.forEachIndexed { f, z -> z.forEachIndexed { i, v -> assertEquals(expectedNative[f * 1024 + i], v, 1e-6f) } }
        for ((pitch, roll) in listOf(-13.25 to 7.5, 0.0 to 0.0, -35.0 to -12.0)) {
            val actual = p.prepare(history, pitch, roll)
            val expected = floats("pose_${pitch}_${roll}_voxels.bin")
            val error = actual.indices.maxOf { abs(actual[it] - expected[it]) }
            assertTrue("pose $pitch/$roll max feature error $error", error < 2e-5f)
        }
    }
}
