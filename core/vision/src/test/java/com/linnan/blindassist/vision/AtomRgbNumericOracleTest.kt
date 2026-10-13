package com.linnan.blindassist.vision

import java.io.DataInputStream
import java.io.File
import kotlin.math.abs
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test

/** Original Python function bodies + frozen readout statements, fixed synthetic inputs.
 * Export with export_mobile_fusion_numeric_oracle.py; no model or device is involved.
 */
class AtomRgbNumericOracleTest {
    private data class Affine(val name: String, val x: DoubleArray, val y: DoubleArray, val fit: AtomRgbGeometry.Fit?)
    private data class Case(
        val name: String, val width: Int, val height: Int, val reset: Boolean, val checkProcess: Boolean,
        val calibration: AtomRgbGeometry.Calibration, val pitch: Double, val roll: Double, val sensorHeight: Double,
        val disparity: FloatArray, val mask: BooleanArray, val distance: DoubleArray, val status: IntArray,
        val zone: DoubleArray, val factor: DoubleArray, val tofFit: AtomRgbGeometry.Fit?, val floorFit: AtomRgbGeometry.Fit?,
        val draws: AtomRgbGeometry.FloorSamples?, val readoutFit: AtomRgbGeometry.Fit?, val pixels: IntArray,
        val alarms: BooleanArray, val anchor: AtomRgbGeometry.Anchor, val poseOk: Boolean,
        val processFit: AtomRgbGeometry.Fit?, val processPixels: IntArray, val processAlarms: BooleanArray,
    ) {
        fun geometry() = AtomRgbGeometry(width, height, calibration)
    }
    private data class Oracle(val affine: List<Affine>, val cases: List<Case>, val temporal: List<List<BooleanArray>>)
    private val oracle by lazy { readOracle() }

    @Test fun affineFitMatchesNumpyIncludingRankOneInputs() {
        var maxError = 0.0
        for (c in oracle.affine) {
            val actual = AtomRgbGeometry.fitAffine(c.y, c.x)
            maxError = maxOf(maxError, assertFit(c.name, c.fit, actual))
        }
        println("affine_cases=${oracle.affine.size} max_coefficient_abs_error=$maxError")
    }

    @Test fun rotatedZoneSamplingAndTofScaleMatchOriginalPython() {
        var maxZone = 0.0
        var maxFactor = 0.0
        var maxFit = 0.0
        for (c in oracle.cases) {
            val g = c.geometry()
            val actual = g.zoneDisparity(c.disparity)
            for (zone in 0..63) {
                maxZone = maxOf(maxZone, assertNumber("${c.name} zone $zone", c.zone[zone], actual.disparity[zone], 1e-6))
                maxFactor = maxOf(maxFactor, assertNumber("${c.name} factor $zone", c.factor[zone], actual.opticalDepthFactor[zone], 1e-12))
            }
            maxFit = maxOf(maxFit, assertFit(c.name, c.tofFit, g.tofAnchor(c.disparity, c.distance, c.status)))
        }
        println("zone_cases=${oracle.cases.size} zones=${oracle.cases.size * 64} max_zone_abs_error=$maxZone max_factor_abs_error=$maxFactor max_tof_fit_abs_error=$maxFit")
    }

    @Test fun floorFitMatchesPythonWithIdenticalOutlierSamples() {
        var maxError = 0.0
        for (c in oracle.cases) {
            val actual = c.geometry().floorAnchor(c.disparity, c.pitch, c.roll, c.sensorHeight, c.draws)
            maxError = maxOf(maxError, assertFit(c.name, c.floorFit, actual))
        }
        println("floor_cases=${oracle.cases.size} max_coefficient_abs_error=$maxError")
    }

    @Test fun readoutMatchesFrozenFloat32ArithmeticAndInclusiveFaces() {
        var count = 0
        for (c in oracle.cases) {
            val fit = c.readoutFit ?: continue
            val actual = c.geometry().readout(c.disparity, c.mask, fit, c.pitch, c.roll)
            assertQuery(c.name, c.pixels, c.alarms, actual.first, actual.second)
            count++
        }
        println("readout_cases=$count query_counts_and_alarms_exact=true")
    }

    @Test fun processFallbackAndGatedFrameStateMatchFrozenRules() {
        var geometry: AtomRgbGeometry? = null
        var count = 0
        for (c in oracle.cases.filter { it.checkProcess }) {
            if (c.reset || geometry == null) geometry = c.geometry()
            val actual = geometry.process(c.disparity, c.mask, c.distance, c.status,
                c.pitch, c.roll, c.sensorHeight, c.draws)
            assertEquals(c.name, c.anchor, actual.anchor)
            assertEquals(c.name, c.poseOk, actual.poseOk)
            assertFit(c.name, c.processFit, actual.fit)
            assertQuery(c.name, c.processPixels, c.processAlarms, actual.head, actual.body)
            count++
        }
        println("state_frames=$count anchor_pose_queries_exact=true")
    }

    @Test fun twoOfThreeMatchesAllFiveFrameTwoQuerySequences() {
        val temporal = AtomRgbTwoOfThree()
        var count = 0
        for ((sequence, rows) in oracle.temporal.withIndex()) {
            temporal.reset()
            for ((frame, row) in rows.withIndex()) {
                assertEquals("sequence=$sequence frame=$frame", row[2] to row[3], temporal.push(row[0], row[1]))
                count++
            }
        }
        println("temporal_sequences=${oracle.temporal.size} steps=$count query_decisions=${count * 2} exact=true")
    }

    private fun assertQuery(name: String, pixels: IntArray, alarms: BooleanArray,
                            head: AtomRgbGeometry.Query, body: AtomRgbGeometry.Query) {
        assertEquals("$name HEAD pixels", pixels[0], head.pixels)
        assertEquals("$name BODY pixels", pixels[1], body.pixels)
        assertEquals("$name HEAD alarm", alarms[0], head.alarm)
        assertEquals("$name BODY alarm", alarms[1], body.alarm)
    }

    private fun assertFit(name: String, expected: AtomRgbGeometry.Fit?, actual: AtomRgbGeometry.Fit?): Double {
        if (expected == null) { assertNull(name, actual); return 0.0 }
        assertNotNull(name, actual)
        val a = assertNumber("$name a", expected.a, actual!!.a, 1e-5)
        val b = assertNumber("$name b", expected.b, actual.b, 1e-5)
        return maxOf(a, b)
    }

    private fun assertNumber(name: String, expected: Double, actual: Double, tolerance: Double): Double {
        if (!expected.isFinite()) {
            if (expected.isNaN()) assertTrue(name, actual.isNaN()) else assertEquals(name, expected, actual, 0.0)
            return 0.0
        }
        assertEquals(name, expected, actual, tolerance)
        return abs(expected - actual)
    }

    private fun readOracle(): Oracle {
        val path = System.getenv("BLINDASSIST_MOBILE_FUSION_ORACLE")
        assumeTrue("Set BLINDASSIST_MOBILE_FUSION_ORACLE to the exported synthetic .bin", path != null)
        DataInputStream(File(path!!).inputStream().buffered()).use { input ->
            fun doubles(n: Int) = DoubleArray(n) { input.readDouble() }
            fun ints(n: Int) = IntArray(n) { input.readInt() }
            fun flags(n: Int) = BooleanArray(n) { input.readBoolean() }
            fun fit() = if (input.readBoolean()) AtomRgbGeometry.Fit(input.readDouble(), input.readDouble()) else null
            assertEquals(0x4d464e31, input.readInt())
            val affine = List(input.readInt()) {
                val name = input.readUTF(); val n = input.readInt()
                Affine(name, doubles(n), doubles(n), fit())
            }
            val cases = List(input.readInt()) {
                val name = input.readUTF(); val w = input.readInt(); val h = input.readInt()
                val reset = input.readBoolean(); val process = input.readBoolean()
                val k = doubles(4); val rct = doubles(9)
                val pitch = input.readDouble(); val roll = input.readDouble(); val sensorHeight = input.readDouble()
                val disparity = FloatArray(w * h) { input.readFloat() }; val mask = flags(w * h)
                val distance = doubles(64); val status = ints(64); val zone = doubles(64); val factor = doubles(64)
                val tof = fit(); val floor = fit()
                val samples = input.readInt()
                val draws = if (samples == 0) null else AtomRgbGeometry.FloorSamples(ints(samples),
                    List(64) { input.readInt() to input.readInt() })
                val readout = fit(); val pixels = ints(2); val alarms = flags(2)
                val anchor = AtomRgbGeometry.Anchor.valueOf(input.readUTF()); val pose = input.readBoolean()
                Case(name, w, h, reset, process, AtomRgbGeometry.Calibration(k[0], k[1], k[2], k[3], rct),
                    pitch, roll, sensorHeight, disparity, mask, distance, status, zone, factor, tof, floor,
                    draws, readout, pixels, alarms, anchor, pose, fit(), ints(2), flags(2))
            }
            val sequences = input.readInt(); val steps = input.readInt()
            val temporal = List(sequences) { List(steps) { flags(4) } }
            assertEquals("Unexpected oracle tail", -1, input.read())
            return Oracle(affine, cases, temporal)
        }
    }
}
