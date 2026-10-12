package com.linnan.blindassist.vision

import java.io.DataInputStream
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/** Optional local consumed-frame parity fixture, exported by export_atom_rgb_geometry_fixture.py.
 * A missing payload is an explicit skipped test, never a claimed real-data parity pass.
 */
class AtomRgbGeometryFixtureTest {
    @Test fun consumedFrameMatchesPythonWithIdenticalFloorDraws() {
        val path = System.getenv("BLINDASSIST_ATOM_RGB_FIXTURE")
        assumeTrue("Set BLINDASSIST_ATOM_RGB_FIXTURE to the exported .bin payload", path != null)
        DataInputStream(File(path!!).inputStream().buffered()).use { input ->
            assertEquals(0x41524731, input.readInt())
            val w = input.readInt(); val h = input.readInt()
            val k = DoubleArray(4) { input.readDouble() }
            val rct = DoubleArray(9) { input.readDouble() }
            val pitch = input.readDouble(); val roll = input.readDouble()
            val disp = FloatArray(w * h) { input.readFloat() }
            val mask = BooleanArray(w * h) { input.readBoolean() }
            val distance = DoubleArray(64) { input.readDouble() }
            val status = IntArray(64) { input.readInt() }
            val expectedDz = DoubleArray(64) { input.readDouble() }
            val expectedFactor = DoubleArray(64) { input.readDouble() }
            fun fit(): AtomRgbGeometry.Fit? = if (input.readBoolean())
                AtomRgbGeometry.Fit(input.readDouble(), input.readDouble()) else null
            val expectedTof = fit(); val expectedFloor = fit()
            val count = input.readInt()
            val samples = if (count == 0) null else AtomRgbGeometry.FloorSamples(
                IntArray(count) { input.readInt() }, List(64) { input.readInt() to input.readInt() })
            val head = input.readInt(); val body = input.readInt()
            assertEquals(-1, input.read())
            val g = AtomRgbGeometry(w, h, AtomRgbGeometry.Calibration(k[0], k[1], k[2], k[3], rct))
            val zones = g.zoneDisparity(disp)
            for (i in 0 until 64) {
                if (expectedDz[i].isNaN()) assertTrue(zones.disparity[i].isNaN())
                else assertEquals(expectedDz[i], zones.disparity[i], 1e-6)
                if (expectedFactor[i].isNaN()) assertTrue(zones.opticalDepthFactor[i].isNaN())
                else assertEquals(expectedFactor[i], zones.opticalDepthFactor[i], 1e-12)
            }
            fun assertFit(expected: AtomRgbGeometry.Fit?, actual: AtomRgbGeometry.Fit?) {
                if (expected == null) assertNull(actual)
                else {
                    assertTrue(actual != null)
                    assertEquals(expected.a, actual!!.a, 1e-5)
                    assertEquals(expected.b, actual.b, 1e-5)
                }
            }
            assertFit(expectedTof, g.tofAnchor(disp, distance, status))
            assertFit(expectedFloor, g.floorAnchor(disp, pitch, roll, samples = samples))
            val anchored = expectedTof ?: expectedFloor
            if (anchored != null) {
                val result = g.readout(disp, mask, anchored, pitch, roll)
                assertEquals(head, result.first.pixels)
                assertEquals(body, result.second.pixels)
            }
        }
    }
}
