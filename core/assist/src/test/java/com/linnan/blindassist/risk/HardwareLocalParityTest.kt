package com.linnan.blindassist.risk

import java.io.DataInputStream
import java.io.File
import java.util.zip.GZIPInputStream
import org.junit.Assert.*
import org.junit.Test

class HardwareLocalParityTest {
    private fun root(): File = generateSequence(File(System.getProperty("user.dir"))) { it.parentFile }
        .first { File(it, "settings.gradle.kts").isFile }

    @Test fun frozenPythonRepresentationAndProbabilitiesMatch() {
        val root = root()
        val fixture = requireNotNull(javaClass.getResourceAsStream("/hardware_local/local-parity.bin.gz"))
        val model = File(root, "app/src/main/assets/hardware_local/local-v1.bin").inputStream().use { HardwareLocalModel.read(it) }
        fixture.use { raw ->
            val stream = DataInputStream(GZIPInputStream(raw))
            assertEquals(0x42414631, stream.readInt())
            repeat(stream.readInt()) { case ->
                val rgb = IntArray(320 * 180) { (stream.readUnsignedByte() shl 16) or (stream.readUnsignedByte() shl 8) or stream.readUnsignedByte() }
                val tof = Array(64) { DoubleArray(6) { stream.readDouble() } }
                val expected = Array(6) { FloatArray(961) { stream.readFloat() } }
                val probability = DoubleArray(6) { stream.readDouble() }
                val actual = HardwareLocalFeatures.extract(rgb, tof)
                val centre = HardwareLocalFeatures.extract(rgb, tof, intArrayOf(1, 4))
                assertArrayEquals(actual[1], centre[0], 0f)
                assertArrayEquals(actual[4], centre[1], 0f)
                repeat(6) { query ->
                    assertArrayEquals("case=$case query=$query", expected[query], actual[query], 1e-6f)
                    assertEquals("case=$case query=$query probability", probability[query], model.predict(actual[query]), 1e-12)
                }
            }
            assertEquals(-1, stream.read())
        }
    }

    @Test fun overlappingAndMalformedInputsRejected() {
        val rgb = IntArray(320 * 180)
        val overlapping = Array(64) { doubleArrayOf(.2, 1.0, 0.0, 0.0, 1.0, 1.0) }
        assertThrows(IllegalArgumentException::class.java) { HardwareLocalFeatures.extract(rgb, overlapping) }
        assertThrows(IllegalArgumentException::class.java) { HardwareLocalFeatures.extract(IntArray(1), overlapping) }
    }
}
