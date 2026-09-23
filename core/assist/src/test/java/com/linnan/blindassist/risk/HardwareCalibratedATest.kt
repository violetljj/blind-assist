package com.linnan.blindassist.risk

import org.junit.Assert.*
import org.junit.Test

class HardwareCalibratedATest {
    @Test fun frozenPythonOracleParity() {
        val lines = javaClass.getResourceAsStream("/hardware_a_parity.tsv")!!.bufferedReader().readLines()
        var count = 0
        for (line in lines.filterNot { it.startsWith("#") || it.isBlank() }) {
            val f = line.split('\t')
            val t = Array(64) { DoubleArray(6) }
            if (f[8].isNotEmpty() && f[8] != "-") for (row in f[8].split(';')) {
                val r = row.split(',')
                t[r[0].toInt()] = r.drop(1).map(String::toDouble).toDoubleArray()
            }
            val d = HardwareCalibratedA.evaluate(t)
            assertEquals(f[0], f[1] == "1", d.alert)
            assertEquals(f[0], f[2] == "1", d.unknown)
            assertEquals(f[0], f[3].toDouble(), d.score, 1e-12)
            assertEquals(f[0], f[4].toInt(), d.validZones)
            assertEquals(f[0], f[5].toInt(), d.definiteZones)
            assertEquals(f[0], f[6], d.state)
            val zones = if (f[7].isEmpty()) emptyList() else f[7].split(',').map(String::toInt)
            assertEquals(f[0], zones, d.triggeringZones)
            count++
        }
        assertTrue(count >= 99)
    }

    @Test fun missingAndOutsideNeverCertifyClearSpace() {
        val t = Array(64) { DoubleArray(6) }
        assertTrue(HardwareCalibratedA.evaluate(t).unknown)
        t[0] = doubleArrayOf(1.0, 1.0, 0.4, 0.4, 0.6, 0.6)
        val d = HardwareCalibratedA.evaluate(t)
        assertFalse(d.alert)
        assertTrue(d.unknown)
        assertEquals(1, d.validZones)
    }
}
