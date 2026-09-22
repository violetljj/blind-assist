package com.linnan.blindassist.device.glasses

import org.junit.Assert.*
import org.junit.Test

class HardwareWifiContractTest {
    @Test fun acceptsPrivateNumericHotspotAddresses() {
        assertEquals("http://192.168.43.27", HardwareWifiEndpoint.normalize("192.168.43.27"))
        assertEquals("http://10.42.0.2", HardwareWifiEndpoint.normalize("http://10.42.0.2:80/"))
        assertEquals("http://172.31.4.3", HardwareWifiEndpoint.normalize("172.31.4.3"))
    }

    @Test fun rejectsPublicLoopbackHostnameCredentialsAndAlternatePorts() {
        listOf("127.0.0.1", "8.8.8.8", "172.32.1.2", "example.com", "http://user@192.168.1.2",
            "http://192.168.1.2:81", "http://192.168.1.2/api", "http://192.168.1.2?redirect=x",
            "https://192.168.1.2", "192.168.01.2", "192.168.1.255", "192.168.1.0").forEach {
            assertTrue("Must reject $it", runCatching { HardwareWifiEndpoint.normalize(it) }.isFailure)
        }
    }

    @Test fun repeatsCannotRefreshAStoppedSensorByChangingOnlySendTime() {
        val gate = HardwareWifiProgress()
        assertTrue(gate.accept("boot-A", 1, 1000))
        assertFalse(gate.accept("boot-A", 1, 1000))
        assertFalse(gate.accept("boot-A", 2, 1000))
        assertFalse(gate.accept("boot-A", 1, 2000))
        assertTrue(gate.accept("boot-A", 2, 2000))
    }

    @Test fun restartResetsSequenceButUnannouncedRegressionIsRejected() {
        val gate = HardwareWifiProgress()
        assertTrue(gate.accept("boot-A", 900, 100_000))
        assertFalse(gate.accept("boot-A", 1, 1000))
        assertTrue(gate.accept("boot-B", 1, 1000))
        assertFalse(gate.accept("", 2, 2000))
        assertFalse(gate.accept("boot-B", -1, 2000))
    }

    @Test fun ageBoundIncludesEntireRequestAndDeviceReadoutAge() {
        assertEquals(151L, HardwareWifiFreshness.requestAgeUpperMs(1_000_000, 1_100_000, 100, 150))
        assertNull(HardwareWifiFreshness.requestAgeUpperMs(1_000_000, 1_700_000, 100, 200))
        assertNull(HardwareWifiFreshness.requestAgeUpperMs(100, 99, 100, 150))
        assertNull(HardwareWifiFreshness.requestAgeUpperMs(100, 200, 150, 100))
        assertNull(HardwareWifiFreshness.requestAgeUpperMs(100, Long.MAX_VALUE, 100, 150))
    }

    @Test fun expiredAndFutureReceiptsFailClosed() {
        assertTrue(HardwareWifiFreshness.fresh(1000, 50, 1700))
        assertFalse(HardwareWifiFreshness.fresh(1000, 50, 1701))
        assertFalse(HardwareWifiFreshness.fresh(1000, 50, 999))
        assertFalse(HardwareWifiFreshness.fresh(1000, -1, 1001))
        assertFalse(HardwareWifiFreshness.fresh(1000, Long.MAX_VALUE, 1001))
    }
}
