package com.linnan.blindassist.capture

import org.junit.Assert.*
import org.junit.Test

class CnhCaptureEndpointTest {
    @Test fun acceptsPhoneHotspotAndSameLanIpv4() {
        assertEquals("192.168.5.10", captureHost(" 192.168.5.10 "))
        assertEquals("10.42.0.2", captureHost("http://10.42.0.2:80/"))
        assertEquals("172.31.0.5", captureHost("172.31.0.5"))
    }

    @Test fun rejectsPublicDnsAndUrlPartsThatDoNotNameTheDevice() {
        listOf("8.8.8.8", "127.0.0.1", "example.com", "192.168.05.10", "192.168.5.0",
            "192.168.5.255", "172.32.0.5", "http://user@192.168.5.10", "https://192.168.5.10",
            "192.168.5.10/api", "192.168.5.10?query=1", "192.168.5.10:3335").forEach {
            assertTrue("Must reject $it", runCatching { captureHost(it) }.isFailure)
        }
    }
}
