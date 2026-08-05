package com.linnan.blindassist.device.glasses

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GlassesConnectionRepositoryDeviceTest {
    @Test
    fun phoneConnectsToAtomS3rStatusRangeAndStream() {
        val status = GlassesConnectionRepository()
            .connect("http://192.168.5.11")
            .getOrThrow()

        assertTrue(status.firmwareVersion.startsWith("atoms3r_m12_tof4m_"))
        assertTrue(status.tofValid)
        assertTrue(status.tofRangeMm != null)
        assertTrue(status.streamReachable)
    }
}
