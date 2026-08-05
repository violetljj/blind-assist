package com.linnan.blindassist.device.glasses

import org.junit.Assert.assertTrue
import org.junit.Test

class GlassesConnectionRepositoryTest {
    @Test
    fun rejectsEndpointWithoutHttpScheme() {
        assertTrue(GlassesConnectionRepository().connect("192.168.5.11").isFailure)
    }
}
