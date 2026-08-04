package com.linnan.blindassist.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ExpiringLatestResultTest {
    @Test fun unavailableFreshAndExpiredAreExplicit() {
        val store = ExpiringLatestResult<String>(100)
        assertEquals(ExpiringLatestResult.UnknownReason.UNAVAILABLE,
            (store.readAt(0) as ExpiringLatestResult.State.Unknown).reason)
        store.update("depth", capturedAtNanos = 10, completedAtNanos = 30)
        val fresh = store.readAt(110)
        assertTrue(fresh is ExpiringLatestResult.State.Fresh)
        assertEquals(100, (fresh as ExpiringLatestResult.State.Fresh).ageNanos)
        assertEquals(ExpiringLatestResult.UnknownReason.EXPIRED,
            (store.readAt(111) as ExpiringLatestResult.State.Unknown).reason)
    }
}
