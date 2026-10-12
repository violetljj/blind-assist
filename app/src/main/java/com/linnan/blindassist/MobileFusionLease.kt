package com.linnan.blindassist

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Keeps new H3 clients behind asynchronous experiment teardown across Activity transitions. */
internal object MobileFusionLease {
    private val owner = AtomicReference<CountDownLatch?>()
    val busy: Boolean get() = owner.get() != null
    fun acquire(): CountDownLatch {
        val latch=CountDownLatch(1)
        check(owner.compareAndSet(null,latch)) { "上一实验仍在释放资源，请稍候" }
        return latch
    }
    fun release(latch:CountDownLatch) { owner.compareAndSet(latch,null); latch.countDown() }
    fun awaitReleased() {
        check(owner.get()?.await(45,TimeUnit.SECONDS) != false) { "融合实验尚未释放，请稍候重试" }
    }
}
