package com.linnan.blindassist.session

class FpsTracker(
    private val windowMs: Long = 1000L,
    private val clock: () -> Long = { System.nanoTime() / 1_000_000L }
) {
    private var frameCount = 0
    private var windowStartMs = clock()
    private var currentFps = 0f

    fun onFrame(): Float {
        frameCount += 1
        val now = clock()
        val elapsed = now - windowStartMs
        if (elapsed >= windowMs) {
            currentFps = frameCount * 1000f / elapsed.toFloat()
            frameCount = 0
            windowStartMs = now
        }
        return currentFps
    }

    fun reset() {
        frameCount = 0
        windowStartMs = clock()
        currentFps = 0f
    }
}
