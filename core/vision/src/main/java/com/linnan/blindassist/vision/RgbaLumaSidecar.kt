package com.linnan.blindassist.vision

import java.util.concurrent.Executor

/**
 * Camera-analyser hand-off for an optional, non-blocking luma sidecar.
 *
 * [submit] must be called before the caller closes its [RgbaVisionFrame]. It
 * synchronously makes an owned luma copy, so the worker never retains a
 * CameraX buffer. Processing is latest-only and results are age-bounded by
 * [LatestOnlySidecar]. This class deliberately contains neither a vision
 * model nor alert semantics.
 */
class RgbaLumaSidecar<O>(
    executor: Executor,
    maxResultAgeNanos: Long,
    outputSize: Int = DEFAULT_OUTPUT_SIZE,
    mode: RgbaLumaResampler.Mode = RgbaLumaResampler.Mode.WEIGHTED_RGB,
    process: (OwnedLumaFrame) -> O,
    onFreshResult: (LatestOnlySidecar.Result<O>) -> Unit,
    onFailure: (Throwable) -> Unit = {},
    nowNanos: () -> Long = System::nanoTime
) : AutoCloseable {
    private val resampler = RgbaLumaResampler(outputSize, mode)
    private val sidecar = LatestOnlySidecar(
        executor = executor,
        maxResultAgeNanos = maxResultAgeNanos,
        process = process,
        onFreshResult = onFreshResult,
        onFailure = onFailure,
        nowNanos = nowNanos
    )

    /** Takes ownership of a copied luma frame; the RGBA source remains caller-owned. */
    fun submit(frame: RgbaVisionFrame, capturedAtNanos: Long): Boolean =
        sidecar.submit(OwnedLumaFrame(resampler.sample(frame)), capturedAtNanos)

    override fun close() = sidecar.close()

    /**
     * The pixels are valid only for the synchronous [process] callback. They
     * are cleared after processing or replacement, making retained data a bug.
     */
    class OwnedLumaFrame internal constructor(val pixels: ByteArray) : AutoCloseable {
        private var closed = false

        override fun close() {
            if (!closed) {
                closed = true
                pixels.fill(0)
            }
        }
    }

    private companion object {
        const val DEFAULT_OUTPUT_SIZE = 320
    }
}
