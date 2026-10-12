package com.linnan.blindassist

import android.os.Trace
import androidx.compose.runtime.Composer
import androidx.compose.runtime.CompositionTracer
import androidx.compose.runtime.InternalComposeTracingApi
import java.util.concurrent.ConcurrentHashMap

/** Opt-in diagnostic only. Install/close between main-thread frames, never during composition. */
@OptIn(InternalComposeTracingApi::class)
internal class HardwareComposeTrace private constructor(
    private val previous: CompositionTracer?,
) : CompositionTracer, AutoCloseable {
    private val names = ConcurrentHashMap<String, String>()

    // Remain stable throughout the capture: compiler start/end guards must agree for nested events.
    override fun isTraceInProgress() = true

    override fun traceEventStart(key: Int, dirty1: Int, dirty2: Int, info: String) {
        val name = names.getOrPut(info) { info.replace('\n', ' ').replace('|', '/').take(127) }
        Trace.beginSection(name)
    }

    override fun traceEventEnd() { Trace.endSection() }

    override fun close() {
        if (previous != null) Composer.setTracer(previous)
        else {
            // Runtime 1.7.4 accepts a null tracer in bytecode, but Kotlin declares the argument non-null.
            // Invoke that same setter reflectively to restore the original absence, not a permanent no-op.
            Composer.Companion.javaClass.getMethod("setTracer", CompositionTracer::class.java)
                .invoke(Composer.Companion, null)
        }
        names.clear()
    }

    companion object {
        fun install(): HardwareComposeTrace {
            // There is no public getter. Preserve any tracer installed by another diagnostic library.
            val previous = Class.forName("androidx.compose.runtime.ComposerKt")
                .getDeclaredField("compositionTracer").apply { isAccessible = true }
                .get(null) as? CompositionTracer
            return HardwareComposeTrace(previous).also { Composer.setTracer(it) }
        }
    }
}
