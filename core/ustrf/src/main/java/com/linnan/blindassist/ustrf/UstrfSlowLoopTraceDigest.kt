package com.linnan.blindassist.ustrf

import java.security.MessageDigest

/**
 * Image-free and text-free slow-loop audit record. Semantic labels and goal text are deliberately
 * absent, so replay can diagnose admission behavior without persisting OCR/user-query content.
 */
data class UstrfSlowLoopTraceRecord(
    val queryFrame: UstrfFrameStamp,
    val trigger: UstrfSlowLoopTrigger,
    val resolution: UstrfSlowLoopResolution
)

object UstrfSlowLoopTraceDigest {
    const val SCHEMA_VERSION = "ustrf-slow-loop-trace-v1"

    fun canonicalText(records: List<UstrfSlowLoopTraceRecord>): String = buildString {
        append(SCHEMA_VERSION).append('\n')
        records.forEach { record ->
            append(record.queryFrame.frameId).append('|')
            append(record.queryFrame.capturedAtNs).append('|')
            append(record.queryFrame.coordinateFrame).append('|')
            append(record.trigger.name).append('|')
            when (val resolution = record.resolution) {
                is UstrfSlowLoopResolution.Available -> {
                    append("AVAILABLE|")
                    append("semantic=1|")
                    append("scenePersistent=").append(if (resolution.persistentSceneFact == null) 0 else 1).append('|')
                    append("sceneDeferred=").append(if (resolution.sceneMemoryDeferredForEphemeralWorldFrame) 1 else 0).append('|')
                    append("taskGoal=").append(if (resolution.taskGoal == null) 0 else 1)
                }
                is UstrfSlowLoopResolution.Unavailable -> append("UNAVAILABLE|").append(resolution.failure.name)
            }
            append('\n')
        }
    }

    fun sha256(records: List<UstrfSlowLoopTraceRecord>): String {
        val bytes = MessageDigest.getInstance("SHA-256").digest(canonicalText(records).toByteArray(Charsets.UTF_8))
        return bytes.joinToString("") { "%02x".format(it) }
    }
}
