package com.linnan.blindassist.ustrf

import java.security.MessageDigest

/** Image-free, versioned evidence for assembled session replay, including explicit adapter failures. */
object UstrfSessionTraceDigest {
    const val SCHEMA_VERSION = "ustrf-session-trace-v2"

    fun canonicalText(records: List<UstrfSessionRecord>): String = buildString {
        append(SCHEMA_VERSION).append('\n')
        records.forEach { record ->
            val decision = record.decision
            append(record.frameId).append('|')
            append(record.capturedAtNs).append('|')
            append(decision.action.name).append('|')
            append(decision.risk).append('|')
            append(decision.confidence).append('|')
            append(decision.validUntilNs).append('|')
            append(decision.experimentalCorridorOffsetCells ?: "none").append('|')
            append(decision.reasons.map { it.name }.sorted().joinToString(",")).append('|')
            append(record.assemblyFailures.map { it.name }.sorted().joinToString(",")).append('|')
            val structured = record.structuredOutput
            append(structured.action.name).append('|')
            append(structured.headingDeltaRadians).append('|')
            append(structured.speedScale).append('|')
            append(structured.corridorWidthMeters ?: "none").append('|')
            append(structured.shadowOnly)
            append('\n')
        }
    }

    fun sha256(records: List<UstrfSessionRecord>): String {
        val bytes = MessageDigest.getInstance("SHA-256").digest(canonicalText(records).toByteArray(Charsets.UTF_8))
        return bytes.joinToString("") { "%02x".format(it) }
    }
}
