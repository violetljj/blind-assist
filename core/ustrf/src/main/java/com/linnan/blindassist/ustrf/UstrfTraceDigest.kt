package com.linnan.blindassist.ustrf

import java.security.MessageDigest

/**
 * Canonical, image-free trace evidence for deterministic replay checks.
 * Raw video, pose payloads, and user data stay in artifacts.local; this class only fingerprints
 * the safety decisions that a replay produced.
 */
object UstrfTraceDigest {
    const val SCHEMA_VERSION = "ustrf-trace-v1"

    fun canonicalText(records: List<UstrfReplayRecord>): String = buildString {
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
            append(decision.reasons.map { it.name }.sorted().joinToString(","))
            append('\n')
        }
    }

    fun sha256(records: List<UstrfReplayRecord>): String {
        val bytes = MessageDigest.getInstance("SHA-256").digest(canonicalText(records).toByteArray(Charsets.UTF_8))
        return bytes.joinToString("") { "%02x".format(it) }
    }
}
