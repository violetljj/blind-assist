package com.linnan.blindassist.goalcapture

import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest
import java.time.Instant

internal const val CAPTURE_PLAN_SCHEMA = "blindassist_p1_pa3_prospective_first_person_capture_plan_v1"
internal const val CAPTURE_RECEIPT_SCHEMA = "blindassist_p1_pa3_prospective_first_person_capture_v1"
internal const val SOURCE_ROLE = "PROSPECTIVE_FIRST_PERSON_PHYSICAL_CAPTURE_AFTER_GOAL"
internal const val CAPTURE_INSTRUCTION = "APPROACH_NAMED_BUILDING_AND_STOP_AFTER_ENTRANCE_IS_IN_VIEW_V1"
internal const val FRAME_SELECTION_RULE = "FIXED_OFFSETS_FROM_VIDEO_END_NO_PIXEL_OR_TRUTH_SELECTION"
internal val FRAME_OFFSETS = listOf(-2.5, -1.5, -0.5)

internal data class CapturePlanEpisode(
    val episodeId: String,
    val goalTextOriginal: String,
    val mediaRelativePath: String,
)

internal data class CapturePlan(
    val bodySha256: String,
    val goalReceiptBodySha256: String,
    val armedAt: Instant,
    val episodes: List<CapturePlanEpisode>,
    val originalJson: String,
)

internal data class CompletedCapture(
    val episodeId: String,
    val captureStartedAt: Instant,
    val captureCompletedAt: Instant,
    val mediaPath: String,
    val mediaSha256: String,
    val width: Int,
    val height: Int,
    val durationSeconds: Double,
)

internal object CanonicalJson {
    fun encode(value: Any?): String = buildString {
        appendValue(value)
        append('\n')
    }

    fun sha256(value: Any?): String = sha256Bytes(encode(value).toByteArray(Charsets.UTF_8))

    fun sha256Bytes(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it) }

    private fun StringBuilder.appendValue(value: Any?) {
        when (value) {
            null, JSONObject.NULL -> append("null")
            is String -> appendQuoted(value)
            is Boolean -> append(if (value) "true" else "false")
            is Byte, is Short, is Int, is Long -> append(value.toString())
            is Float -> appendFinite(value.toDouble())
            is Double -> appendFinite(value)
            is Map<*, *> -> {
                val keys = value.keys.map {
                    require(it is String) { "canonical JSON object keys must be strings" }
                    it
                }.sorted()
                append('{')
                keys.forEachIndexed { index, key ->
                    if (index > 0) append(',')
                    appendQuoted(key)
                    append(':')
                    appendValue(value[key])
                }
                append('}')
            }
            is Iterable<*> -> {
                append('[')
                value.forEachIndexed { index, child ->
                    if (index > 0) append(',')
                    appendValue(child)
                }
                append(']')
            }
            else -> error("unsupported canonical JSON type: ${value.javaClass.name}")
        }
    }

    private fun StringBuilder.appendFinite(value: Double) {
        require(value.isFinite()) { "canonical JSON numbers must be finite" }
        append(value.toString())
    }

    private fun StringBuilder.appendQuoted(value: String) {
        append('"')
        value.forEach { character ->
            when (character) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (character.code < 0x20) append("\\u%04x".format(character.code)) else append(character)
            }
        }
        append('"')
    }
}

internal object CapturePlanParser {
    private val safeEpisodeId = Regex("^[a-z0-9][a-z0-9-]{2,79}$")
    private val sha256 = Regex("^[0-9a-f]{64}$")

    fun parse(text: String): CapturePlan {
        val root = JSONObject(text)
        val declaredHash = root.requireString("capture_plan_body_sha256")
        require(sha256.matches(declaredHash)) { "capture plan hash must be lowercase SHA-256" }
        val body = root.toKotlinMap().toMutableMap().also { it.remove("capture_plan_body_sha256") }
        require(CanonicalJson.sha256(body) == declaredHash) { "capture plan body hash mismatch" }
        require(root.requireString("schema_version") == CAPTURE_PLAN_SCHEMA)
        require(root.requireString("source_role") == SOURCE_ROLE)
        require(root.requireString("capture_instruction_id") == CAPTURE_INSTRUCTION)
        require(root.requireString("frame_selection_rule") == FRAME_SELECTION_RULE)
        require(root.requireString("truth_state_at_arming") == "NOT_CREATED")
        require(root.requireString("provider_state_at_arming") == "NOT_STARTED")
        require(root.requireString("capture_state_at_arming") == "NOT_STARTED")
        require(root.getJSONArray("frame_offsets_from_end_seconds").toKotlinList() == FRAME_OFFSETS)
        val episodeRows = root.getJSONArray("episodes")
        require(episodeRows.length() >= 5) { "capture plan requires at least five episodes" }
        require(root.getInt("episode_count") == episodeRows.length()) { "capture plan episode count drift" }
        val episodes = (0 until episodeRows.length()).map { index ->
            val row = episodeRows.getJSONObject(index)
            val episodeId = row.requireString("episode_id")
            val mediaPath = row.requireString("media_relative_path")
            require(safeEpisodeId.matches(episodeId)) { "unsafe episode id" }
            require(mediaPath == "$episodeId.mp4") { "media path differs from frozen naming rule" }
            require(row.requireString("camera_view") == "FIRST_PERSON_FORWARD")
            require(row.getBoolean("continuous_capture"))
            CapturePlanEpisode(episodeId, row.requireString("goal_text_original"), mediaPath)
        }
        require(episodes.map { it.episodeId }.toSet().size == episodes.size) { "duplicate capture episode" }
        val goalReceiptBodySha256 = root.requireString("goal_receipt_body_sha256")
        require(sha256.matches(goalReceiptBodySha256)) { "goal receipt hash must be lowercase SHA-256" }
        return CapturePlan(
            bodySha256 = declaredHash,
            goalReceiptBodySha256 = goalReceiptBodySha256,
            armedAt = Instant.parse(root.requireString("armed_at_utc")),
            episodes = episodes,
            originalJson = text,
        )
    }

    private fun JSONObject.requireString(key: String): String = getString(key).trim().also { require(it.isNotEmpty()) }
}

internal object CaptureReceiptBuilder {
    fun build(plan: CapturePlan, captures: List<CompletedCapture>, receiptCreatedAt: Instant): Map<String, Any?> {
        require(captures.size == plan.episodes.size) { "partial capture roster cannot produce a receipt" }
        require(captures.map { it.episodeId } == plan.episodes.map { it.episodeId }) { "capture order or identity drift" }
        require(captures.map { it.mediaSha256 }.toSet().size == captures.size) { "physical capture media was reused" }
        val episodes = captures.mapIndexed { index, capture ->
            val planned = plan.episodes[index]
            require(capture.mediaPath == planned.mediaRelativePath)
            require(plan.armedAt < capture.captureStartedAt)
            require(capture.captureStartedAt < capture.captureCompletedAt)
            require(capture.captureCompletedAt <= receiptCreatedAt)
            require(capture.width > 0 && capture.height > 0)
            require(capture.durationSeconds in 3.0..45.0)
            linkedMapOf<String, Any?>(
                "episode_id" to capture.episodeId,
                "capture_started_at_utc" to capture.captureStartedAt.toString(),
                "capture_completed_at_utc" to capture.captureCompletedAt.toString(),
                "media_path" to capture.mediaPath,
                "media_sha256" to capture.mediaSha256,
                "width" to capture.width,
                "height" to capture.height,
                "duration_seconds" to capture.durationSeconds,
                "camera_view" to "FIRST_PERSON_FORWARD",
                "continuous_capture" to true,
            )
        }
        val body = linkedMapOf<String, Any?>(
            "schema_version" to CAPTURE_RECEIPT_SCHEMA,
            "goal_receipt_body_sha256" to plan.goalReceiptBodySha256,
            "capture_plan_body_sha256" to plan.bodySha256,
            "source_role" to SOURCE_ROLE,
            "recorder_authority" to "DEVICE_OWNED_CONTINUOUS_VIDEO_RECORDER",
            "capture_instruction_id" to CAPTURE_INSTRUCTION,
            "truth_state_at_capture" to "NOT_CREATED",
            "provider_state_at_capture" to "NOT_STARTED",
            "continuous_capture_required" to true,
            "frame_selection_rule" to FRAME_SELECTION_RULE,
            "frame_offsets_from_end_seconds" to FRAME_OFFSETS,
            "receipt_created_at_utc" to receiptCreatedAt.toString(),
            "episode_count" to episodes.size,
            "episodes" to episodes,
        )
        return LinkedHashMap(body).also { it["capture_body_sha256"] = CanonicalJson.sha256(body) }
    }
}

private fun JSONObject.toKotlinMap(): Map<String, Any?> = keys().asSequence().associateWith { key -> get(key).toKotlinValue() }

private fun JSONArray.toKotlinList(): List<Any?> = (0 until length()).map { index -> get(index).toKotlinValue() }

private fun Any?.toKotlinValue(): Any? = when (this) {
    null, JSONObject.NULL -> null
    is JSONObject -> toKotlinMap()
    is JSONArray -> toKotlinList()
    else -> this
}
