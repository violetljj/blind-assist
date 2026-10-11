package com.linnan.blindassist.capture

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction

/** Decoder for the existing xiao-cnh-h3[-wifi]-v1 newline JSON contract. */
class CnhCaptureDecoder(private val maxLineBytes: Int = 65_536) {
    private val pending = ByteArrayOutputStream()
    private var lineNumber = 0L
    private var previousSequence: Long? = null
    private var previousMs: Long? = null
    private var lastReceiptNs = 0L

    init { require(maxLineBytes > 0) }

    /** The timestamp is the receipt of the chunk containing the final newline, not exposure. */
    fun feed(chunk: ByteArray, hostReceivedMonotonicNs: Long): List<CnhCaptureRecord> {
        require(hostReceivedMonotonicNs >= 0)
        lastReceiptNs = hostReceivedMonotonicNs
        val records = mutableListOf<CnhCaptureRecord>()
        for (byte in chunk) {
            pending.write(byte.toInt())
            if (byte == '\n'.code.toByte()) {
                records += decode(pending.toByteArray(), hostReceivedMonotonicNs)
                pending.reset()
            } else if (pending.size() >= maxLineBytes) {
                // Keep every byte in records even when an unsupported overlong packet arrives.
                records += CnhCaptureRecord(++lineNumber, pending.toByteArray(),
                    hostReceivedMonotonicNs, null, null, listOf("line_too_long"))
                pending.reset()
                discardingOversizedLine = true
            }
        }
        return records
    }

    private var discardingOversizedLine = false

    /** Preserve an interrupted packet as raw evidence; never promote it into a complete frame. */
    fun finish(): CnhCaptureRecord? {
        if (pending.size() == 0) return null
        val bytes = pending.toByteArray()
        pending.reset()
        return CnhCaptureRecord(++lineNumber, bytes, lastReceiptNs, null, null,
            listOf(if (discardingOversizedLine) "line_too_long_tail" else "truncated_line"))
    }

    private fun decode(bytes: ByteArray, receivedNs: Long): CnhCaptureRecord {
        val number = ++lineNumber
        if (discardingOversizedLine) {
            discardingOversizedLine = false
            return CnhCaptureRecord(number, bytes, receivedNs, null, null,
                listOf("line_too_long_tail"))
        }
        var kind: String? = null
        val issues = mutableListOf<String>()
        var frame: CnhCaptureFrame? = null
        try {
            val text = Charsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes)).toString()
            if (text.isBlank()) return CnhCaptureRecord(number, bytes, receivedNs, null, null, emptyList())
            val tokener = JSONTokener(text)
            val value = tokener.nextValue()
            require(value is JSONObject && tokener.nextClean() == '\u0000') { "not_one_json_object" }
            kind = value.opt("type") as? String
            require(!kind.isNullOrBlank()) { "missing_record_type" }
            when (kind) {
                "cnh_frame" -> frame = validateFrame(value)
                "frame" -> issues += "range_only_frame_without_cnh"
            }
        } catch (exception: Exception) {
            issues += "invalid_packet:${exception.message ?: exception.javaClass.simpleName}"
        }
        frame?.let {
            val previous = previousSequence
            if (previous != null) {
                val step = (it.sequence - previous) and UINT32_MAX
                when {
                    step == 0L -> issues += "duplicate_sequence"
                    step == 1L -> Unit
                    step < 0x80000000L -> issues += "sequence_gap:${step - 1}"
                    else -> issues += "sequence_reset_or_reorder"
                }
            }
            previousMs?.let { last ->
                val elapsed = (it.deviceMs - last) and UINT32_MAX
                if (elapsed == 0L || elapsed >= 0x80000000L) issues += "device_clock_discontinuity"
            }
            previousSequence = it.sequence
            previousMs = it.deviceMs
        }
        return CnhCaptureRecord(number, bytes, receivedNs, kind, frame, issues)
    }

    private fun validateFrame(obj: JSONObject): CnhCaptureFrame {
        require(integer(obj.opt("rows"), "rows", 8, 8) == 8L)
        require(integer(obj.opt("cols"), "cols", 8, 8) == 8L)
        require(integer(obj.opt("bins"), "bins", 16, 16) == 16L)
        val header = longVector(obj.opt("cnh_header"), "cnh_header", 5, 0, UINT32_MAX)
        require((header[4] and 0xffffL) == 64L && (header[4] ushr 16) == 16L) { "cnh_header_layout" }
        return CnhCaptureFrame(
            sequence = integer(obj.opt("seq"), "seq", 0, UINT32_MAX),
            deviceMs = integer(obj.opt("ms"), "ms", 0, UINT32_MAX),
            readUs = integer(obj.opt("read_us"), "read_us", 0, UINT32_MAX),
            stream = integer(obj.opt("stream"), "stream", 0, 255).toInt(),
            distanceMm = vector(obj.opt("distance_mm"), "distance_mm", 64, -32768, 32767),
            targetStatus = vector(obj.opt("target_status"), "target_status", 64, 0, 255),
            targetCount = vector(obj.opt("nb_target"), "nb_target", 64, 0, 255),
            histRaw = matrix(obj.opt("hist_raw"), "hist_raw", Int.MIN_VALUE.toLong(), Int.MAX_VALUE.toLong()),
            histScaler = matrix(obj.opt("hist_scaler"), "hist_scaler", -128, 127),
            ambientRaw = vector(obj.opt("ambient_raw"), "ambient_raw", 64, Int.MIN_VALUE.toLong(), Int.MAX_VALUE.toLong()),
            ambientScaler = vector(obj.opt("ambient_scaler"), "ambient_scaler", 64, -128, 127),
            cnhHeader = header
        )
    }

    private fun matrix(value: Any?, name: String, low: Long, high: Long): List<List<Int>> {
        require(value is JSONArray && value.length() == 64) { "$name:expected_64_zones" }
        return (0 until 64).map { vector(value.opt(it), "$name[$it]", 16, low, high) }
    }

    private fun vector(value: Any?, name: String, length: Int, low: Long, high: Long): List<Int> =
        longVector(value, name, length, low, high).map(Long::toInt)

    private fun longVector(value: Any?, name: String, length: Int, low: Long, high: Long): List<Long> {
        require(value is JSONArray && value.length() == length) { "$name:expected_$length" }
        return (0 until length).map { integer(value.opt(it), "$name[$it]", low, high) }
    }

    private fun integer(value: Any?, name: String, low: Long, high: Long): Long {
        // Reject JSON strings, booleans, fractions and floating values rather than coerce them.
        require(value is Int || value is Long) { "$name:expected_integer" }
        val result = (value as Number).toLong()
        require(result in low..high) { "$name:out_of_range" }
        return result
    }

    private companion object { const val UINT32_MAX = 0xffffffffL }
}
