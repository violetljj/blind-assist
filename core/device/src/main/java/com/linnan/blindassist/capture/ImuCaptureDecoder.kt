package com.linnan.blindassist.capture

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction

/** Callbacks run on the source worker. Device microseconds are not Android clock time. */
interface ImuCaptureSource : AutoCloseable {
    val transport: String
    fun start(
        onRecord: (ImuCaptureRecord) -> Unit,
        onStatus: (CnhCaptureStatus) -> Unit,
        onRawChunk: ((ByteArray, Long) -> Unit)? = null,
    )
}

data class ImuCaptureRecord(
    val lineNumber: Long,
    val rawLine: ByteArray,
    val hostReceivedMonotonicNs: Long,
    val kind: String?,
    val frame: ImuCaptureFrame?,
    val issues: List<String>,
)

/** Native BMI270 axes. Pitch/roll are estimates; this frame is NOT_TOF_OPTICAL. */
data class ImuCaptureFrame(
    val sequence: Long,
    val deviceUs: Long,
    val deviceId: String,
    val bootId: String,
    val accelRaw: List<Int>,
    val gyroRaw: List<Int>,
    val accelG: List<Double>,
    val gyroDps: List<Double>,
    val pitchDeg: Double,
    val rollDeg: Double,
    val accelCorrection: Boolean,
    val dtUs: Long,
    val readUs: Long,
    val sampleGap: Boolean,
    val gravityUnit: List<Double>? = null,
)

/** Decoder for atom-bmi270-v1 TCP 3336 NDJSON. Invalid and partial bytes are retained. */
class ImuCaptureDecoder(private val maxLineBytes: Int = 16_384) {
    private val pending = ByteArrayOutputStream()
    private var lineNumber = 0L
    private var lastReceiptNs = 0L
    private var discardingOversizedLine = false
    private var previousSequence: Long? = null
    private var previousUs: Long? = null
    private var previousBoot: String? = null

    init { require(maxLineBytes > 0) }

    /** Receipt of the chunk containing the final newline, not IMU sampling time. */
    fun feed(chunk: ByteArray, hostReceivedMonotonicNs: Long): List<ImuCaptureRecord> {
        require(hostReceivedMonotonicNs >= 0)
        lastReceiptNs = hostReceivedMonotonicNs
        val records = mutableListOf<ImuCaptureRecord>()
        for (byte in chunk) {
            pending.write(byte.toInt())
            if (byte == '\n'.code.toByte()) {
                records += decode(pending.toByteArray(), hostReceivedMonotonicNs)
                pending.reset()
            } else if (pending.size() >= maxLineBytes) {
                records += ImuCaptureRecord(++lineNumber, pending.toByteArray(), hostReceivedMonotonicNs,
                    null, null, listOf("line_too_long"))
                pending.reset()
                discardingOversizedLine = true
            }
        }
        return records
    }

    fun finish(): ImuCaptureRecord? {
        if (pending.size() == 0) return null
        val bytes = pending.toByteArray()
        pending.reset()
        return ImuCaptureRecord(++lineNumber, bytes, lastReceiptNs, null, null,
            listOf(if (discardingOversizedLine) "line_too_long_tail" else "truncated_line"))
    }

    private fun decode(bytes: ByteArray, receivedNs: Long): ImuCaptureRecord {
        val number = ++lineNumber
        if (discardingOversizedLine) {
            discardingOversizedLine = false
            return ImuCaptureRecord(number, bytes, receivedNs, null, null, listOf("line_too_long_tail"))
        }
        var kind: String? = null
        var frame: ImuCaptureFrame? = null
        val issues = mutableListOf<String>()
        try {
            val text = Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString()
            if (text.isBlank()) return ImuCaptureRecord(number, bytes, receivedNs, null, null, emptyList())
            val tokener = JSONTokener(text)
            val obj = tokener.nextValue()
            require(obj is JSONObject && tokener.nextClean() == '\u0000') { "not_one_json_object" }
            kind = obj.opt("type") as? String
            require(!kind.isNullOrBlank()) { "missing_record_type" }
            if (kind == "imu_frame") {
                require(integer(obj.opt("schema_version"), "schema_version", 1, 1) == 1L)
                if (!boolean(obj.opt("valid"), "valid")) issues += "invalid_imu_read"
                else frame = validateFrame(obj)
            }
        } catch (error: Exception) {
            issues += "invalid_packet:${error.message ?: error.javaClass.simpleName}"
        }
        frame?.let {
            if (previousBoot != null && previousBoot != it.bootId) {
                issues += "boot_id_changed"
                previousSequence = null
                previousUs = null
            }
            previousSequence?.let { last ->
                val step = (it.sequence - last) and UINT32_MAX
                when {
                    step == 0L -> issues += "duplicate_sequence"
                    step == 1L -> Unit
                    step < 0x80000000L -> issues += "sequence_gap:${step - 1}"
                    else -> issues += "sequence_reset_or_reorder"
                }
            }
            if (previousUs?.let { last -> it.deviceUs <= last } == true) issues += "device_clock_discontinuity"
            if (it.sampleGap) issues += "device_sample_gap"
            previousSequence = it.sequence
            previousUs = it.deviceUs
            previousBoot = it.bootId
        }
        return ImuCaptureRecord(number, bytes, receivedNs, kind, frame, issues)
    }

    private fun validateFrame(obj: JSONObject) = ImuCaptureFrame(
        sequence = integer(obj.opt("seq"), "seq", 0, UINT32_MAX),
        deviceUs = integer(obj.opt("device_us"), "device_us", 0, Long.MAX_VALUE),
        deviceId = identifier(obj.opt("device_id"), "device_id"),
        bootId = identifier(obj.opt("boot_id"), "boot_id"),
        accelRaw = rawVector(obj.opt("accel_raw"), "accel_raw"),
        gyroRaw = rawVector(obj.opt("gyro_raw"), "gyro_raw"),
        accelG = vector(obj.opt("accel_g"), "accel_g"),
        gyroDps = vector(obj.opt("gyro_dps"), "gyro_dps"),
        pitchDeg = finiteNumber(obj.opt("pitch_deg"), "pitch_deg"),
        rollDeg = finiteNumber(obj.opt("roll_deg"), "roll_deg"),
        accelCorrection = boolean(obj.opt("accel_correction"), "accel_correction"),
        dtUs = integer(obj.opt("dt_us"), "dt_us", 0, UINT32_MAX),
        readUs = integer(obj.opt("read_us"), "read_us", 0, UINT32_MAX),
        sampleGap = boolean(obj.opt("sample_gap"), "sample_gap"),
        gravityUnit = if (obj.has("gravity_unit") && !obj.isNull("gravity_unit"))
            vector(obj.opt("gravity_unit"), "gravity_unit") else null,
    )

    private fun identifier(value: Any?, name: String): String {
        require(value is String && value.isNotBlank() && value.length <= 128) { "$name:invalid_identifier" }
        return value
    }

    private fun rawVector(value: Any?, name: String): List<Int> {
        require(value is JSONArray && value.length() == 3) { "$name:expected_three_axes" }
        return (0..2).map { integer(value.opt(it), "$name[$it]", -32768, 32767).toInt() }
    }

    private fun vector(value: Any?, name: String): List<Double> {
        require(value is JSONArray && value.length() == 3) { "$name:expected_three_axes" }
        return (0..2).map { finiteNumber(value.opt(it), "$name[$it]") }
    }

    private fun finiteNumber(value: Any?, name: String): Double {
        require(value is Number && value.toDouble().isFinite()) { "$name:expected_finite_number" }
        return value.toDouble()
    }

    private fun boolean(value: Any?, name: String): Boolean {
        require(value is Boolean) { "$name:expected_boolean" }
        return value
    }

    private fun integer(value: Any?, name: String, low: Long, high: Long): Long {
        require(value is Int || value is Long) { "$name:expected_integer" }
        return (value as Number).toLong().also { require(it in low..high) { "$name:out_of_range" } }
    }

    private companion object { const val UINT32_MAX = 0xffffffffL }
}
