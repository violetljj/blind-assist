package com.linnan.blindassist.capture

/** Acquisition callbacks run on the source worker. Store raw bytes before doing slower work. */
interface CnhCaptureSource : AutoCloseable {
    val transport: String

    fun start(
        onRecord: (CnhCaptureRecord) -> Unit,
        onStatus: (CnhCaptureStatus) -> Unit,
        onRawChunk: ((ByteArray, Long) -> Unit)? = null
    )
}

enum class CnhConnectionState { CONNECTING, CONNECTED, DISCONNECTED, ERROR, STOPPED }

data class CnhCaptureStatus(
    val state: CnhConnectionState,
    val message: String,
    val reason: CaptureConnectionReason? = null,
)

/** Device milliseconds and host elapsed realtime belong to independent, unmapped clocks. */
data class CnhCaptureRecord(
    val lineNumber: Long,
    val rawLine: ByteArray,
    val hostReceivedMonotonicNs: Long,
    val kind: String?,
    val frame: CnhCaptureFrame?,
    val issues: List<String>
)

/** Native zone/bin order and signed histogram values are preserved without orientation changes. */
data class CnhCaptureFrame(
    val sequence: Long,
    val deviceMs: Long,
    val readUs: Long,
    val stream: Int,
    val distanceMm: List<Int>,
    val targetStatus: List<Int>,
    val targetCount: List<Int>,
    val histRaw: List<List<Int>>,
    val histScaler: List<List<Int>>,
    val ambientRaw: List<Int>,
    val ambientScaler: List<Int>,
    val cnhHeader: List<Long>,
    val rows: Int = 8,
    val cols: Int = 8,
    val bins: Int = 16
) {
    val validCellCount: Int
        get() = distanceMm.indices.count {
            distanceMm[it] > 0 && targetStatus[it] == 5 && targetCount[it] > 0
        }
}
