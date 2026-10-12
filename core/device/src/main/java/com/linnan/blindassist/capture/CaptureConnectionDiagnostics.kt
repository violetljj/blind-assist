package com.linnan.blindassist.capture

import java.io.EOFException
import java.net.ConnectException
import java.net.NoRouteToHostException
import java.net.SocketException
import java.net.SocketTimeoutException

/** Observations only: socket failures do not establish a Wi-Fi or device power failure. */
enum class CaptureConnectionReason {
    ATTEMPT, TRANSPORT_CONNECTED, NO_VALID_FRAMES, TRANSPORT_CLOSED, SOCKET_TIMEOUT,
    CONNECTION_FAILED, NETWORK_UNREACHABLE, SOCKET_ERROR, SOURCE_ERROR,
    BOOT_ID_CHANGED, RECOVERED, STOPPED,
}

data class CaptureConnectionEvent(
    val stream: String,
    val reason: CaptureConnectionReason,
    val elapsedRealtimeNs: Long,
    val message: String,
    val attempts: Int,
    val recoveryElapsedMs: Long? = null,
    val wallClockMs: Long = System.currentTimeMillis(),
)

data class CaptureConnectionDiagnostics(
    val latest: List<CaptureConnectionEvent> = emptyList(),
    val history: List<CaptureConnectionEvent> = emptyList(),
    val persistenceError: String? = null,
)

internal fun captureFailureReason(error: Exception): CaptureConnectionReason = when (error) {
    is SocketTimeoutException -> CaptureConnectionReason.SOCKET_TIMEOUT
    is NoRouteToHostException -> CaptureConnectionReason.NETWORK_UNREACHABLE
    is ConnectException -> CaptureConnectionReason.CONNECTION_FAILED
    is EOFException -> CaptureConnectionReason.TRANSPORT_CLOSED
    is SocketException -> CaptureConnectionReason.SOCKET_ERROR
    else -> CaptureConnectionReason.SOURCE_ERROR
}

/** Accessed under the controller lock. Receipt clocks remain untouched. */
internal class CaptureConnectionTracker(
    previous: List<CaptureConnectionEvent> = emptyList(),
    private val emit: (CaptureConnectionEvent) -> Unit = {},
) {
    private class Stream {
        var connectedAt: Long? = null
        var lastFrame: Long? = null
        var problemAt: Long? = null
        var attempts = 0
        var bootId: String? = null
        var transport: String? = null
        var stalled = false
        var awaitingFrame = false
        var latest: CaptureConnectionEvent? = null
    }
    private val streams = linkedMapOf<String, Stream>()
    private val history = ArrayDeque<CaptureConnectionEvent>().apply { addAll(previous.takeLast(64)) }
    private fun add(key: String, state: Stream, reason: CaptureConnectionReason, now: Long,
        message: String, recoveryMs: Long? = null) {
        val event = CaptureConnectionEvent(key, reason, now, message.take(256), state.attempts, recoveryMs)
        state.latest = event
        if (history.size == 64) history.removeFirst()
        history.addLast(event)
        emit(event)
    }
    fun attempt(key: String, transport: String, now: Long) {
        val state = streams.getOrPut(key) { Stream() }
        if (state.transport != transport) {
            state.bootId = null
            state.problemAt = null
            state.attempts = 0
        }
        state.transport = transport
        state.attempts++
        if (state.problemAt == null) state.problemAt = now
        state.connectedAt = null
        state.lastFrame = null
        state.stalled = false
        state.awaitingFrame = true
        add(key, state, CaptureConnectionReason.ATTEMPT, now, "正在连接；第 ${state.attempts} 次尝试")
    }
    fun factoryFailed(key: String, error: Exception, now: Long) {
        val state = streams.getOrPut(key) { Stream() }
        state.attempts++
        if (state.problemAt == null) state.problemAt = now
        add(key, state, captureFailureReason(error), now, "连接初始化失败：${error.message ?: error.javaClass.simpleName}")
    }
    fun status(key: String, status: CnhCaptureStatus, now: Long) {
        if (status.state == CnhConnectionState.STOPPED && key !in streams) return
        val state = streams.getOrPut(key) { Stream() }
        when (status.state) {
            CnhConnectionState.CONNECTING -> Unit // attempt is recorded before source start
            CnhConnectionState.CONNECTED -> {
                state.connectedAt = now
                add(key, state, CaptureConnectionReason.TRANSPORT_CONNECTED, now, "连接已建立，等待有效数据")
            }
            CnhConnectionState.ERROR, CnhConnectionState.DISCONNECTED -> {
                state.connectedAt = null
                if (state.problemAt == null) state.problemAt = now
                add(key, state, status.reason ?: CaptureConnectionReason.SOURCE_ERROR, now, status.message)
            }
            CnhConnectionState.STOPPED -> {
                state.connectedAt = null
                state.awaitingFrame = false
                // Keep the observed failure visible when sibling streams close during finalization.
                if (state.latest?.reason in setOf(CaptureConnectionReason.ATTEMPT, CaptureConnectionReason.TRANSPORT_CONNECTED)) {
                    state.problemAt = null
                }
                if (state.problemAt == null && state.latest?.reason != CaptureConnectionReason.STOPPED) {
                    add(key, state, CaptureConnectionReason.STOPPED, now, "采集连接已停止")
                }
            }
        }
    }
    /** A deliberate stop ends the retry episode; historical failures remain in the journal. */
    fun intentionalStop(now: Long) {
        streams.forEach { (key, state) ->
            state.connectedAt = null
            state.lastFrame = null
            state.problemAt = null
            state.awaitingFrame = false
            state.stalled = false
            if (state.latest?.reason != CaptureConnectionReason.STOPPED) {
                add(key, state, CaptureConnectionReason.STOPPED, now, "已主动结束采集连接")
            }
            state.attempts = 0
        }
    }

    fun frame(key: String, now: Long, bootId: String? = null) {
        val state = streams.getOrPut(key) { Stream() }
        val knownBoot = bootId?.takeIf { it.isNotBlank() }
        if (knownBoot != null && state.bootId != null && knownBoot != state.bootId) {
            add(key, state, CaptureConnectionReason.BOOT_ID_CHANGED, now,
                "设备启动标识已变化；观察到新的启动周期，不能据此确定此前断连原因")
        }
        if (knownBoot != null) state.bootId = knownBoot
        state.lastFrame = now
        if (state.awaitingFrame || state.stalled || state.problemAt != null) {
            val duration = state.problemAt?.let { ((now - it).coerceAtLeast(0)) / 1_000_000 }
            add(key, state, CaptureConnectionReason.RECOVERED, now, "已收到有效数据", duration)
            state.problemAt = null
            state.attempts = 0
            state.stalled = false
            state.awaitingFrame = false
        }
    }
    fun check(now: Long) {
        streams.forEach { (key, state) ->
            val since = state.lastFrame ?: state.connectedAt
            if (state.connectedAt != null && since != null && now - since > 2_000_000_000L && !state.stalled) {
                state.stalled = true
                if (state.problemAt == null) state.problemAt = since
                add(key, state, CaptureConnectionReason.NO_VALID_FRAMES, now,
                    "连接尚未报告断开，但超过 2 秒未收到有效数据；原因未确定")
            }
        }
    }
    fun snapshot() = CaptureConnectionDiagnostics(streams.values.mapNotNull { it.latest }, history.toList())
}
