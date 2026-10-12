package com.linnan.blindassist.capture

import android.os.SystemClock
import java.net.InetSocketAddress
import java.net.Socket
import java.net.URI
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/** One read-only TCP connection. Restart explicitly so a recording cannot hide reconnections. */
class TcpCnhCaptureSource(host: String, private val port: Int = 3335) : CnhCaptureSource {
    private val host = captureHost(host)
    override val transport: String = "tcp:$host:$port"
    private val started = AtomicBoolean()
    private val closed = AtomicBoolean()
    private val worker = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "cnh-tcp-capture").also { workerThread = it }
    }
    @Volatile private var workerThread: Thread? = null
    @Volatile private var socket: Socket? = null

    init { require(port in 1..65535) }

    override fun start(
        onRecord: (CnhCaptureRecord) -> Unit,
        onStatus: (CnhCaptureStatus) -> Unit,
        onRawChunk: ((ByteArray, Long) -> Unit)?
    ) {
        check(!closed.get()) { "Capture source is closed" }
        check(started.compareAndSet(false, true)) { "Capture source already started" }
        worker.execute {
            val decoder = CnhCaptureDecoder()
            var terminalStatus: CnhCaptureStatus? = null
            try {
                onStatus(CnhCaptureStatus(CnhConnectionState.CONNECTING, "连接 $host:$port"))
                Socket().use { connection ->
                    socket = connection
                    if (closed.get()) return@use
                    connection.connect(InetSocketAddress(host, port), 2500)
                    connection.soTimeout = 1000
                    connection.tcpNoDelay = true
                    if (closed.get()) return@use
                    onStatus(CnhCaptureStatus(CnhConnectionState.CONNECTED, "ToF TCP 已连接"))
                    CaptureReadAhead(connection.getInputStream(), SystemClock::elapsedRealtimeNanos,
                        "cnh-tcp-reader").use { readAhead ->
                        while (true) {
                            val chunk = readAhead.next() ?: break
                            onRawChunk?.invoke(chunk.bytes, chunk.receivedNs)
                            decoder.feed(chunk.bytes, chunk.receivedNs).forEach(onRecord)
                        }
                        terminalStatus = CnhCaptureStatus(CnhConnectionState.DISCONNECTED, "ToF 连接中断")
                    }
                }
            } catch (exception: Exception) {
                if (!closed.get()) terminalStatus = CnhCaptureStatus(CnhConnectionState.ERROR,
                    "ToF 采集失败：${exception.message ?: exception.javaClass.simpleName}")
            } finally {
                socket = null
                // The controller finalizes its writer on terminal status; publish the raw tail first.
                runCatching { decoder.finish()?.let(onRecord) }
                val status = if (closed.get()) {
                    CnhCaptureStatus(CnhConnectionState.STOPPED, "ToF 采集已停止")
                } else terminalStatus
                if (status != null) runCatching {
                    onStatus(status)
                }
                worker.shutdown()
            }
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        runCatching { socket?.close() }
        // Socket close cancels reads. Keep a terminal callback un-interrupted while
        // the controller drains the other streams and finalizes their journals.
        worker.shutdown()
        // Do not finalize the journal while accepted read-ahead chunks are still pending.
        if (Thread.currentThread() != workerThread) worker.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)
    }
}

/** Accept the phone hotspot/LAN address; avoid DNS or remote internet collection endpoints. */
internal fun captureHost(input: String): String {
    val trimmed = input.trim()
    val uri = URI(if (trimmed.contains("://")) trimmed else "http://$trimmed")
    require(uri.scheme == "http" && uri.userInfo == null && uri.rawQuery == null && uri.rawFragment == null &&
        (uri.rawPath.isNullOrEmpty() || uri.rawPath == "/") && (uri.port == -1 || uri.port == 80)) {
        "请输入局域网 ToF IP 地址"
    }
    val host = requireNotNull(uri.host) { "请输入 ToF IP 地址" }
    val parts = host.split('.')
    require(parts.size == 4 && parts.all {
        val number = it.toIntOrNull()
        number != null && number in 0..255 && (it == "0" || !it.startsWith('0'))
    }) { "请输入 IPv4 地址" }
    val octets = parts.map(String::toInt)
    require(octets[3] in 1..254 && (octets[0] == 10 ||
        octets[0] == 172 && octets[1] in 16..31 || octets[0] == 192 && octets[1] == 168)) {
        "ToF 必须连接手机热点或同一局域网"
    }
    return host
}
