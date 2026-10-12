package com.linnan.blindassist.capture

import android.os.SystemClock
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/** Read-only Atom BMI270 stream. Reconnect explicitly, preserving every received packet/tail. */
class TcpImuCaptureSource(host: String, private val port: Int = 3336) : ImuCaptureSource {
    private val host = try { captureHost(host) } catch (error: IllegalArgumentException) {
        throw IllegalArgumentException("请输入 Atom IMU 的局域网 IPv4 地址", error)
    }
    override val transport = "tcp:${this.host}:$port"
    private val started = AtomicBoolean()
    private val closed = AtomicBoolean()
    private val worker = Executors.newSingleThreadExecutor { task ->
        Thread(task, "imu-tcp-capture").also { workerThread = it }
    }
    @Volatile private var workerThread: Thread? = null
    @Volatile private var socket: Socket? = null

    init { require(port in 1..65535) }

    override fun start(
        onRecord: (ImuCaptureRecord) -> Unit,
        onStatus: (CnhCaptureStatus) -> Unit,
        onRawChunk: ((ByteArray, Long) -> Unit)?,
    ) {
        check(!closed.get()) { "IMU source is closed" }
        check(started.compareAndSet(false, true)) { "IMU source already started" }
        worker.execute {
            val decoder = ImuCaptureDecoder()
            var terminalStatus: CnhCaptureStatus? = null
            try {
                onStatus(CnhCaptureStatus(CnhConnectionState.CONNECTING, "连接 Atom IMU ${this.host}:$port"))
                Socket().use { connection ->
                    socket = connection
                    if (closed.get()) return@use
                    connection.connect(InetSocketAddress(this.host, port), 2500)
                    connection.soTimeout = 1000
                    connection.tcpNoDelay = true
                    if (closed.get()) return@use
                    onStatus(CnhCaptureStatus(CnhConnectionState.CONNECTED, "Atom IMU TCP 已连接"))
                    CaptureReadAhead(connection.getInputStream(), SystemClock::elapsedRealtimeNanos,
                        "imu-tcp-reader").use { readAhead ->
                        while (true) {
                            val chunk = readAhead.next() ?: break
                            onRawChunk?.invoke(chunk.bytes, chunk.receivedNs)
                            decoder.feed(chunk.bytes, chunk.receivedNs).forEach(onRecord)
                        }
                        terminalStatus = CnhCaptureStatus(CnhConnectionState.DISCONNECTED, "Atom IMU 连接中断")
                    }
                }
            } catch (error: Exception) {
                if (!closed.get()) terminalStatus = CnhCaptureStatus(CnhConnectionState.ERROR,
                    "Atom IMU 采集失败：${error.message ?: error.javaClass.simpleName}")
            } finally {
                socket = null
                // A terminal callback may finalize storage; deliver raw partial bytes first.
                runCatching { decoder.finish()?.let(onRecord) }
                val status = if (closed.get()) CnhCaptureStatus(CnhConnectionState.STOPPED, "Atom IMU 采集已停止")
                    else terminalStatus
                if (status != null) runCatching { onStatus(status) }
                worker.shutdown()
            }
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        runCatching { socket?.close() }
        // The socket close cancels reads without interrupting a terminal callback
        // that must still drain/finalize the other capture streams.
        worker.shutdown()
        // Source stop must drain accepted chunks before the controller writes its manifest.
        if (Thread.currentThread() != workerThread) worker.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)
    }
}
