package com.linnan.blindassist

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.capture.CnhCaptureStatus
import com.linnan.blindassist.capture.CnhConnectionState
import com.linnan.blindassist.capture.TcpCnhCaptureSource
import com.linnan.blindassist.capture.TcpImuCaptureSource
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Real Android sockets on the phone itself; no sensor endpoint or saved session is touched. */
@RunWith(AndroidJUnit4::class)
class CaptureTcpDrainDeviceTest {
    @Test
    fun tofCloseDrainsReadAheadAndPartialTailBeforeStopped() = verifyDrain(imu = false)

    @Test
    fun imuCloseDrainsReadAheadAndPartialTailBeforeStopped() = verifyDrain(imu = true)

    private fun verifyDrain(imu: Boolean) {
        val address = Collections.list(NetworkInterface.getNetworkInterfaces()).asSequence()
            .filter { it.isUp }
            .flatMap { Collections.list(it.inetAddresses).asSequence() }
            .filterIsInstance<Inet4Address>()
            .firstOrNull { it.isSiteLocalAddress && (it.address[3].toInt() and 255) in 1..254 }
        assumeTrue("A private IPv4 interface is required by capture endpoint policy", address != null)
        val server = ServerSocket(0, 1, requireNotNull(address)).apply { soTimeout = 5000 }
        val firstCallback = CountDownLatch(1)
        val releaseCallback = CountDownLatch(1)
        val sent = CountDownLatch(1)
        val releaseServer = CountDownLatch(1)
        val closeEntered = CountDownLatch(1)
        val closeDone = CountDownLatch(1)
        val serverFailure = AtomicReference<Throwable?>()
        val closeFailure = AtomicReference<Throwable?>()
        val accepted = AtomicReference<Socket?>()
        val callbackThread = AtomicReference<Thread?>()
        val raw = ByteArrayOutputStream()
        val records = mutableListOf<ByteArray>()
        val events = mutableListOf<String>()
        val receipts = mutableListOf<Long>()
        val first = "{\"type\":\"test_start\"}\r\n".toByteArray()
        val rest = buildString {
            repeat(20) { append("{\"type\":\"test_record\",\"seq\":$it,\"payload\":\"${"x".repeat(80)}\"}\n") }
            append("{\"type\":\"unfinished_tail\"")
        }.toByteArray()
        val expected = first + rest
        val source: AutoCloseable = if (imu) TcpImuCaptureSource(address.hostAddress!!, server.localPort)
            else TcpCnhCaptureSource(address.hostAddress!!, server.localPort)
        val workerNames = if (imu) setOf("imu-tcp-reader", "imu-tcp-capture")
            else setOf("cnh-tcp-reader", "cnh-tcp-capture")
        val existingThreads = Thread.getAllStackTraces().keys.toSet()
        val producer = Thread({
            try {
                server.accept().use { connection ->
                    accepted.set(connection)
                    connection.tcpNoDelay = true
                    connection.getOutputStream().write(first)
                    check(firstCallback.await(5, TimeUnit.SECONDS)) { "First callback did not arrive" }
                    connection.getOutputStream().write(rest)
                    // EOF lets the test observe that the reader accepted every byte while
                    // the decoder/writer callback is still deliberately stalled.
                    connection.shutdownOutput()
                    sent.countDown()
                    check(releaseServer.await(10, TimeUnit.SECONDS)) { "Server cleanup not released" }
                }
            } catch (error: Throwable) {
                serverFailure.set(error)
                sent.countDown()
            }
        }, "capture-drain-test-server")
        val closer = Thread({
            closeEntered.countDown()
            try { source.close() } catch (error: Throwable) { closeFailure.set(error) }
            finally { closeDone.countDown() }
        }, "capture-drain-test-close")
        val onRecord: (ByteArray) -> Unit = { bytes ->
            callbackThread.compareAndSet(null, Thread.currentThread())
            if (records.isEmpty()) {
                firstCallback.countDown()
                check(releaseCallback.await(10, TimeUnit.SECONDS)) { "Writer stall was not released" }
            }
            records += bytes
            events += "record"
        }
        val onStatus: (CnhCaptureStatus) -> Unit = { events += it.state.name }
        val onRaw: (ByteArray, Long) -> Unit = { bytes, receipt ->
            raw.write(bytes)
            receipts += receipt
        }
        try {
            producer.start()
            when (source) {
                is TcpImuCaptureSource -> source.start({ onRecord(it.rawLine) }, onStatus, onRaw)
                is TcpCnhCaptureSource -> source.start({ onRecord(it.rawLine) }, onStatus, onRaw)
            }
            assertTrue("No first callback", firstCallback.await(5, TimeUnit.SECONDS))
            // The reader exists before the first callback; it may already have reached EOF.
            val reader = Thread.getAllStackTraces().keys.firstOrNull {
                it !in existingThreads && it.name == workerNames.first()
            }
            assertTrue("Server did not finish sending", sent.await(5, TimeUnit.SECONDS))
            serverFailure.get()?.let { throw AssertionError("Local TCP server failed", it) }
            reader?.join(5000)
            assertTrue("Socket pump is blocked behind the writer", reader == null || !reader.isAlive)
            closer.start()
            assertTrue(closeEntered.await(5, TimeUnit.SECONDS))
            assertFalse("close returned before the blocked writer drained", closeDone.await(100, TimeUnit.MILLISECONDS))
            releaseCallback.countDown()
            assertTrue("Source close did not drain", closeDone.await(5, TimeUnit.SECONDS))
            closeFailure.get()?.let { throw AssertionError("Source close failed", it) }
            assertArrayEquals("Raw callbacks lost or reordered bytes", expected, raw.toByteArray())
            assertArrayEquals("Decoder lost or reordered bytes/tail", expected,
                ByteArrayOutputStream().apply { records.forEach { write(it) } }.toByteArray())
            assertEquals(22, records.size)
            assertArrayEquals("Unterminated tail must be delivered", "{\"type\":\"unfinished_tail\"".toByteArray(), records.last())
            assertEquals(CnhConnectionState.STOPPED.name, events.last())
            assertEquals(1, events.count { it == CnhConnectionState.STOPPED.name })
            assertFalse(events.contains(CnhConnectionState.ERROR.name))
            assertFalse(events.contains(CnhConnectionState.DISCONNECTED.name))
            assertTrue(receipts.all { it > 0 } && receipts.zipWithNext().all { (a, b) -> b >= a })
            callbackThread.get()?.join(1000)
            assertTrue("Capture workers leaked", Thread.getAllStackTraces().keys.none {
                it !in existingThreads && it.name in workerNames && it.isAlive
            })
        } finally {
            releaseCallback.countDown()
            releaseServer.countDown()
            runCatching { accepted.get()?.close() }
            server.close()
            if (closer.state == Thread.State.NEW) closer.start()
            closer.join(5000)
            producer.join(5000)
        }
    }
}
