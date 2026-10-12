package com.linnan.blindassist.capture

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream
import java.net.SocketTimeoutException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import org.junit.Assert.*
import org.junit.Test

class CaptureReadAheadTest {
    @Test(timeout = 5000) fun stalledConsumerKeepsReadReceiptsAndExactBytesWithBoundedBackpressure() {
        val bytes = ByteArray(23) { it.toByte() }
        val threeReads = CountDownLatch(3)
        val fourthRead = CountDownLatch(1)
        val reads = AtomicInteger()
        val clock = AtomicLong(100)
        val input = object : ByteArrayInputStream(bytes) {
            override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
                val count = super.read(buffer, offset, length)
                if (count > 0) {
                    if (reads.incrementAndGet() == 4) fourthRead.countDown()
                    threeReads.countDown()
                }
                return count
            }
        }
        CaptureReadAhead(input, { clock.getAndIncrement() }, "test-read-ahead-bounded", 2, 3).use { pump ->
            assertTrue(threeReads.await(1, TimeUnit.SECONDS))
            // Two queued chunks plus one producer chunk; no fourth read while paused.
            assertFalse(fourthRead.await(100, TimeUnit.MILLISECONDS))
            val output = ByteArrayOutputStream()
            val receipts = mutableListOf<Long>()
            while (true) {
                val chunk = pump.next() ?: break
                output.write(chunk.bytes)
                receipts += chunk.receivedNs
            }
            assertArrayEquals(bytes, output.toByteArray())
            assertEquals((100L..107L).toList(), receipts)
        }
    }

    @Test(timeout = 5000) fun concurrentInputCloseDrainsQueuedBytesAndDecoderTailBeforeEof() {
        val chunks = listOf("boot\n", "first\n", "tail").map { it.toByteArray() }
        val allRead = CountDownLatch(chunks.size)
        val closed = CountDownLatch(1)
        val clock = AtomicLong(10)
        val input = object : InputStream() {
            var next = 0
            override fun read(): Int = error("bulk reads required")
            override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
                if (next == chunks.size) {
                    closed.await()
                    return -1
                }
                val chunk = chunks[next++]
                chunk.copyInto(buffer, offset)
                allRead.countDown()
                return chunk.size
            }
            override fun close() { closed.countDown() }
        }
        CaptureReadAhead(input, { clock.getAndIncrement() }, "test-read-ahead-stop", 2).use { pump ->
            assertTrue(allRead.await(1, TimeUnit.SECONDS))
            val stopper = Thread { input.close() }.apply { start() }
            stopper.join()
            val decoder = CnhCaptureDecoder()
            val records = mutableListOf<CnhCaptureRecord>()
            while (true) {
                val chunk = pump.next() ?: break
                records += decoder.feed(chunk.bytes, chunk.receivedNs)
            }
            decoder.finish()?.let { records += it }
            assertEquals(3, records.size)
            records.zip(chunks).forEach { (record, raw) -> assertArrayEquals(raw, record.rawLine) }
            assertEquals(listOf(10L, 11L, 12L), records.map { it.hostReceivedMonotonicNs })
            assertTrue(records.last().issues.isNotEmpty())
        }
    }

    @Test(timeout = 5000) fun transportFailureFollowsAlreadyReadBytes() {
        val failure = IOException("peer reset")
        val input = object : InputStream() {
            var calls = 0
            override fun read(): Int = error("bulk reads required")
            override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
                if (calls++ > 0) throw failure
                buffer[offset] = 42
                return 1
            }
        }
        CaptureReadAhead(input, { 73 }, "test-read-ahead-error", 1).use { pump ->
            assertArrayEquals(byteArrayOf(42), pump.next()!!.bytes)
            try {
                pump.next()
                fail("transport error must be visible after queued data")
            } catch (error: IOException) { assertSame(failure, error) }
        }
    }

    @Test(timeout = 5000) fun consumerFailureCanReleaseProducerBlockedOnFullQueue() {
        val reads = CountDownLatch(2)
        val input = object : InputStream() {
            override fun read(): Int = error("bulk reads required")
            override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
                buffer[offset] = 42
                reads.countDown()
                return 1
            }
        }
        val pump = CaptureReadAhead(input, { 0 }, "test-read-ahead-abort", 1)
        try {
            assertTrue(reads.await(1, TimeUnit.SECONDS))
        } finally {
            // Must not deadlock waiting for a producer whose queue nobody will drain.
            pump.close()
        }
        assertFalse(Thread.getAllStackTraces().keys.any { it.name == "test-read-ahead-abort" && it.isAlive })
    }

    @Test(timeout = 5000) fun readTimeoutIsNotEofAndDoesNotInventReceipt() {
        val input = object : InputStream() {
            var calls = 0
            override fun read(): Int = error("bulk reads required")
            override fun read(buffer: ByteArray, offset: Int, length: Int): Int = when (calls++) {
                0, 1 -> throw SocketTimeoutException("idle")
                2 -> { buffer[offset] = 7; 1 }
                else -> -1
            }
        }
        val samples = AtomicInteger()
        CaptureReadAhead(input, { samples.incrementAndGet().toLong() }, "test-read-ahead-idle").use { pump ->
            assertEquals(1L, pump.next()!!.receivedNs)
            assertNull(pump.next())
            assertEquals(1, samples.get())
        }
    }
}
