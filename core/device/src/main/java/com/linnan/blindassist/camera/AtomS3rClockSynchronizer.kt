package com.linnan.blindassist.camera

import android.os.SystemClock
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class AtomS3rClockMapping(
    val deviceMinusAndroidNs: Long,
    val roundTripNs: Long,
    val errorBoundNs: Long
) {
    fun deviceToAndroidNs(deviceNs: Long): Long = deviceNs - deviceMinusAndroidNs
}

class AtomS3rClockSynchronizer(
    endpoint: String,
    private val udpPort: Int = 3333,
    private val attempts: Int = 7
) {
    private val host = requireNotNull(URL(endpoint).host.takeIf(String::isNotBlank))

    fun synchronize(): AtomS3rClockMapping {
        val address = InetAddress.getByName(host)
        DatagramSocket().use { socket ->
            socket.soTimeout = 1_500
            var best: AtomS3rClockMapping? = null
            repeat(attempts) { requestId ->
                val startNs = SystemClock.elapsedRealtimeNanos()
                val request = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
                    .put("BAT0".toByteArray(Charsets.US_ASCII))
                    .putInt(requestId)
                    .putLong(startNs)
                    .array()
                socket.send(DatagramPacket(request, request.size, address, udpPort))
                val response = ByteArray(24)
                val packet = DatagramPacket(response, response.size)
                socket.receive(packet)
                val endNs = SystemClock.elapsedRealtimeNanos()
                require(packet.length == response.size) { "Unexpected AtomS3R timing response length" }
                val decoded = ByteBuffer.wrap(response).order(ByteOrder.LITTLE_ENDIAN)
                val magic = ByteArray(4).also(decoded::get).toString(Charsets.US_ASCII)
                require(magic == "BAT1" && decoded.int == requestId) { "Mismatched AtomS3R timing response" }
                val deviceReceivedNs = decoded.long * NANOS_PER_MICROSECOND
                val deviceSendNs = decoded.long * NANOS_PER_MICROSECOND
                val deviceWorkNs = deviceSendNs - deviceReceivedNs
                val rttNs = (endNs - startNs - deviceWorkNs).coerceAtLeast(0L)
                val mapping = AtomS3rClockMapping(
                    deviceMinusAndroidNs = ((deviceReceivedNs + deviceSendNs) / 2L) -
                        ((startNs + endNs) / 2L),
                    roundTripNs = rttNs,
                    errorBoundNs = rttNs / 2L
                )
                if (best == null || mapping.roundTripNs < best!!.roundTripNs) best = mapping
            }
            return requireNotNull(best)
        }
    }

    private companion object {
        const val NANOS_PER_MICROSECOND = 1_000L
    }
}
