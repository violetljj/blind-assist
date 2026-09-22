package com.linnan.blindassist.device.glasses

import org.json.JSONObject
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.NetworkInterface
import java.net.SocketTimeoutException
import java.net.URI

/** Numeric private-LAN addresses only. Never resolves arbitrary hostnames or follows redirects. */
object HardwareWifiEndpoint {
    fun normalize(value: String): String {
        val raw = value.trim()
        val uri = URI(if (raw.contains("://")) raw else "http://$raw")
        require(uri.scheme == "http" && uri.userInfo == null && uri.query == null && uri.fragment == null)
        require(uri.port == -1 || uri.port == 80)
        require(uri.path.isNullOrEmpty() || uri.path == "/")
        val host = requireNotNull(uri.host)
        val parts = host.split('.')
        require(parts.size == 4 && parts.all { it.isNotEmpty() && it.all(Char::isDigit) &&
            (it.length == 1 || it[0] != '0') && (it.toIntOrNull() ?: -1) in 0..255 })
        val numbers = parts.map(String::toInt)
        require(numbers[0] == 10 || (numbers[0] == 172 && numbers[1] in 16..31) ||
            (numbers[0] == 192 && numbers[1] == 168)) { "仅支持手机热点或私有局域网 IPv4 地址" }
        require(numbers[3] in 1..254) { "请填写设备地址" }
        return "http://$host"
    }
}

data class HardwareWifiDevice(val role: String, val deviceId: String, val endpoint: String, val firmware: String)

object HardwareWifiDiscovery {
    /** Bounded broadcast discovery; caller runs this off the main thread. No subnet scanning. */
    fun discover(timeoutMs: Int = 3000): List<HardwareWifiDevice> {
        require(timeoutMs in 100..10_000)
        val found = linkedMapOf<String, HardwareWifiDevice>()
        val broadcasts = linkedSetOf("255.255.255.255")
        NetworkInterface.getNetworkInterfaces()?.toList()?.filter { it.isUp && !it.isLoopback }
            ?.flatMap { it.interfaceAddresses }?.mapNotNull { it.broadcast?.hostAddress }
            ?.forEach(broadcasts::add)
        DatagramSocket().use { socket ->
            socket.broadcast = true
            val request = "BADEMO_DISCOVER_V1".toByteArray(Charsets.US_ASCII)
            val deadline = System.nanoTime() + timeoutMs * 1_000_000L
            var nextSend = 0L
            while (System.nanoTime() < deadline && !Thread.currentThread().isInterrupted) {
                val now = System.nanoTime()
                if (now >= nextSend) {
                    broadcasts.forEach { address ->
                        runCatching { socket.send(DatagramPacket(request, request.size, InetAddress.getByName(address), 3334)) }
                    }
                    nextSend = now + 750_000_000L
                }
                socket.soTimeout = ((deadline - now) / 1_000_000L).coerceIn(1L, 200L).toInt()
                val packet = DatagramPacket(ByteArray(2048), 2048)
                try {
                    socket.receive(packet)
                    runCatching {
                        val json = JSONObject(String(packet.data, packet.offset, packet.length, Charsets.UTF_8))
                        val role = json.getString("role")
                        require(role == "camera" || role == "tof")
                        val endpoint = HardwareWifiEndpoint.normalize(json.getString("endpoint"))
                        require(URI(endpoint).host == packet.address.hostAddress)
                        val id = json.getString("device_id").also { require(it.isNotBlank() && it.length <= 128) }
                        val firmware = json.getString("firmware").take(128)
                        found["$role:$id"] = HardwareWifiDevice(role, id, endpoint, firmware)
                    }
                } catch (_: SocketTimeoutException) { /* Continue until the single deadline. */ }
            }
        }
        return found.values.toList()
    }
}

/** Reject repeated samples even if a sender changes only its response/send timestamp. */
internal class HardwareWifiProgress {
    private var boot: String? = null
    private var sequence = -1L
    private var sampledUs = -1L
    fun accept(nextBoot: String, nextSequence: Long, nextSampledUs: Long): Boolean {
        if (nextBoot.isBlank() || nextSequence < 0L || nextSampledUs < 0L) return false
        if (boot != nextBoot) {
            boot = nextBoot
            sequence = -1L
            sampledUs = -1L
        }
        if (nextSequence <= sequence || nextSampledUs <= sampledUs) return false
        sequence = nextSequence
        sampledUs = nextSampledUs
        return true
    }
}

internal object HardwareWifiFreshness {
    const val MAX_AGE_MS = 750L
    fun requestAgeUpperMs(sampledUs: Long, sendUs: Long, requestMs: Long, receiveMs: Long): Long? {
        if (sampledUs < 0 || sendUs < sampledUs || requestMs < 0 || receiveMs < requestMs) return null
        val deviceAge = (sendUs - sampledUs) / 1000L
        val elapsed = receiveMs - requestMs
        if (deviceAge > MAX_AGE_MS || elapsed > MAX_AGE_MS) return null
        return (deviceAge + elapsed + 1L).takeIf { it <= MAX_AGE_MS }
    }
    fun fresh(receivedMs: Long, ageAtReceiveMs: Long, nowMs: Long): Boolean =
        receivedMs >= 0L && nowMs >= receivedMs && ageAtReceiveMs >= 0L &&
            ageAtReceiveMs <= MAX_AGE_MS && nowMs - receivedMs <= MAX_AGE_MS - ageAtReceiveMs
}
