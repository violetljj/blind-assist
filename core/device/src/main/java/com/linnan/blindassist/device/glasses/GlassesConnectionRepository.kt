package com.linnan.blindassist.device.glasses

import java.io.BufferedInputStream
import java.net.HttpURLConnection
import java.net.URL
import javax.inject.Inject

data class GlassesDeviceStatus(
    val endpoint: String,
    val firmwareVersion: String,
    val wifiRssiDbm: Int,
    val tofValid: Boolean,
    val tofRangeMm: Int?,
    val streamReachable: Boolean
)

class GlassesConnectionRepository @Inject constructor() {
    fun connect(endpoint: String): Result<GlassesDeviceStatus> = runCatching {
        val base = endpoint.trim().trimEnd('/')
        require(base.startsWith("http://") || base.startsWith("https://")) {
            "Endpoint must use http:// or https://"
        }
        val statusJson = getText("$base/api/status")
        val rangeJson = getText("$base/api/range")
        val firmware = stringField(statusJson, "firmware_version")
            ?: error("Missing firmware_version")
        require(firmware.startsWith("atoms3r_m12_tof4m_")) {
            "Unexpected device firmware: $firmware"
        }
        val streamHost = base.replace(Regex(":\\d+$"), "")
        GlassesDeviceStatus(
            endpoint = base,
            firmwareVersion = firmware,
            wifiRssiDbm = intField(statusJson, "rssi_dbm") ?: 0,
            tofValid = boolField(rangeJson, "valid") ?: false,
            tofRangeMm = intField(rangeJson, "range_mm"),
            streamReachable = probeMjpeg("$streamHost:81/stream")
        )
    }

    private fun getText(url: String): String {
        val connection = open(url)
        return try {
            require(connection.responseCode == HttpURLConnection.HTTP_OK) {
                "HTTP ${connection.responseCode} from $url"
            }
            connection.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
        } finally {
            connection.disconnect()
        }
    }

    private fun probeMjpeg(url: String): Boolean {
        val connection = open(url, readTimeoutMs = 3_000)
        return try {
            if (connection.responseCode != HttpURLConnection.HTTP_OK) return false
            if (!connection.contentType.orEmpty().startsWith("multipart/x-mixed-replace")) return false
            BufferedInputStream(connection.inputStream).use { input ->
                input.read(ByteArray(128)) > 0
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun open(url: String, readTimeoutMs: Int = 5_000): HttpURLConnection {
        return (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 3_000
            readTimeout = readTimeoutMs
            useCaches = false
            setRequestProperty("Connection", "close")
        }
    }

    private fun stringField(json: String, name: String): String? =
        Regex("\\\"${Regex.escape(name)}\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"")
            .find(json)?.groupValues?.get(1)

    private fun intField(json: String, name: String): Int? =
        Regex("\\\"${Regex.escape(name)}\\\"\\s*:\\s*(-?\\d+)")
            .find(json)?.groupValues?.get(1)?.toIntOrNull()

    private fun boolField(json: String, name: String): Boolean? =
        Regex("\\\"${Regex.escape(name)}\\\"\\s*:\\s*(true|false)")
            .find(json)?.groupValues?.get(1)?.toBooleanStrictOrNull()
}
