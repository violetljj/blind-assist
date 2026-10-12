package com.linnan.blindassist.capture

import android.util.AtomicFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Bounded app-private diagnostic history; never used to recover stream clocks or readiness. */
internal class CaptureConnectionJournal(directory: File) {
    private val file = AtomicFile(File(directory, "capture-connection-history.json"))
    fun read(): List<CaptureConnectionEvent> = runCatching {
        val array = JSONArray(file.openRead().bufferedReader(Charsets.UTF_8).use { it.readText() })
        (maxOf(0, array.length() - 64) until array.length()).map { index ->
            val obj = array.getJSONObject(index)
            CaptureConnectionEvent(obj.getString("stream"),
                CaptureConnectionReason.valueOf(obj.getString("reason")),
                obj.getLong("observed_elapsed_realtime_ns"), obj.getString("message"), obj.getInt("attempts"),
                if (obj.isNull("recovery_elapsed_ms")) null else obj.getLong("recovery_elapsed_ms"),
                obj.getLong("wall_clock_ms"))
        }
    }.getOrDefault(emptyList())

    fun write(events: List<CaptureConnectionEvent>) {
        val bytes = JSONArray().apply { events.takeLast(64).forEach { put(connectionEventJson(it)) } }
            .toString().toByteArray(Charsets.UTF_8)
        val output = file.startWrite()
        try {
            output.write(bytes)
            file.finishWrite(output)
        } catch (error: Exception) {
            file.failWrite(output)
            throw error
        }
    }
}

internal fun connectionEventJson(event: CaptureConnectionEvent): JSONObject = JSONObject().apply {
    put("stream", event.stream)
    put("reason", event.reason.name)
    put("observed_elapsed_realtime_ns", event.elapsedRealtimeNs)
    put("wall_clock_ms", event.wallClockMs)
    put("message", event.message)
    put("attempts", event.attempts)
    put("recovery_elapsed_ms", event.recoveryElapsedMs ?: JSONObject.NULL)
    put("clock", "android_elapsed_realtime_observation; wall_clock_for_display_only")
}
