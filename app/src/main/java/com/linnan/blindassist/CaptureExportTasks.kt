package com.linnan.blindassist

import android.content.Context
import android.content.ClipData
import android.content.Intent
import android.net.Uri
import android.util.AtomicFile
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONObject
import java.io.File
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

enum class CaptureExportStatus { QUEUED, RUNNING, CANCELLING, COMPLETED, FAILED, CANCELLED, INTERRUPTED }

data class CaptureExportTask(
    val id: String,
    val sessionId: String,
    val destination: String,
    val status: CaptureExportStatus,
    val bytesWritten: Long = 0,
    val totalBytes: Long = 0,
    val startedAtMs: Long,
    val finishedAtMs: Long? = null,
    val message: String = "准备导出",
) {
    val isActive: Boolean get() = status in setOf(CaptureExportStatus.QUEUED,
        CaptureExportStatus.RUNNING, CaptureExportStatus.CANCELLING)

    internal fun json(): JSONObject = JSONObject().apply {
        put("id", id); put("session", sessionId); put("destination", destination)
        put("status", status.name); put("written", bytesWritten); put("total", totalBytes)
        put("started", startedAtMs); put("finished", finishedAtMs ?: JSONObject.NULL)
        put("message", message)
    }

    companion object {
        internal fun restore(json: JSONObject): CaptureExportTask {
            val saved = CaptureExportTask(json.getString("id"), json.getString("session"),
                json.getString("destination"), CaptureExportStatus.valueOf(json.getString("status")),
                json.optLong("written"), json.optLong("total"), json.getLong("started"),
                if (json.isNull("finished")) null else json.getLong("finished"), json.optString("message"))
            return if (saved.isActive) saved.copy(status = CaptureExportStatus.INTERRUPTED,
                finishedAtMs = System.currentTimeMillis(),
                message = "导出被系统中断，目标文件可能不完整；请重新选择位置导出。原始记录仍在。") else saved
        }
    }
}

/** One user-started export, owned by the process/service instead of an Activity. */
class CaptureExportTasks private constructor(private val context: Context) {
    private val record = AtomicFile(File(context.noBackupFilesDir, "capture-export-task.json"))
    private val mutableState = MutableStateFlow(runCatching {
        record.openRead().bufferedReader().use { CaptureExportTask.restore(JSONObject(it.readText())) }
    }.getOrNull())
    val state: StateFlow<CaptureExportTask?> = mutableState.asStateFlow()
    private var cancellation = AtomicBoolean(false)
    private var executionOwner: String? = null

    @Synchronized
    fun start(sessionId: String, destination: Uri): Boolean {
        if (mutableState.value?.isActive == true || executionOwner != null) return false
        require(destination.scheme == "content") { "请选择文档保存位置" }
        mutableState.value?.let { releaseGrant(Uri.parse(it.destination)) }
        // Some providers grant only transient access. That is sufficient for this run;
        // every explicit retry asks for a new document rather than assuming a grant.
        runCatching { context.contentResolver.takePersistableUriPermission(destination,
            Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION) }
        cancellation = AtomicBoolean(false)
        val task = CaptureExportTask(UUID.randomUUID().toString(), sessionId, destination.toString(),
            CaptureExportStatus.QUEUED, startedAtMs = System.currentTimeMillis())
        executionOwner = task.id
        publish(task, durable = true)
        try {
            context.startForegroundService(Intent(context, CaptureExportService::class.java)
                .setData(destination)
                .apply { clipData = ClipData.newRawUri("采集导出", destination) }
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
                .putExtra(CaptureExportService.EXTRA_TASK_ID, task.id))
        } catch (error: Exception) {
            publish(task.copy(status = CaptureExportStatus.FAILED, finishedAtMs = System.currentTimeMillis(),
                message = "无法启动导出：${error.message}。请重新选择位置；目标可能为空。"), durable = true)
            releaseGrant(destination)
            executionOwner = null
            throw error
        }
        return true
    }

    @Synchronized
    fun cancel() {
        val task = mutableState.value?.takeIf { it.isActive } ?: return
        cancellation.set(true)
        publish(task.copy(status = CaptureExportStatus.CANCELLING,
            message = "正在取消，等待当前文件写入结束…"), durable = true)
    }

    @Synchronized
    internal fun update(id: String, transform: (CaptureExportTask) -> CaptureExportTask, durable: Boolean = false) {
        val old = mutableState.value?.takeIf { it.id == id } ?: return
        publish(transform(old), durable)
    }

    @Synchronized
    internal fun token(id: String): AtomicBoolean? = cancellation.takeIf { executionOwner == id }

    @Synchronized
    internal fun releaseExecution(id: String) {
        if (executionOwner == id) executionOwner = null
    }

    @Synchronized
    internal fun hasExecution(): Boolean = executionOwner != null

    private fun publish(task: CaptureExportTask, durable: Boolean) {
        var persistenceFailure = false
        if (durable) {
            persistenceFailure = runCatching {
                val output = record.startWrite()
                try {
                    output.write(task.json().toString().toByteArray(Charsets.UTF_8))
                    record.finishWrite(output)
                } catch (error: Exception) {
                    record.failWrite(output)
                    throw error
                }
            }.isFailure
        }
        mutableState.value = if (persistenceFailure) task.copy(message = task.message +
            "（任务状态保存失败，重启后请检查导出文件。）") else task
    }

    internal fun releaseGrant(uri: Uri) {
        runCatching { context.contentResolver.releasePersistableUriPermission(uri,
            Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION) }
    }

    companion object {
        @Volatile private var instance: CaptureExportTasks? = null
        fun get(context: Context): CaptureExportTasks = instance ?: synchronized(this) {
            instance ?: CaptureExportTasks(context.applicationContext).also { instance = it }
        }
    }
}
