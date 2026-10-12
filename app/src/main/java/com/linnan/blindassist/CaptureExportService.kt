package com.linnan.blindassist

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.provider.DocumentsContract
import com.linnan.blindassist.capture.PhoneCaptureStore
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** A bounded streaming ZIP export. No recordings are deleted, even after cancellation. */
class CaptureExportService : Service() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var tasks: CaptureExportTasks
    private var activeId: String? = null
    private var cancellation: AtomicBoolean? = null
    private val destroyed = AtomicBoolean(false)
    private val workerFinished = AtomicBoolean(true)

    override fun onCreate() {
        super.onCreate()
        tasks = CaptureExportTasks.get(this)
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL, "采集导出", NotificationManager.IMPORTANCE_LOW))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            if (tasks.state.value?.id == intent.getStringExtra(EXTRA_TASK_ID)) tasks.cancel()
            if (activeId == null) stopSelf(startId)
            return START_NOT_STICKY
        }
        val task = tasks.state.value?.takeIf {
            it.id == intent?.getStringExtra(EXTRA_TASK_ID) && it.isActive
        }
        if (task == null) {
            if (activeId == null) stopSelf(startId)
            return START_NOT_STICKY
        }
        if (activeId != null) return START_NOT_STICKY
        activeId = task.id
        cancellation = tasks.token(task.id) ?: run { stopSelf(); return START_NOT_STICKY }
        if (Build.VERSION.SDK_INT >= 29) startForeground(NOTIFICATION_ID, notification(task),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        else startForeground(NOTIFICATION_ID, notification(task))
        workerFinished.set(false)
        worker.execute { export(task, requireNotNull(cancellation)) }
        return START_NOT_STICKY
    }

    private fun export(task: CaptureExportTask, cancellation: AtomicBoolean) {
        val uri = Uri.parse(task.destination)
        var destinationOpened = false
        try {
            if (cancellation.get()) throw CancellationException()
            tasks.update(task.id, { it.copy(status = CaptureExportStatus.RUNNING, message = "正在导出") }, true)
            val session = PhoneCaptureStore.listSessions(this).singleOrNull { it.id == task.sessionId }
                ?: error("找不到原始记录")
            // `wt` requests truncation: never append a retry to a previous ZIP.
            val output = contentResolver.openOutputStream(uri, "wt") ?: error("无法打开保存位置")
            destinationOpened = true
            output.use {
                PhoneCaptureStore.exportZip(session, it, isCancelled = { cancellation.get() }) { written, total ->
                    tasks.update(task.id, { current -> current.copy(bytesWritten = written, totalBytes = total) })
                    tasks.state.value?.takeIf { !cancellation.get() }?.let { current ->
                        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(current))
                    }
                }
            }
            if (cancellation.get()) throw CancellationException()
            val seconds = (System.currentTimeMillis() - task.startedAtMs) / 1000.0
            tasks.update(task.id, {
                // Same manager lock as cancel(): cancellation cannot be overwritten by completion.
                if (cancellation.get()) throw CancellationException()
                it.copy(status = CaptureExportStatus.COMPLETED,
                finishedAtMs = System.currentTimeMillis(),
                message = "导出完成：${task.sessionId}.zip（${"%.1f".format(seconds)} 秒）") }, true)
        } catch (error: Exception) {
            val cancelled = cancellation.get() || error is CancellationException
            // This URI belongs to ACTION_CREATE_DOCUMENT for this task. Some providers
            // cannot delete it; report the remaining partial file instead of hiding it.
            val removed = runCatching { DocumentsContract.deleteDocument(contentResolver, uri) }.getOrDefault(false)
            val partial = if (removed) "未完成的目标文件已移除。" else if (destinationOpened)
                "目标文件可能不完整，请手动删除。" else "目标可能留有空文件，请检查保存位置。"
            tasks.update(task.id, { it.copy(status = if (cancelled) CaptureExportStatus.CANCELLED else CaptureExportStatus.FAILED,
                finishedAtMs = System.currentTimeMillis(),
                message = (if (cancelled) "导出已取消。" else "导出失败：${error.message}。") + partial + "原始记录仍在，可重新导出。") }, true)
        } finally {
            tasks.releaseGrant(uri)
            stopForeground(STOP_FOREGROUND_REMOVE)
            // A notification cancel creates a newer startId; stopSelf(oldId) would
            // leave the finished service alive and permanently block the next export.
            stopSelf()
            workerFinished.set(true)
            releaseWhenFinished()
        }
    }

    private fun notification(task: CaptureExportTask): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, PhoneCaptureActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val cancel = PendingIntent.getService(this, 1, Intent(this, CaptureExportService::class.java)
            .setAction(ACTION_CANCEL).putExtra(EXTRA_TASK_ID, task.id),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val progress = if (task.totalBytes > 0) (task.bytesWritten * 100 / task.totalBytes).toInt().coerceIn(0, 100) else 0
        return Notification.Builder(this, CHANNEL).setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle("正在导出采集记录")
            .setContentText(if (task.status == CaptureExportStatus.CANCELLING) task.message else
                "$progress% · ${task.bytesWritten / 1024 / 1024} / ${task.totalBytes / 1024 / 1024} MB")
            .setContentIntent(open).setOnlyAlertOnce(true).setOngoing(true)
            .setProgress(100, progress, task.totalBytes == 0L)
            .addAction(Notification.Action.Builder(null, "取消", cancel).build()).build()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        tasks.cancel()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        cancellation?.set(true)
        destroyed.set(true)
        worker.shutdown() // Cooperatively stop between bounded writes; never block the main thread.
        releaseWhenFinished()
        super.onDestroy()
    }

    private fun releaseWhenFinished() {
        // A stopped service can still have a provider write blocked on its worker.
        // Retain ownership until both have ended; another task cannot reset its token.
        if (destroyed.get() && workerFinished.get()) activeId?.let(tasks::releaseExecution)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        const val EXTRA_TASK_ID = "capture_export_task_id"
        private const val ACTION_CANCEL = "com.linnan.blindassist.CANCEL_CAPTURE_EXPORT"
        private const val CHANNEL = "capture_export"
        private const val NOTIFICATION_ID = 2103
    }
}
