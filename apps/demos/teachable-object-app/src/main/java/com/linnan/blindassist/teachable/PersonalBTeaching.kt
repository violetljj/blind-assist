package com.linnan.blindassist.teachable

import android.util.Log
import java.io.File
import java.io.IOException
import java.security.MessageDigest

/** B memory for the personal library, with a snapshot for the next single-object append. */
internal object PersonalBTeaching {
    class Result(
        val taught: TeachablePipeline.TeachingMemory,
        val mode: String,
        val restoreMs: Double = 0.0,
        val saveMs: Double = 0.0,
        val cacheBytes: Long = 0L,
    )

    fun build(
        store: WclTeachingStore, pipeline: TeachablePipeline, backbone: File, head: File,
        teachFiles: List<List<File>>, forceFull: Boolean = false, captureEvidence: Boolean = false,
    ): Result {
        fun full(cacheBytes: Long = 0L) = Result(
            pipeline.buildTeachingMemory(teachFiles, captureEvidence = captureEvidence, captureFeatureHash = false),
            "full_cutoff", cacheBytes = cacheBytes,
        )
        if (forceFull) return full()

        val (currentIdentity, previousIdentity) = identities(backbone, head, teachFiles)
        val directory = File(store.root, "b-teaching-cache")
        val current = File(directory, "$currentIdentity.bin")
        if (current.isFile) return full(current.length())

        val previous = File(directory, "$previousIdentity.bin")
        val restoreStart = System.nanoTime()
        val prior = if (teachFiles.size > 2 && previous.isFile) {
            try {
                pipeline.restoreTeachingAppend(previous, previousIdentity)
            } catch (error: IOException) {
                Log.w(TAG, "Could not read the saved B teaching cache", error)
                null
            } catch (error: IllegalArgumentException) {
                Log.w(TAG, "Saved B teaching cache failed validation", error)
                null
            }
        } else null
        val restoreMs = if (prior == null) 0.0 else (System.nanoTime() - restoreStart) / 1e6
        val next = if (prior != null) {
            pipeline.appendTeachingAndCache(prior, teachFiles.last(), captureEvidence, captureFeatureHash = false)
        } else {
            if (teachFiles.size == TeachablePipeline.OMAX) return full()
            pipeline.prepareTeachingAppend(teachFiles, captureEvidence, captureFeatureHash = false)
        }

        val saveStart = System.nanoTime()
        val savedBytes = try {
            directory.mkdirs()
            pipeline.saveTeachingAppend(next, current, currentIdentity)
            // The old snapshot remains available until the replacement has been committed.
            directory.listFiles().orEmpty().filter { it != current && CACHE_FILE.matches(it.name) }.forEach {
                if (!it.delete()) Log.w(TAG, "Could not remove a superseded B teaching cache")
            }
            current.length()
        } catch (error: IOException) {
            // Recognition can use the new memory even if disk space or storage prevents saving its cache.
            Log.w(TAG, "Could not save the B teaching cache for the next append", error)
            0L
        }
        val saveMs = (System.nanoTime() - saveStart) / 1e6
        val mode = if (prior == null) "full_seed" else "restored_append"
        return Result(next.initial, mode, restoreMs, saveMs, savedBytes)
    }

    private fun identities(backbone: File, head: File, teachFiles: List<List<File>>): Pair<String, String> {
        val model = "personal-b-teaching-v1|${WclTeachingStore.PREPROCESSING}|${WclTeachingStore.modelIdentity(backbone)}|" +
            WclTeachingStore.modelIdentity(head)
        val fileDigest = MessageDigest.getInstance("SHA-256")
        val objectDigest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(64 * 1024)
        val objects = teachFiles.map { frames ->
            for (frame in frames) {
                frame.inputStream().use { stream ->
                    while (true) {
                        val count = stream.read(buffer)
                        if (count < 0) break
                        fileDigest.update(buffer, 0, count)
                    }
                }
                objectDigest.update(fileDigest.digest())
            }
            objectDigest.digest()
        }
        fun key(count: Int): String {
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update(model.toByteArray(Charsets.UTF_8))
            for (index in 0 until count) digest.update(objects[index])
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
        return key(objects.size) to key(objects.size - 1)
    }

    private const val TAG = "PersonalBTeaching"
    private val CACHE_FILE = Regex("[0-9a-f]{64}\\.bin")
}
