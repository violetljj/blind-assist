package com.linnan.blindassist.teachable

import android.util.AtomicFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.UUID

/** Personal teaching is independent of the ORBIT replay packages and of the S local memory. */
class WclTeachingStore(val root: File) {
    data class ObjectFrames(val directory: File, val name: String, val frames: List<File>)
    data class Library(val names: List<String>, val prototypes: FloatArray, val counts: List<Int>, val whitening: SiblingWhitening? = null)
    internal data class BuildResult(val library: Library, val encodedFrames: Int, val reusedFrames: Int, val buildMs: Double)
    internal data class Warning(val clip: ObjectFrames, val match: WclTeachingWarning.Match)
    internal data class Prepared(val revision: String?, val clips: List<ObjectFrames>, val prior: List<ObjectFrames>,
                                 val modelIdentity: String, val rows: List<List<FloatArray>>, val encodedFrames: Int,
                                 val reusedFrames: Int, val prepareMs: Double, val warnings: List<Warning>)

    private fun completedClips(): List<ObjectFrames> = File(root, "objects").listFiles().orEmpty()
        .filter { it.isDirectory && File(it, "name.txt").isFile && File(it, "complete").isFile }
        .sortedBy { it.name }
        .map { dir -> ObjectFrames(dir, File(dir, "name.txt").readText(), dir.listFiles().orEmpty()
            .filter { it.extension == "jpg" }.sortedBy { it.name }) }
        .filter { it.frames.isNotEmpty() }

    private fun manifest(): JSONObject? {
        val file = File(root, "prototypes.json")
        if (!file.isFile) return null
        return JSONObject(String(AtomicFile(file).readFully(), Charsets.UTF_8))
    }

    private fun clipIds(json: JSONObject?): List<String> {
        if (json == null) return emptyList()
        json.optJSONArray("clips")?.let { a -> return List(a.length()) { a.getString(it) } }
        // The v1 identity already contains the committed directory ids, including when a draft exists.
        return json.getString("source_identity").lineSequence().filter { it.isNotEmpty() }
            .map { it.substringBefore(':') }.toList()
    }

    private fun savedClips(json: JSONObject?, available: List<ObjectFrames>): List<ObjectFrames> {
        val byId = available.associateBy { it.directory.name }
        return clipIds(json).map { id -> checkNotNull(byId[id]) { "已保存的教学图片缺失，请检查教学库" } }
    }

    fun savedClips(): List<ObjectFrames> = synchronized(saveLock) { savedClips(manifest(), completedClips()) }

    fun pendingClips(): List<ObjectFrames> = synchronized(saveLock) {
        val json = manifest()
        val selected = (clipIds(json) + excludedClips(json)).toSet()
        completedClips().filter { it.directory.name !in selected }
    }

    internal fun savedObjects(): List<ObjectFrames> = groupByName(savedClips())

    // The teaching page also exposes completed drafts. Live recognition uses savedObjects().
    fun objects(): List<ObjectFrames> = synchronized(saveLock) { groupByName(savedClips() + pendingClips()) }

    private fun groupByName(clips: List<ObjectFrames>): List<ObjectFrames> =
        clips.groupBy { it.name }.map { (name, sessions) ->
            ObjectFrames(sessions.first().directory, name, sessions.flatMap { it.frames })
        }

    fun newObject(name: String): File {
        require(name.isNotBlank()) { "请先输入物品名字" }
        require(objects().none { it.name == name }) { "这个名字已经使用，请给新物品换一个名字" }
        return newClip(name)
    }

    fun newClip(name: String): File {
        require(name.isNotBlank()) { "请先输入物品名字" }
        return File(root, "objects/${System.currentTimeMillis()}_${UUID.randomUUID()}").apply {
            check(mkdirs()) { "无法保存教学图片" }
            File(this, "name.txt").writeText(name)
        }
    }

    fun finishObject(directory: File): Boolean {
        val count = directory.listFiles().orEmpty().count { it.extension == "jpg" }
        if (count < MIN_FRAMES) return false
        File(directory, "complete").writeText(count.toString())
        return true
    }

    private fun sourceIdentity(clips: List<ObjectFrames>): String = clips.joinToString("\n") { o ->
        o.directory.name + ":" + o.name + ":" + o.frames.joinToString(",") { "${it.name}:${it.length()}:${it.lastModified()}" }
    }

    private fun excludedClips(json: JSONObject?): List<String> = json?.optJSONArray("withdrawn")?.let { a ->
        List(a.length()) { a.getString(it) }
    }.orEmpty()

    /** Only a fully built library replaces the previous one; interrupted builds keep the old bytes. */
    fun save(library: Library, modelIdentity: String) = synchronized(saveLock) {
        save(library, modelIdentity, savedClips() + pendingClips(), excludedClips(manifest()))
    }

    private fun save(library: Library, modelIdentity: String, clips: List<ObjectFrames>, excluded: List<String>) {
        require(library.counts.size == library.names.size)
        require(library.prototypes.size == library.names.size * WclPipeline.D)
        require(library.prototypes.all { it.isFinite() } && library.counts.all { it > 0 })
        root.mkdirs()
        val source = sourceIdentity(clips)
        val json = JSONObject().put("version", 2).put("model_identity", modelIdentity)
            .put("preprocessing", PREPROCESSING).put("source_identity", source)
            .put("clips", JSONArray(clips.map { it.directory.name })).put("withdrawn", JSONArray(excluded))
            .put("names", JSONArray(library.names)).put("counts", JSONArray(library.counts))
            .put("prototypes", JSONArray(library.prototypes.toList())).toString()
        library.whitening?.let { w ->                                    // written first: a stale prototypes.json never pairs with new groups
            val whitening = whiteningIdentity(modelIdentity, source)
            val wf = AtomicFile(File(root, "$whitening-whitening.bin")); val out = wf.startWrite()
            try { w.write(out, whitening); finish(wf, out) } catch (error: Throwable) { wf.failWrite(out); throw error }
        }
        val file = AtomicFile(File(root, "prototypes.json"))
        val output = file.startWrite()
        try { output.write(json.toByteArray(Charsets.UTF_8)); file.finishWrite(output) }
        catch (error: Throwable) { file.failWrite(output); throw error }
        val currentWhitening = "${whiteningIdentity(modelIdentity, source)}-whitening.bin"
        root.listFiles().orEmpty().filter { it.name.endsWith("-whitening.bin") && it.name != currentWhitening }
            .forEach { it.delete() }
    }

    private fun finish(f: AtomicFile, out: java.io.FileOutputStream) { out.flush(); f.finishWrite(out) }

    private fun whiteningIdentity(modelIdentity: String, source: String): String = MessageDigest.getInstance("SHA-256")
        .digest((modelIdentity + "|" + PREPROCESSING + "|" + source).toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }

    /** Sibling-group whitening saved with this library, or null when absent or built for other teaching (then scores stay raw). */
    private fun loadWhitening(modelIdentity: String, json: JSONObject): SiblingWhitening? {
        val identity = whiteningIdentity(modelIdentity, json.getString("source_identity"))
        val f = File(root, if (json.getInt("version") == 1) "whitening.bin" else "$identity-whitening.bin")
        if (!f.isFile) return null
        return AtomicFile(f).openRead().use { SiblingWhitening.read(it, identity) }
    }

    fun load(modelIdentity: String): Library = synchronized(saveLock) {
        val json = checkNotNull(manifest()) { "请先保存教学" }
        check(json.getInt("version") in 1..2 && json.getString("preprocessing") == PREPROCESSING &&
            json.getString("model_identity") == modelIdentity) { "识别模型已改变，请重新保存教学后再识别" }
        check(json.getString("source_identity") == sourceIdentity(savedClips(json, completedClips()))) {
            "已保存的教学内容有变动，请先重新保存教学"
        }
        val names = json.getJSONArray("names"); val counts = json.getJSONArray("counts"); val values = json.getJSONArray("prototypes")
        check(names.length() > 0 && values.length() == names.length() * WclPipeline.D && counts.length() == names.length()) { "教学记录不完整，请重新保存" }
        Library(List(names.length()) { names.getString(it) }, FloatArray(values.length()) { values.getDouble(it).toFloat() },
            List(counts.length()) { counts.getInt(it) }, loadWhitening(modelIdentity, json)).also {
            check(it.prototypes.all { value -> value.isFinite() } && it.counts.all { count -> count > 0 }) { "教学记录无效，请重新保存" }
        }
    }

    /** Same normalised-frame embeddings and mean prototype as WclDeviceTest; no local-memory budget applies. */
    fun build(pipeline: WclPipeline, modelIdentity: String, progress: (String) -> Unit = {}): Library =
        buildWithStats(pipeline, modelIdentity, progress = progress).library

    internal fun buildWithStats(
        pipeline: WclPipeline, modelIdentity: String, reuseEmbeddings: Boolean = true, progress: (String) -> Unit = {},
    ): BuildResult {
        val prepared = prepare(pipeline, modelIdentity, reuseEmbeddings, checkWarnings = false, progress = progress)
        return commit(prepared, progress)
    }

    internal fun prepare(
        pipeline: WclPipeline, modelIdentity: String, reuseEmbeddings: Boolean = true,
        checkWarnings: Boolean = true, removing: ObjectFrames? = null, progress: (String) -> Unit = {},
    ): Prepared {
        val start = System.nanoTime()
        val (json, prior, pending) = synchronized(saveLock) {
            val snapshot = manifest()
            Triple(snapshot, savedClips(snapshot, completedClips()), pendingClips())
        }
        require(removing == null || prior.any { it.directory == removing.directory }) { "该片段已不在教学库中" }
        val taught = if (removing == null) prior + pending else prior.filterNot { it.directory == removing.directory }
        require(removing != null || taught.isNotEmpty()) { "请先录制至少一件物品" }
        val saved = if (checkWarnings && removing == null && pending.any { draft -> prior.any { it.name == draft.name } }) {
            // Rebuild from the committed clips if their model identity changed.
            runCatching { load(modelIdentity) }.getOrNull()
        } else null
        var encodedFrames = 0
        var reusedFrames = 0
        val embeddings = taught.mapIndexed { i, obj ->
            require(obj.frames.size >= MIN_FRAMES) { "${obj.name}只有${obj.frames.size}帧，请重新录制至少2秒" }
            progress("正在保存第 ${i + 1}/${taught.size} 件：${obj.name}")
            val cache = File(obj.directory, WclEmbeddingCache.FILE_NAME)
            val identity = if (reuseEmbeddings) WclEmbeddingCache.identity(modelIdentity, obj.frames) else null
            val cached = identity?.let { WclEmbeddingCache.read(cache, it, obj.frames.size) }
            if (cached != null) {
                reusedFrames += cached.size
                cached
            } else {
                val rows = obj.frames.map { frame ->
                    pipeline.embed(WclPipeline.imageOf(frame)).also { encodedFrames++ }
                }
                if (identity != null) WclEmbeddingCache.write(cache, identity, rows)
                rows
            }
        }
        val pairs = taught.zip(embeddings)
        val reference = if (!checkWarnings || removing != null) null else saved ?: prior.takeIf { it.isNotEmpty() }?.let {
            buildLibrary(pairs.take(it.size), whitening = false)
        }
        val warnings = if (reference == null) emptyList() else pairs.drop(prior.size).mapNotNull { (clip, rows) ->
            WclTeachingWarning.compare(rows, clip.name, reference)
                ?.takeIf { it.gap > WclTeachingWarning.THRESHOLD }?.let { Warning(clip, it) }
        }
        return Prepared(json?.toString(), taught, prior, modelIdentity, embeddings, encodedFrames, reusedFrames,
            (System.nanoTime() - start) / 1e6, warnings)
    }

    private fun buildLibrary(pairs: List<Pair<ObjectFrames, List<FloatArray>>>, whitening: Boolean): Library {
        val objects = pairs.groupBy({ it.first.name }, { it.second }).mapValues { (_, clips) -> clips.flatten() }
        val rowsByObject = objects.values.toList()
        val sums = rowsByObject.map { rows ->
            FloatArray(WclPipeline.D).also { sum -> for (e in rows) for (d in sum.indices) sum[d] += e[d] }
        }
        return Library(objects.keys.toList(), WclPipeline.prototypesOf(sums), rowsByObject.map { it.size },
            if (whitening) SiblingWhitening.build(rowsByObject) else null)
    }

    internal fun commit(prepared: Prepared, progress: (String) -> Unit = {}): BuildResult = synchronized(saveLock) {
        val start = System.nanoTime()
        val json = manifest()
        check(json?.toString() == prepared.revision) { "教学库已更新，请重新检查后保存" }
        val available = completedClips().associateBy { it.directory }
        check(prepared.clips.all { available[it.directory] == it }) { "教学片段已更新，请重新检查后保存" }
        val retained = prepared.clips.mapTo(HashSet()) { it.directory.name }
        val removed = prepared.prior.filter { it.directory.name !in retained }
        val excluded = (excludedClips(json) + removed.map { it.directory.name }).toSet()
        progress("正在比较相似的物品")
        val library = buildLibrary(prepared.clips.zip(prepared.rows), whitening = true)
        save(library, prepared.modelIdentity, prepared.clips, excluded.sorted())
        // Membership changes with prototypes.json. Leftover files after an interruption stay withdrawn.
        removed.forEach { it.directory.deleteRecursively() }
        BuildResult(library, prepared.encodedFrames, prepared.reusedFrames,
            prepared.prepareMs + (System.nanoTime() - start) / 1e6)
    }

    internal fun discardDraft(clip: ObjectFrames) = synchronized(saveLock) {
        check(pendingClips().any { it.directory == clip.directory }) { "该片段已更新" }
        check(File(clip.directory, "complete").delete()) { "无法撤销待保存片段" }
        clip.directory.deleteRecursively()
    }

    companion object {
        private val saveLock = Any()
        const val PERSONAL = "personal"
        const val MIN_FRAMES = 5
        const val PREPROCESSING = "center-square-rotation-bilinear224/rgb-chw-0..255/normalized-mean-v1"

        fun directory(filesDir: File, library: String): File {
            require(library.matches(Regex("[A-Za-z0-9_-]+"))) { "无效的教学库名字" }
            return File(filesDir, "wcl/libraries/$library")
        }

        /** Hash the actual model bytes, rather than accepting prototypes from a different adapter/model. */
        fun modelIdentity(model: File): String {
            check(model.isFile) { "识别模型尚未安装" }
            val digest = MessageDigest.getInstance("SHA-256")
            model.inputStream().buffered(1024 * 1024).use { input ->
                val buffer = ByteArray(1024 * 1024)
                while (true) { val read = input.read(buffer); if (read < 0) break; digest.update(buffer, 0, read) }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
