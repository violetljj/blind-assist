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
    data class Library(val names: List<String>, val prototypes: FloatArray, val counts: List<Int>)

    fun objects(): List<ObjectFrames> = File(root, "objects").listFiles().orEmpty()
        .filter { it.isDirectory && File(it, "name.txt").isFile && File(it, "complete").isFile }
        .sortedBy { it.name }
        .map { dir -> ObjectFrames(dir, File(dir, "name.txt").readText(), dir.listFiles().orEmpty()
            .filter { it.extension == "jpg" }.sortedBy { it.name }) }
        .filter { it.frames.isNotEmpty() }

    fun newObject(name: String): File {
        require(name.isNotBlank()) { "请先输入物品名字" }
        require(objects().none { it.name == name }) { "这个名字已经使用，请给新物品换一个名字" }
        return File(root, "objects/${System.currentTimeMillis()}_${UUID.randomUUID()}").apply {
            check(mkdirs()) { "无法保存教学图片" }
            File(this, "name.txt").writeText(name)
        }
    }

    fun hasSavedLibrary(): Boolean = File(root, "prototypes.json").isFile

    fun finishObject(directory: File): Boolean {
        val count = directory.listFiles().orEmpty().count { it.extension == "jpg" }
        if (count < MIN_FRAMES) return false
        File(directory, "complete").writeText(count.toString())
        return true
    }

    private fun sourceIdentity(): String = objects().joinToString("\n") { o ->
        o.directory.name + ":" + o.name + ":" + o.frames.joinToString(",") { "${it.name}:${it.length()}:${it.lastModified()}" }
    }

    /** Only a fully built library replaces the previous one; interrupted builds keep the old bytes. */
    fun save(library: Library, modelIdentity: String) {
        require(library.names.isNotEmpty() && library.counts.size == library.names.size)
        require(library.prototypes.size == library.names.size * WclPipeline.D)
        require(library.prototypes.all { it.isFinite() } && library.counts.all { it > 0 })
        root.mkdirs()
        val json = JSONObject().put("version", 1).put("model_identity", modelIdentity)
            .put("preprocessing", PREPROCESSING).put("source_identity", sourceIdentity())
            .put("names", JSONArray(library.names)).put("counts", JSONArray(library.counts))
            .put("prototypes", JSONArray(library.prototypes.toList())).toString()
        val file = AtomicFile(File(root, "prototypes.json"))
        val output = file.startWrite()
        try { output.write(json.toByteArray(Charsets.UTF_8)); file.finishWrite(output) }
        catch (error: Throwable) { file.failWrite(output); throw error }
    }

    fun load(modelIdentity: String): Library {
        val file = AtomicFile(File(root, "prototypes.json"))
        val json = JSONObject(String(file.readFully(), Charsets.UTF_8))
        check(json.getInt("version") == 1 && json.getString("preprocessing") == PREPROCESSING &&
            json.getString("model_identity") == modelIdentity) { "识别模型已改变，请重新保存教学后再识别" }
        check(json.getString("source_identity") == sourceIdentity()) { "教学内容有更新，请先保存教学" }
        val names = json.getJSONArray("names"); val counts = json.getJSONArray("counts"); val values = json.getJSONArray("prototypes")
        check(names.length() > 0 && values.length() == names.length() * WclPipeline.D && counts.length() == names.length()) { "教学记录不完整，请重新保存" }
        return Library(List(names.length()) { names.getString(it) }, FloatArray(values.length()) { values.getDouble(it).toFloat() },
            List(counts.length()) { counts.getInt(it) }).also {
            check(it.prototypes.all { value -> value.isFinite() } && it.counts.all { count -> count > 0 }) { "教学记录无效，请重新保存" }
        }
    }

    /** Same normalised-frame embeddings and mean prototype as WclDeviceTest; no local-memory budget applies. */
    fun build(pipeline: WclPipeline, modelIdentity: String, progress: (String) -> Unit = {}): Library {
        val taught = objects()
        require(taught.isNotEmpty()) { "请先录制至少一件物品" }
        val sums = taught.mapIndexed { i, obj ->
            require(obj.frames.size >= MIN_FRAMES) { "${obj.name}只有${obj.frames.size}帧，请重新录制至少2秒" }
            progress("正在保存第 ${i + 1}/${taught.size} 件：${obj.name}")
            FloatArray(WclPipeline.D).also { sum ->
                for (frame in obj.frames) {
                    val embedding = pipeline.embed(WclPipeline.imageOf(frame))
                    for (d in sum.indices) sum[d] += embedding[d]
                }
            }
        }
        return Library(taught.map { it.name }, WclPipeline.prototypesOf(sums), taught.map { it.frames.size })
            .also { save(it, modelIdentity) }
    }

    companion object {
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
