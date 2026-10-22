package com.linnan.blindassist.teachable

import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * M0 offline chain on pushed ORBIT validation-user packages (Graduation repo phone_m0_20261022/export_m0.py):
 * filesDir/m0/ holds backbone_fp32.dlc, head_dyn.dlc, head_ref/ (PC head reference inputs) and one directory per user
 * with meta.json, pc_parts.raw and img/.
 * Writes filesDir/m0/results/<user>.json and filesDir/m0/results/device.json. Measures; asserts only that the chain runs.
 */
@RunWith(AndroidJUnit4::class)
class M0DeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    // App-internal storage (filled with `run-as ... cp` from /data/local/tmp): native code cannot open adb-pushed files
    // under Android/data on this device.
    private val root = File(context.filesDir, "m0")
    private val results = File(root, "results").apply { mkdirs() }
    private var peakNative = 0L
    private var simmax: SnpeEngine? = null
    private var simtop: SnpeEngine? = null

    private fun ms(start: Long) = (System.nanoTime() - start) / 1e6
    private fun raw(file: File) = FloatArray((file.length() / 4).toInt()).also {
        ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it)
    }
    private fun sample() { peakNative = maxOf(peakNative, Debug.getNativeHeapAllocatedSize()) }

    @Test
    fun offlineChain() {
        val t0 = System.nanoTime()
        TeachablePipeline(File(root, "backbone_fp32.dlc"), File(root, "head_dyn.dlc"), context.applicationInfo.nativeLibraryDir).use { p ->
            val device = JSONObject().put("init_ms", ms(t0)).put("head_outputs", JSONArray(p.headOutputs))
            // head sanity on the PC reference inputs
            val ref = File(root, "head_ref")
            if (ref.isDirectory) {
                val m = TeachablePipeline.Memory(7, raw(File(ref, "ref_parts.raw")), raw(File(ref, "ref_protos.raw")), raw(File(ref, "ref_valid.raw")))
                device.put("head_ref_first_logits", JSONArray(p.identify(FloatArray(3 * 224 * 224), raw(File(ref, "ref_tokens.raw")), m).firstLogits.toList()))
            }
            // HTP simmax engine for the memory build's other-object max (optional) and its sanity check
            val simmaxFile = File(root, "simmax.dlc")
            simmax = if (simmaxFile.isFile) SnpeEngine(simmaxFile.path, true, context.applicationInfo.nativeLibraryDir, listOf("mx")) else null
            val simtopFile = File(root, "simtop.dlc")
            simtop = if (simtopFile.isFile) SnpeEngine(simtopFile.path, true, context.applicationInfo.nativeLibraryDir, listOf("tv")) else null
            simmax?.let { e ->
                val a = raw(File(ref, "simmax_a.raw")); val b = raw(File(ref, "simmax_b.raw")); val want = raw(File(ref, "simmax_ref.raw"))
                val t = System.nanoTime(); val got = e.run(mapOf("a" to a, "b" to b)).values.single()
                device.put("simmax_first_run_ms", ms(t)).put("simmax_ref_max_abs_diff", got.indices.maxOf { kotlin.math.abs(got[it] - want[it]) })
                val t2 = System.nanoTime(); repeat(5) { e.run(mapOf("a" to a, "b" to b)) }; device.put("simmax_run_ms_mean", ms(t2) / 5)
            }
            // warm-up of both graphs
            val warm = FloatArray(3 * 224 * 224) { 128f }
            repeat(5) { p.tokens(warm) }
            val users = root.listFiles { f -> File(f, "meta.json").isFile }!!.sortedBy { it.name }
            for (u in users) runUser(p, u)
            simmax?.close(); simtop?.close()
            sample()
            device.put("users", JSONArray(users.map { it.name })).put("peak_native_heap_bytes", peakNative)
                .put("total_ms", ms(t0))
            File(results, "device.json").writeText(device.toString(1))
        }
    }

    private fun runUser(p: TeachablePipeline, dir: File) {
        val meta = JSONObject(File(dir, "meta.json").readText())
        val img = File(dir, "img")
        val objects = meta.getJSONArray("objects").length()
        val teach = meta.getJSONArray("teach")
        val out = JSONObject().put("user", meta.getString("user")).put("objects", objects).put("K", meta.getInt("K"))
        // teaching: one backbone pass per teaching frame; patches grouped by object, CLS summed per object
        val counts = IntArray(objects); val cls = List(objects) { FloatArray(TeachablePipeline.D) }
        val patchChunks = ArrayList<FloatArray>(); val decodeMs = ArrayList<Double>(); val bbMs = ArrayList<Double>()
        val tTeach = System.nanoTime()
        for (o in 0 until objects) {
            val frames = teach.getJSONArray(o)
            for (i in 0 until frames.length()) {
                var t = System.nanoTime(); val x = p.image(File(img, frames.getString(i))); decodeMs += ms(t)
                t = System.nanoTime(); val tok = p.tokens(x); bbMs += ms(t)
                for (d in 0 until TeachablePipeline.D) cls[o][d] += tok[d]
                patchChunks += tok.copyOfRange(TeachablePipeline.D, tok.size); counts[o] += 256
            }
        }
        val teachMs = ms(tTeach)
        val patches = FloatArray(patchChunks.sumOf { it.size }).also { a -> var k = 0; for (c in patchChunks) { c.copyInto(a, k); k += c.size } }
        patchChunks.clear(); sample()
        var t = System.nanoTime()
        val cpu = Native.build(patches, counts, TeachablePipeline.KEEP, TeachablePipeline.PARTS)
        val buildMs = ms(t); sample()
        val phoneParts = cpu.parts
        out.put("teach_frames", decodeMs.size).put("teach_total_ms", teachMs).put("teach_decode_ms_mean", decodeMs.average())
            .put("teach_backbone_ms_mean", bbMs.average()).put("memory_build_ms", buildMs)
            .put("memory_build_breakdown_ms", JSONArray(listOf(cpu.otherMs, cpu.ownMs, cpu.kmeansMs)))
            .put("memory_bytes", objects * TeachablePipeline.PARTS * TeachablePipeline.D * 4)
        val arms = linkedMapOf("phone" to p.memory(objects, phoneParts, cls), "pc" to p.memory(objects, raw(File(dir, "pc_parts.raw")), cls))
        simmax?.let { e ->                                               // other-object max on the HTP, own top-6 on the CPU
            t = System.nanoTime()
            val htp = Native.build(patches, counts, TeachablePipeline.KEEP, TeachablePipeline.PARTS, e)
            out.put("memory_build_htpmax_ms", ms(t)).put("memory_build_htpmax_breakdown_ms", JSONArray(listOf(htp.otherMs, htp.ownMs, htp.kmeansMs)))
            arms["phone_htpmax"] = p.memory(objects, htp.parts, cls)
        }
        if (simmax != null && simtop != null) {                          // both similarity terms on the HTP
            t = System.nanoTime()
            val htp = Native.build(patches, counts, TeachablePipeline.KEEP, TeachablePipeline.PARTS, simmax, simtop)
            out.put("memory_build_htp_ms", ms(t)).put("memory_build_htp_breakdown_ms", JSONArray(listOf(htp.otherMs, htp.ownMs, htp.kmeansMs)))
            arms["phone_htp"] = p.memory(objects, htp.parts, cls)
        }
        File(results, "${meta.getString("user")}_phone_parts.raw").writeBytes(
            ByteBuffer.allocate(phoneParts.size * 4).order(ByteOrder.LITTLE_ENDIAN).also { it.asFloatBuffer().put(phoneParts) }.array())
        // identification
        val queries = meta.getJSONArray("queries"); val rows = JSONArray()
        for (i in 0 until queries.length()) {
            val q = queries.getJSONObject(i)
            t = System.nanoTime(); val x = p.image(File(img, q.getString("name"))); val dec = ms(t)
            t = System.nanoTime(); val tok = p.tokens(x); val bb = ms(t)
            val row = JSONObject().put("name", q.getString("name")).put("decode_ms", dec).put("backbone_ms", bb)
            for ((arm, m) in arms) {
                t = System.nanoTime(); val r = p.identify(x, tok, m); val total = ms(t)
                row.put("${arm}_pred", r.pred).put("${arm}_looked", r.looked).put("${arm}_ms", JSONObject(r.ms as Map<*, *>))
                    .put("${arm}_identify_ms", total).put("${arm}_first_logits", JSONArray(r.firstLogits.toList()))
            }
            rows.put(row)
        }
        sample()
        out.put("queries", rows)
        File(results, "${meta.getString("user")}.json").writeText(out.toString())
        Log.i("TeachableM0", "${meta.getString("user")}: $objects objects, build ${"%.0f".format(buildMs)} ms, ${queries.length()} queries")
    }
}
