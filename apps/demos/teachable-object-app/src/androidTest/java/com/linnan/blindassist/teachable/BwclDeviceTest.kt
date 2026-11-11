package com.linnan.blindassist.teachable

import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.SystemClock
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
 * The thesis full system B+WCL on the phone (Graduation bwcl_phone_chain_20261111): the DINOv2-B local-memory branch
 * (B backbone on the HTP, B pooling head + localiser on the CPU, memory built on the phone, gated half-frame relook) and
 * the WCL whole-frame branch, fused per frame as z(B scores) + z(WCL cosine) (thesis eq. 4-6).
 * Inputs: filesDir/m0/<user>/ (M0 packages), filesDir/bwcl/{backbone_b_cached.dlc, bhead_fp32.dlc, <user>_pc_parts_b.raw},
 * filesDir/wcl/wcl_cached.dlc.
 * offlineChain: teaching, both memory arms (built on the phone; built on the PC) and all package queries; writes
 * filesDir/bwcl/results/<user>.json. replay: the full chain at `rate` frames per second (default 3) for `durationSec`
 * (default 600) over one user's queries, phone-built memory; writes filesDir/bwcl/replay.json. Both measure only.
 */
@RunWith(AndroidJUnit4::class)
class BwclDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val m0 = File(context.filesDir, "m0")
    private val root = File(context.filesDir, "bwcl")
    private val results = File(root, "results").apply { mkdirs() }
    private val dsp get() = context.applicationInfo.nativeLibraryDir

    private fun ms(start: Long) = (System.nanoTime() - start) / 1e6
    private fun raw(file: File) = FloatArray((file.length() / 4).toInt()).also {
        ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it)
    }

    private fun bPipeline() = TeachablePipeline(File(root, "backbone_b_cached.dlc"), File(root, "bhead_fp32.dlc"), dsp, dim = B_DIM)
    private fun wPipeline() = WclPipeline(File(context.filesDir, "wcl/wcl_cached.dlc"), dsp)

    /** Teaching for both branches from one decode per frame. */
    private class Taught(val bCls: List<FloatArray>, val bPatches: FloatArray, val counts: IntArray, val wProtos: FloatArray,
                         val frames: Int, val bMs: Double, val wMs: Double, val decodeMs: Double)

    private fun teach(b: TeachablePipeline, w: WclPipeline, dir: File, meta: JSONObject): Taught {
        val objects = meta.getJSONArray("objects").length(); val teach = meta.getJSONArray("teach"); val img = File(dir, "img")
        val counts = IntArray(objects); val cls = List(objects) { FloatArray(B_DIM) }; val sums = List(objects) { FloatArray(WclPipeline.D) }
        val chunks = ArrayList<FloatArray>(); var n = 0; var bMs = 0.0; var wMs = 0.0; var dMs = 0.0
        for (o in 0 until objects) {
            val frames = teach.getJSONArray(o)
            for (i in 0 until frames.length()) {
                var t = System.nanoTime(); val x = b.image(File(img, frames.getString(i))); dMs += ms(t)
                t = System.nanoTime(); val tok = b.tokens(x); bMs += ms(t)
                for (d in 0 until B_DIM) cls[o][d] += tok[d]
                chunks += tok.copyOfRange(B_DIM, tok.size); counts[o] += 256
                t = System.nanoTime(); val e = w.embed(x); wMs += ms(t)
                for (d in 0 until WclPipeline.D) sums[o][d] += e[d]
                n++
            }
        }
        val patches = FloatArray(chunks.sumOf { it.size }).also { a -> var k = 0; for (c in chunks) { c.copyInto(a, k); k += c.size } }
        return Taught(cls, patches, counts, w.prototypes(sums), n, bMs, wMs, dMs)
    }

    /** One frame through both branches; fused = z(B final scores) + z(WCL cosine). */
    private class Frame(val pred: Int, val bPred: Int, val wPred: Int, val looked: Boolean, val bScores: FloatArray, val wScores: FloatArray,
                        val ms: Map<String, Double>)

    private fun frame(b: TeachablePipeline, w: WclPipeline, x: FloatArray, mem: TeachablePipeline.Memory, protos: FloatArray,
                      tok: FloatArray? = null, wcos: FloatArray? = null): Frame {
        val tm = LinkedHashMap<String, Double>()
        var t = System.nanoTime()
        val ws = wcos ?: w.scores(w.embed(x), protos).also { tm["wcl"] = ms(t) }
        t = System.nanoTime()
        val tk = tok ?: b.tokens(x).also { tm["b_backbone"] = ms(t) }
        t = System.nanoTime()
        val r = b.identify(x, tk, mem)
        tm["b_identify"] = ms(t); tm.putAll(r.ms.mapKeys { "b_" + it.key })
        val zb = TeachablePipeline.z(r.scores); val zw = TeachablePipeline.z(ws)
        val f = FloatArray(zb.size) { zb[it] + zw[it] }
        return Frame(f.indices.maxBy { f[it] }, r.pred, ws.indices.maxBy { ws[it] }, r.looked, r.scores, ws, tm)
    }

    @Test
    fun offlineChain() {
        val t0 = System.nanoTime()
        bPipeline().use { b -> wPipeline().use { w ->
            val device = JSONObject().put("init_ms", ms(t0))
            val warm = FloatArray(3 * 224 * 224) { 128f }
            repeat(5) { b.tokens(warm); w.embed(warm) }
            val users = m0.listFiles { f -> File(f, "meta.json").isFile }!!.sortedBy { it.name }
            for (dir in users) {
                val meta = JSONObject(File(dir, "meta.json").readText()); val user = meta.getString("user")
                val objects = meta.getJSONArray("objects").length()
                val taught = teach(b, w, dir, meta)
                var t = System.nanoTime()
                val built = Native.build(taught.bPatches, taught.counts, TeachablePipeline.KEEP, TeachablePipeline.PARTS, dim = B_DIM)
                val buildMs = ms(t)
                val arms = linkedMapOf("phone" to b.memory(objects, built.parts, taught.bCls),
                                       "pc" to b.memory(objects, raw(File(root, "${user}_pc_parts_b.raw")), taught.bCls))
                val out = JSONObject().put("user", user).put("objects", objects).put("teach_frames", taught.frames)
                    .put("teach_b_ms", taught.bMs).put("teach_wcl_ms", taught.wMs).put("teach_decode_ms", taught.decodeMs)
                    .put("memory_build_ms", buildMs).put("memory_build_breakdown_ms", JSONArray(listOf(built.otherMs, built.ownMs, built.kmeansMs)))
                val queries = meta.getJSONArray("queries"); val rows = JSONArray(); val img = File(dir, "img")
                for (i in 0 until queries.length()) {
                    val name = queries.getJSONObject(i).getString("name")
                    t = System.nanoTime(); val x = b.image(File(img, name)); val dec = ms(t)
                    t = System.nanoTime(); val wcos = w.scores(w.embed(x), taught.wProtos); val wMs = ms(t)
                    t = System.nanoTime(); val tok = b.tokens(x); val bbMs = ms(t)
                    val row = JSONObject().put("name", name).put("decode_ms", dec).put("wcl_ms", wMs).put("b_backbone_ms", bbMs)
                        .put("w_scores", JSONArray(wcos.map { it.toDouble() }))
                    for ((arm, mem) in arms) {
                        val f = frame(b, w, x, mem, taught.wProtos, tok, wcos)
                        row.put("${arm}_pred", f.pred).put("${arm}_b_pred", f.bPred).put("${arm}_looked", f.looked)
                            .put("${arm}_b_scores", JSONArray(f.bScores.map { it.toDouble() })).put("${arm}_ms", JSONObject(f.ms as Map<*, *>))
                    }
                    rows.put(row)
                }
                out.put("queries", rows)
                File(results, "$user.json").writeText(out.toString())
                Log.i(TAG, "$user: teach ${taught.frames} frames, memory ${"%.0f".format(buildMs)} ms, ${queries.length()} queries")
            }
            File(results, "device.json").writeText(device.put("total_ms", ms(t0)).toString(1))
        } }
    }

    @Test
    fun replay() {
        val args = InstrumentationRegistry.getArguments()
        val duration = args.getString("durationSec")?.toDouble() ?: 600.0
        val rate = args.getString("rate")?.toDouble() ?: 3.0
        val user = args.getString("user") ?: "P189"
        bPipeline().use { b -> wPipeline().use { w ->
            val dir = File(m0, user); val meta = JSONObject(File(dir, "meta.json").readText())
            val objects = meta.getJSONArray("objects").length()
            val taught = teach(b, w, dir, meta)
            val built = Native.build(taught.bPatches, taught.counts, TeachablePipeline.KEEP, TeachablePipeline.PARTS, dim = B_DIM)
            val mem = b.memory(objects, built.parts, taught.bCls)
            val queries = meta.getJSONArray("queries"); val img = File(dir, "img")
            val frames = JSONArray(); val temps = JSONArray()
            val period = 1000.0 / rate; val start = SystemClock.elapsedRealtime(); var k = 0; var lastTemp = -1e9
            while (SystemClock.elapsedRealtime() - start < duration * 1000) {
                val due = start + k * period
                val wait = (due - SystemClock.elapsedRealtime()).toLong(); if (wait > 0) SystemClock.sleep(wait)
                val tick = System.nanoTime()
                val x = b.image(File(img, queries.getJSONObject(k % queries.length()).getString("name")))
                val decodeMs = ms(tick)
                val f = frame(b, w, x, mem, taught.wProtos)
                frames.put(JSONObject().put("k", k).put("t_ms", SystemClock.elapsedRealtime() - start).put("total_ms", ms(tick))
                    .put("decode_ms", decodeMs).put("looked", f.looked).put("pred", f.pred).put("ms", JSONObject(f.ms as Map<*, *>)))
                val now = SystemClock.elapsedRealtime() - start
                if (now - lastTemp >= 15000) {
                    lastTemp = now.toDouble()
                    val bat = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                    temps.put(JSONObject().put("t_ms", now).put("battery_c", (bat?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1) / 10.0))
                }
                k++
            }
            File(root, "replay.json").writeText(JSONObject().put("user", user).put("rate", rate).put("duration_s", duration)
                .put("frames", frames).put("battery", temps).toString())
            Log.i(TAG, "replay $user: $k frames in ${duration}s")
        } }
    }

    companion object {
        const val B_DIM = 768
        const val TAG = "TeachableBwcl"
    }
}
