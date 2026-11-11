package com.linnan.blindassist.teachable

import android.Manifest
import android.content.Intent
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * WCL on the phone (Graduation orbit_wcl_phone_20261102). offlineChain: teach and identify on the M0 validation-user
 * packages with filesDir/wcl/wcl_cached.dlc, writes filesDir/wcl/results/<user>.json. liveRun: WclLiveActivity for
 * durationSec (default 600) at rate frames per second (default 3), -e spatial true for the spatial readout, -e confPower 0 -e halfLife 1 for the old plain EMA. offlineSpatial:
 * the same packages with PfPipeline (Graduation orbit_wcl_protofull_phone_20261104), writes filesDir/wcl/results_pf/<user>.json.
 * smootherReplay: ConfidenceEma over the stored per-frame scores of filesDir/wcl/results, writes filesDir/wcl/smoother_replay.json to compare with the offline code.
 * All measure; they assert only that the chain ran.
 */
@RunWith(AndroidJUnit4::class)
class WclDeviceTest {
    @get:Rule val camera: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun ms(start: Long) = (System.nanoTime() - start) / 1e6

    @Test
    fun offlineChain() {
        val root = File(context.filesDir, "m0")
        val results = File(context.filesDir, "wcl/results").apply { mkdirs() }
        val t0 = System.nanoTime()
        WclPipeline(File(context.filesDir, "wcl/wcl_cached.dlc"), context.applicationInfo.nativeLibraryDir).use { p ->
            val loadMs = ms(t0)
            for (dir in root.listFiles { f -> File(f, "meta.json").isFile }!!.sortedBy { it.name }) {
                val meta = JSONObject(File(dir, "meta.json").readText())
                val objects = meta.getJSONArray("objects").length()
                val teach = meta.getJSONArray("teach")
                val sums = List(objects) { FloatArray(WclPipeline.D) }
                val teachStart = System.nanoTime(); var teachFrames = 0
                for (o in 0 until objects) {
                    val frames = teach.getJSONArray(o)
                    for (i in 0 until frames.length()) {
                        val e = p.embed(p.image(File(dir, "img/" + frames.getString(i))))
                        for (d in 0 until WclPipeline.D) sums[o][d] += e[d]
                        teachFrames++
                    }
                }
                val protos = p.prototypes(sums)
                val teachMs = ms(teachStart)
                val queries = meta.getJSONArray("queries"); val rows = JSONArray()
                for (i in 0 until queries.length()) {
                    val q = queries.getJSONObject(i)
                    var t = System.nanoTime(); val x = p.image(File(dir, "img/" + q.getString("name"))); val dec = ms(t)
                    t = System.nanoTime(); val e = p.embed(x); val em = ms(t)
                    val s = p.scores(e, protos)
                    rows.put(JSONObject().put("name", q.getString("name")).put("decode_ms", dec).put("embed_ms", em)
                        .put("pred", s.indices.maxBy { s[it] }).put("scores", JSONArray(s.map { it.toDouble() })))
                }
                File(results, "${dir.name}.json").writeText(JSONObject().put("user", dir.name).put("load_ms", loadMs).put("teach_frames", teachFrames)
                    .put("teach_ms", teachMs).put("queries", rows).toString())
                Log.i(WclLiveActivity.TAG, "${dir.name}: teach $teachFrames frames ${"%.0f".format(teachMs)} ms, ${queries.length()} queries")
            }
        }
    }

    /** Sibling-group whitening on the M0 packages (Graduation orbit_wcl_whiten_20261108): per user, teach/query embeddings as float32
     *  files and raw plus whitened scores at tau 0.70, 0.5 and the full panel, for an exact comparison with group_white.py. */
    @Test
    fun offlineWhiten() {
        val root = File(context.filesDir, "m0")
        val results = File(context.filesDir, "wcl/results_whiten").apply { mkdirs() }
        WclPipeline(File(context.filesDir, "wcl/wcl_cached.dlc"), context.applicationInfo.nativeLibraryDir).use { p ->
            for (dir in root.listFiles { f -> File(f, "meta.json").isFile }!!.sortedBy { it.name }) {
                val meta = JSONObject(File(dir, "meta.json").readText())
                val teach = meta.getJSONArray("teach"); val objects = teach.length()
                val rows = List(objects) { o -> List(teach.getJSONArray(o).length()) { i -> p.embed(p.image(File(dir, "img/" + teach.getJSONArray(o).getString(i)))) } }
                val sums = rows.map { r -> FloatArray(WclPipeline.D).also { s -> for (e in r) for (d in s.indices) s[d] += e[d] } }
                val protos = p.prototypes(sums)
                val taus = listOf(0.70, 0.5, -1.0); val buildMs = ArrayList<Double>()
                val whitenings = taus.map { t -> val t0 = System.nanoTime(); SiblingWhitening.build(rows, t).also { buildMs += ms(t0) } }
                val queries = meta.getJSONArray("queries"); val out = JSONArray(); val qEmb = ArrayList<FloatArray>(); val applyMs = ArrayList<Double>()
                for (i in 0 until queries.length()) {
                    val e = p.embed(p.image(File(dir, "img/" + queries.getJSONObject(i).getString("name")))); qEmb += e
                    val raw = p.scores(e, protos); val row = JSONObject().put("raw", JSONArray(raw.map { it.toDouble() }))
                    for ((k, w) in whitenings.withIndex()) { val t0 = System.nanoTime(); val s = w.apply(e, raw); if (k == 2) applyMs += ms(t0); row.put("tau_${taus[k]}", JSONArray(s.map { it.toDouble() })) }
                    out.put(row)
                }
                fun dump(name: String, v: List<FloatArray>) = java.io.DataOutputStream(File(results, name).outputStream().buffered()).use { o -> v.forEach { e -> e.forEach(o::writeFloat) } }
                dump("${dir.name}_teach.f32", rows.flatten()); dump("${dir.name}_query.f32", qEmb)
                File(results, "${dir.name}.json").writeText(JSONObject().put("user", dir.name).put("teach_counts", JSONArray(rows.map { it.size }))
                    .put("groups", JSONArray(whitenings.map { w -> JSONArray(w.groups.map { g -> JSONArray(g.members.toList()) }) }))
                    .put("build_ms", JSONArray(buildMs)).put("apply_full_panel_ms_median", applyMs.sorted()[applyMs.size / 2]).put("queries", out).toString())
                Log.i(WclLiveActivity.TAG, "${dir.name} whitening: groups ${whitenings.map { it.groups.size }}, build ms $buildMs")
            }
        }
    }

    @Test
    fun offlineSpatial() {
        val root = File(context.filesDir, "m0")
        val results = File(context.filesDir, "wcl/results_pf").apply { mkdirs() }
        val t0 = System.nanoTime()
        PfPipeline(File(context.filesDir, "wcl/pf_body_cached.dlc"), File(context.filesDir, "wcl/pf_head_m20_cached.dlc"), context.applicationInfo.nativeLibraryDir).use { p ->
            val loadMs = ms(t0)
            for (dir in root.listFiles { f -> File(f, "meta.json").isFile }!!.sortedBy { it.name }) {
                val meta = JSONObject(File(dir, "meta.json").readText())
                val objects = meta.getJSONArray("objects").length()
                val teach = meta.getJSONArray("teach")
                val sums = List(objects) { FloatArray(WclPipeline.D) }
                val teachStart = System.nanoTime(); var teachFrames = 0
                for (o in 0 until objects) {
                    val frames = teach.getJSONArray(o)
                    for (i in 0 until frames.length()) {
                        val e = p.embed(WclPipeline.imageOf(File(dir, "img/" + frames.getString(i))))
                        for (d in 0 until WclPipeline.D) sums[o][d] += e[d]
                        teachFrames++
                    }
                }
                val protos = WclPipeline.prototypesOf(sums); p.setPrototypes(protos)
                val teachMs = ms(teachStart)
                val queries = meta.getJSONArray("queries"); val rows = JSONArray()
                for (i in 0 until queries.length()) {
                    val q = queries.getJSONObject(i)
                    var t = System.nanoTime(); val x = WclPipeline.imageOf(File(dir, "img/" + q.getString("name"))); val dec = ms(t)
                    t = System.nanoTime(); val tok = p.tokens(x); val body = ms(t)
                    t = System.nanoTime(); val r = p.read(tok); val read = ms(t)
                    val plain = WclPipeline.cosine(r.embedding, protos)
                    rows.put(JSONObject().put("name", q.getString("name")).put("decode_ms", dec).put("body_ms", body).put("read_ms", read)
                        .put("pred", r.scores.indices.maxBy { r.scores[it] }).put("wcl_pred", plain.indices.maxBy { plain[it] })
                        .put("scores", JSONArray(r.scores.map { it.toDouble() })).put("wcl_scores", JSONArray(plain.map { it.toDouble() })))
                }
                File(results, "${dir.name}.json").writeText(JSONObject().put("user", dir.name).put("load_ms", loadMs).put("teach_frames", teachFrames)
                    .put("teach_ms", teachMs).put("queries", rows).toString())
                Log.i(WclLiveActivity.TAG, "${dir.name} spatial: teach $teachFrames frames ${"%.0f".format(teachMs)} ms, ${queries.length()} queries")
            }
        }
    }

    @Test
    fun smootherReplay() {
        val out = JSONObject()
        for (f in File(context.filesDir, "wcl/results").listFiles { x -> x.name.endsWith(".json") }!!.sortedBy { it.name }) {
            val q = JSONObject(f.readText()).getJSONArray("queries")
            val weighted = ConfidenceEma(ConfidenceEma.alphaOf(3.0)); val plain = ConfidenceEma(ConfidenceEma.alphaOf(3.0, 1.0), 0); val wt = JSONArray(); val pt = JSONArray()
            for (i in 0 until q.length()) {
                val a = q.getJSONObject(i).getJSONArray("scores"); val s = FloatArray(a.length()) { a.getDouble(it).toFloat() }
                wt.put(weighted.update(s)); pt.put(plain.update(s))
            }
            out.put(f.name, JSONObject().put("weighted", wt).put("plain", pt))
        }
        File(context.filesDir, "wcl/smoother_replay.json").writeText(out.toString())
    }

    @Test
    fun liveRun() {
        val args = InstrumentationRegistry.getArguments()
        val seconds = args.getString("durationSec")?.toLong() ?: 600L
        val intent = Intent(context, WclLiveActivity::class.java).putExtra(WclLiveActivity.EXTRA_USER, args.getString("user") ?: "P189")
            .putExtra(WclLiveActivity.EXTRA_RATE, args.getString("rate")?.toDouble() ?: 3.0)
            .putExtra(WclLiveActivity.EXTRA_SPATIAL, args.getString("spatial") == "true")
            .putExtra(WclLiveActivity.EXTRA_BWCL, args.getString("bwcl") == "true")
            .putExtra(WclLiveActivity.EXTRA_CONF_POWER, args.getString("confPower")?.toInt() ?: ConfidenceEma.DEFAULT_POWER)
            .putExtra(WclLiveActivity.EXTRA_HALF_LIFE, args.getString("halfLife")?.toDouble() ?: ConfidenceEma.DEFAULT_HALF_LIFE_S)
        ActivityScenario.launch<WclLiveActivity>(intent).use { scenario ->
            Thread.sleep(seconds * 1000)
            var out: File? = null
            scenario.onActivity { out = it.stopAndDump() }
            check(out!!.length() > 1000) { "no frames analysed" }
        }
    }
}
