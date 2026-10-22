package com.linnan.blindassist.teachable

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.random.Random

/**
 * Times the HTP simmax graphs (filesDir/m0/simmax.dlc = 1024 rows, simmax_<rows>.dlc) on random inputs: engine build,
 * first run and the mean of 5 runs. Writes filesDir/m0/results/simmax_bench.json. Measures only.
 */
@RunWith(AndroidJUnit4::class)
class SimmaxBenchTest {
    @Test
    fun bench() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.filesDir, "m0"); val out = JSONObject()
        val rnd = Random(0)
        // simmax_r<rows>_c<cols>_<mode>.dlc (output mx); simtop6_r<rows>_c<cols>.dlc (output tv,
        // top-6 values, spot-checked against a CPU top-6 on 8 rows); the slow simmax_<rows>.dlc sizes are skipped
        val shape = Regex("sim(max|top6)_r(\\d+)_c(\\d+)(_\\w+)?\\.dlc")
        val files = root.listFiles { x -> shape.matches(x.name) }!!.sortedBy { it.name }   // simmax.dlc is a copy of a named shape
        for (f in files) {
            val m = shape.matchEntire(f.name); val top = m?.groupValues?.get(1) == "top6"
            val rows = m?.groupValues?.get(2)?.toInt() ?: 1024; val cols = m?.groupValues?.get(3)?.toInt() ?: 30000
            val a = FloatArray(rows * 384) { rnd.nextFloat() - .5f }; val b = FloatArray(cols * 384) { rnd.nextFloat() - .5f }
            var t = System.nanoTime()
            SnpeEngine(f.path, true, context.applicationInfo.nativeLibraryDir, listOf(if (top) "tv" else "mx")).use { e ->
                val build = (System.nanoTime() - t) / 1e6
                t = System.nanoTime(); val res = e.run(mapOf("a" to a, "b" to b)).values.single(); val first = (System.nanoTime() - t) / 1e6
                t = System.nanoTime(); repeat(5) { e.run(mapOf("a" to a, "b" to b)) }; val mean = (System.nanoTime() - t) / 1e6 / 5
                val o = JSONObject().put("rows", rows).put("cols", cols).put("build_ms", build).put("first_ms", first)
                    .put("run_ms", mean).put("gmac_per_s", rows.toDouble() * cols * 384 / (mean * 1e6))
                if (top) {
                    var worst = 0.0
                    for (r in 0 until 8) {
                        val sims = DoubleArray(cols) { c -> (0 until 384).sumOf { a[r * 384 + it].toDouble() * b[c * 384 + it] } }
                        val want = sims.sortedDescending().take(6)
                        for (k in 0 until 6) worst = maxOf(worst, kotlin.math.abs(want[k] - res[r * 6 + k]))
                    }
                    o.put("top6_spot_max_abs_diff", worst)
                }
                out.put(f.name, o)
            }
        }
        File(root, "results/simmax_bench.json").writeText(out.toString(1))
    }
}
