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
        // simmax.dlc = 1024 x 30000; simmax_r<rows>_c<cols>_<mode>.dlc; the slow simmax_<rows>.dlc sizes are skipped
        val shape = Regex("simmax_r(\\d+)_c(\\d+)_\\w+\\.dlc")
        val files = root.listFiles { x -> x.name == "simmax.dlc" || shape.matches(x.name) }!!.sortedBy { it.name }
        for (f in files) {
            val m = shape.matchEntire(f.name)
            val rows = m?.groupValues?.get(1)?.toInt() ?: 1024; val cols = m?.groupValues?.get(2)?.toInt() ?: 30000
            val a = FloatArray(rows * 384) { rnd.nextFloat() - .5f }; val b = FloatArray(cols * 384) { rnd.nextFloat() - .5f }
            var t = System.nanoTime()
            SnpeEngine(f.path, true, context.applicationInfo.nativeLibraryDir, listOf("mx")).use { e ->
                val build = (System.nanoTime() - t) / 1e6
                t = System.nanoTime(); e.run(mapOf("a" to a, "b" to b)); val first = (System.nanoTime() - t) / 1e6
                t = System.nanoTime(); repeat(5) { e.run(mapOf("a" to a, "b" to b)) }; val mean = (System.nanoTime() - t) / 1e6 / 5
                out.put(f.name, JSONObject().put("rows", rows).put("cols", cols).put("build_ms", build).put("first_ms", first)
                    .put("run_ms", mean).put("gmac_per_s", rows.toDouble() * cols * 384 / (mean * 1e6)))
            }
        }
        File(root, "results/simmax_bench.json").writeText(out.toString(1))
    }
}
