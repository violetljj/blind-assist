package com.linnan.blindassist.teachable

import android.Manifest
import android.content.Intent
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * M1 live run: LiveActivity on the back camera for durationSec (instrumentation argument, default 600 s) with the PC
 * memory of `user` (default P189); `-e r10 true` uses the r10 relook on the HTP. Writes filesDir/m1/live_<user>[_r10].json. Measures; asserts only that frames were analysed.
 */
@RunWith(AndroidJUnit4::class)
class M1LiveTest {
    @get:Rule val camera: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun liveRun() {
        val args = InstrumentationRegistry.getArguments()
        val seconds = args.getString("durationSec")?.toLong() ?: 600L
        val user = args.getString("user") ?: "P189"
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, LiveActivity::class.java).putExtra(LiveActivity.EXTRA_USER, user)
            .putExtra(LiveActivity.EXTRA_R10, args.getString("r10") == "true")
        ActivityScenario.launch<LiveActivity>(intent).use { scenario ->
            Thread.sleep(seconds * 1000)
            var out: File? = null
            scenario.onActivity { out = it.stopAndDump() }
            Log.i(LiveActivity.TAG, "wrote $out (${out?.length()} bytes)")
            check(out!!.length() > 1000) { "no frames analysed" }
        }
    }
}
