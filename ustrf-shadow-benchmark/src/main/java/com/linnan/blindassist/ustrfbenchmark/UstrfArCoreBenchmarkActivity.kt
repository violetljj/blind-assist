package com.linnan.blindassist.ustrfbenchmark

import android.app.Activity
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.graphics.Color
import android.view.Gravity
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Deliberately isolated OpenGL host for an ARCore instrumentation audit.
 * It is not reachable from the BlindAssist application or any launcher intent.
 */
class UstrfArCoreBenchmarkActivity : Activity() {
    private val glReady = CountDownLatch(1)
    private val cameraTexture = AtomicReference(0)
    private lateinit var surface: GLSurfaceView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        surface = GLSurfaceView(this).apply {
            setEGLContextClientVersion(2)
            setRenderer(object : GLSurfaceView.Renderer {
                override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
                    val textures = IntArray(1)
                    GLES20.glGenTextures(1, textures, 0)
                    GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0])
                    GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
                    GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
                    GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
                    GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
                    cameraTexture.set(textures[0])
                    glReady.countDown()
                }

                override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) = Unit

                override fun onDrawFrame(gl: GL10?) {
                    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
                }
            })
            // The instrumentation test submits Session.update and Session.close as queued GL
            // actions. Keep this isolated host alive between observations so cleanup cannot be
            // stranded behind an idle RENDERMODE_WHEN_DIRTY renderer.
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        setContentView(FrameLayout(this).apply {
            addView(surface)
            addView(TextView(context).apply {
                setTextColor(Color.WHITE)
                setShadowLayer(4f, 1f, 1f, Color.BLACK)
                textSize = 16f
                text = "USTRF ARCore audit active\\nMove the device slowly in a safe, open area.\\nNo navigation command will be issued."
                gravity = Gravity.CENTER
            }, FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            ))
        })
    }

    override fun onResume() {
        super.onResume()
        surface.onResume()
    }

    override fun onPause() {
        surface.onPause()
        super.onPause()
    }

    fun awaitGlReady(timeoutSeconds: Long = 10): Boolean = glReady.await(timeoutSeconds, TimeUnit.SECONDS)

    fun cameraTextureName(): Int = cameraTexture.get()

    fun runOnGlThreadAndWait(timeoutSeconds: Long = 10L, action: () -> Unit) {
        require(timeoutSeconds > 0L)
        val done = CountDownLatch(1)
        val failure = AtomicReference<Throwable?>()
        surface.queueEvent {
            try {
                action()
            } catch (throwable: Throwable) {
                failure.set(throwable)
            } finally {
                done.countDown()
            }
        }
        check(done.await(timeoutSeconds, TimeUnit.SECONDS)) { "GL action timed out" }
        failure.get()?.let { throw AssertionError("GL action failed", it) }
    }
}
