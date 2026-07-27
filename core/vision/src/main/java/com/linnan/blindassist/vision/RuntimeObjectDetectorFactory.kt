package com.linnan.blindassist.vision

import android.content.Context
import android.content.pm.PackageManager

/**
 * Creates the production CPU detector unless an isolated application explicitly
 * names a provider in its own manifest. Provider lookup never falls back after
 * an opt-in: a broken or unavailable candidate remains visibly not ready.
 */
fun interface ObjectDetectorProvider {
    fun create(context: Context): ObjectDetector
}

object RuntimeObjectDetectorFactory {
    const val PROVIDER_META_DATA =
        "com.linnan.blindassist.vision.OBJECT_DETECTOR_PROVIDER"

    @Suppress("DEPRECATION")
    fun create(context: Context): ObjectDetector {
        val applicationInfo = context.packageManager.getApplicationInfo(
            context.packageName,
            PackageManager.GET_META_DATA
        )
        val providerClassName = applicationInfo.metaData
            ?.getString(PROVIDER_META_DATA)
            ?.trim()
            ?.takeIf { it.isNotEmpty() }
            ?: return TfliteYoloDetector(
                context = context,
                executionBackend = DetectorExecutionBackend.CPU_XNNPACK
            )
        val providerClass = Class.forName(providerClassName, true, context.classLoader)
        val provider = providerClass.getDeclaredConstructor().newInstance()
        require(provider is ObjectDetectorProvider) {
            "$providerClassName must implement ${ObjectDetectorProvider::class.java.name}"
        }
        return provider.create(context)
    }
}
