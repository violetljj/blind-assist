import java.util.Properties
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.kapt)
    alias(libs.plugins.hilt.android)
}

val releaseSigningPropertiesFile = rootProject.file("keystore.properties")
val releaseSigningProperties = Properties().apply {
    if (releaseSigningPropertiesFile.exists()) {
        releaseSigningPropertiesFile.inputStream().use(::load)
    }
}
val hasReleaseSigningProperties = listOf("storeFile", "storePassword", "keyAlias", "keyPassword")
    .all { key -> releaseSigningProperties.getProperty(key).isNullOrBlank().not() }
val qnnLegacyPackaging = providers.gradleProperty("qnnLegacyPackaging")
    .map(String::toBoolean)
    .orElse(false)
gradle.taskGraph.whenReady {
    val releasePackageTaskRequested = gradle.startParameter.taskNames.any { requestedTask ->
        requestedTask.substringAfterLast(':') in setOf("assembleRelease", "bundleRelease")
    }
    if (releasePackageTaskRequested && !hasReleaseSigningProperties) {
        throw GradleException(
            "Release signing requires local keystore.properties. " +
                "Copy keystore.properties.example, fill storeFile/storePassword/keyAlias/keyPassword, and keep it untracked."
        )
    }
}

android {
    namespace = "com.linnan.blindassist"
    compileSdk = libs.versions.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "com.linnan.blindassist"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 37
        versionName = "10.9.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("boolean", "USTRF_EXPERIMENT", "false")
        buildConfigField("boolean", "NPU_CANDIDATE", "false")
        buildConfigField("boolean", "DUAL_LOOP_SHADOW", "false")
        buildConfigField("boolean", "DUAL_LOOP_ACTIVE", "false")
        buildConfigField("boolean", "DTR_KNOWN_HEIGHT", "false")
    }

    signingConfigs {
        if (hasReleaseSigningProperties) {
            create("release") {
                storeFile = rootProject.file(releaseSigningProperties.getProperty("storeFile"))
                storePassword = releaseSigningProperties.getProperty("storePassword")
                keyAlias = releaseSigningProperties.getProperty("keyAlias")
                keyPassword = releaseSigningProperties.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            if (hasReleaseSigningProperties) {
                signingConfig = signingConfigs.getByName("release")
            }
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        create("ustrfExperiment") {
            initWith(getByName("debug"))
            applicationIdSuffix = ".ustrf.experimental"
            versionNameSuffix = "-ustrf-exp"
            matchingFallbacks += listOf("debug")
            buildConfigField("boolean", "USTRF_EXPERIMENT", "true")
        }
        create("dualLoopShadow") {
            initWith(getByName("debug"))
            applicationIdSuffix = ".dualloop.shadow"
            versionNameSuffix = "-dual-loop-shadow"
            matchingFallbacks += listOf("debug")
            buildConfigField("boolean", "DUAL_LOOP_SHADOW", "true")
        }
        create("dualLoopActive") {
            initWith(getByName("debug"))
            applicationIdSuffix = ".dualloop.active"
            versionNameSuffix = "-dual-loop-active"
            matchingFallbacks += listOf("debug")
            buildConfigField("boolean", "DUAL_LOOP_ACTIVE", "true")
        }
        create("dtrKnownHeight") {
            initWith(getByName("debug"))
            applicationIdSuffix = ".dtr.knownheight"
            versionNameSuffix = "-dtr-known-height"
            matchingFallbacks += listOf("debug")
            buildConfigField("boolean", "DTR_KNOWN_HEIGHT", "true")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
        targetCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    androidResources {
        noCompress += "tflite"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = qnnLegacyPackaging.get()
        }
    }

}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.fromTarget(libs.versions.jvmTarget.get()))
    }
}

kapt {
    correctErrorTypes = true
}

dependencies {
    implementation(project(":feature:assist"))
    implementation(project(":core:vision"))
    implementation(project(":core:ui"))

    implementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(platform(libs.androidx.compose.bom))

    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.core.splashscreen)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.hilt.android)
    implementation(libs.tflite)
    implementation(libs.qnn.runtime)
    implementation(libs.qnn.litert.delegate)
    kapt(libs.hilt.compiler)

    debugImplementation(libs.androidx.compose.ui.test.manifest)

    testImplementation(libs.junit4)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.rules)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.espresso.core)
    androidTestImplementation(libs.androidx.compose.foundation)
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(project(":core:assist"))
    androidTestImplementation(project(":core:device"))
    androidTestImplementation(project(":core:vision"))
}
