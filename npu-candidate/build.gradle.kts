import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.kapt)
    alias(libs.plugins.hilt.android)
}

android {
    namespace = "com.linnan.blindassist"
    compileSdk = libs.versions.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "com.linnan.blindassist.npu.candidate"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 37
        versionName = "10.9.0-npu-candidate"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("boolean", "USTRF_EXPERIMENT", "false")
        buildConfigField("boolean", "NPU_CANDIDATE", "true")
        buildConfigField("boolean", "DUAL_LOOP_SHADOW", "false")
        buildConfigField("boolean", "DUAL_LOOP_ACTIVE", "false")
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    sourceSets {
        getByName("main") {
            java.srcDir(rootProject.file("app/src/main/java"))
            res.srcDir(rootProject.file("app/src/main/res"))
            assets.srcDir(rootProject.file("app/src/main/assets"))
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
            useLegacyPackaging = true
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
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.core.splashscreen)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.hilt.android)
    kapt(libs.hilt.compiler)

    implementation(libs.tflite)
    implementation(libs.qnn.runtime)
    implementation(libs.qnn.litert.delegate)
}
