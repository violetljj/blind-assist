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
// Optional local research runtime. No SDK/model download and no default assist routing.
val dav2QairtRoot = providers.gradleProperty("qairtRoot")
    .orElse(providers.environmentVariable("QAIRT_ROOT"))
    .orElse("E:/codex-tools/qairt/2.47.0.260601")
val dav2SnpeLibrary = file("${dav2QairtRoot.get()}/lib/aarch64-android/libSNPE.so")
val dav2SnpeEnabled = providers.gradleProperty("enableDav2Snpe")
    .map(String::toBoolean).orElse(dav2SnpeLibrary.isFile).get()
if (dav2SnpeEnabled && !dav2SnpeLibrary.isFile) {
    throw GradleException("DAV2 SNPE requires a local QAIRT SDK: set -PqairtRoot=<SDK> or -PenableDav2Snpe=false")
}
val prepareDav2SnpeRuntime by tasks.registering(Sync::class) {
    into(layout.buildDirectory.dir("generated/dav2SnpeJni/arm64-v8a"))
    from("${dav2QairtRoot.get()}/lib/aarch64-android") {
        include("libSnpeHtpPrepare.so", "libSnpeHtpV75Stub.so", "libSnpeHtpV75CalculatorStub.so")
    }
    from("${dav2QairtRoot.get()}/lib/hexagon-v75/unsigned") { include("libSnpeHtpV75Skel.so") }
    onlyIf { dav2SnpeEnabled }
}
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
    if (dav2SnpeEnabled) ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "com.linnan.blindassist"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 64
        versionName = "10.20.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("boolean", "USTRF_EXPERIMENT", "false")
        buildConfigField("boolean", "NPU_CANDIDATE", "false")
        buildConfigField("boolean", "DUAL_LOOP_SHADOW", "false")
        buildConfigField("boolean", "DUAL_LOOP_ACTIVE", "false")
        buildConfigField("boolean", "DTR_KNOWN_HEIGHT", "false")
        buildConfigField("boolean", "DAV2_SNPE_AVAILABLE", dav2SnpeEnabled.toString())
        if (dav2SnpeEnabled) externalNativeBuild {
            cmake {
                abiFilters += "arm64-v8a"
                arguments += "-DQAIRT_ROOT=${dav2QairtRoot.get()}"
                arguments += "-DANDROID_STL=c++_static"
            }
        }
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
        // Demo install: same id and debug key as `debug` (updates in place, keeps data) but not
        // debuggable, so ART AOT-compiles the app instead of running hot Kotlin loops under JIT.
        create("demo") {
            initWith(getByName("debug"))
            isDebuggable = false
            isJniDebuggable = false
            signingConfig = signingConfigs.getByName("debug")
            matchingFallbacks += listOf("debug")
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
    if (dav2SnpeEnabled) {
        sourceSets.getByName("main").java.srcDir("src/dav2Snpe/java")
        sourceSets.getByName("androidTest").java.srcDir("src/dav2SnpeAndroidTest/java")
        sourceSets.getByName("main").jniLibs.srcDir(layout.buildDirectory.dir("generated/dav2SnpeJni").get().asFile)
        externalNativeBuild {
            cmake {
                path = file("src/dav2Snpe/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
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

tasks.named("preBuild").configure { dependsOn(prepareDav2SnpeRuntime) }

dependencies {
    implementation(project(":core:assist"))
    implementation(project(":core:device"))
    implementation(libs.androidx.compose.foundation)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.extended)
    implementation(libs.androidx.compose.ui)
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
