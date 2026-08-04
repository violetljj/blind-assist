import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.gradle.api.tasks.Sync

val qairtRoot = providers.gradleProperty("qairtRoot")
    .orElse(providers.environmentVariable("QAIRT_ROOT"))
    .orElse("E:/codex-tools/qairt/2.47.0.260601")
val localQairtRuntimeAvailable = file(
    "${qairtRoot.get()}/lib/aarch64-android/libQnnHtp.so",
).isFile

plugins {
    alias(libs.plugins.android.test)
    alias(libs.plugins.kotlin.android)
}

val prepareQairtRuntimeJni by tasks.registering(Sync::class) {
    val generatedRoot = layout.buildDirectory.dir("generated/qairtRuntimeJni")
    into(generatedRoot.map { it.dir("arm64-v8a") })
    from(qairtRoot.map { file("$it/lib/aarch64-android") }) {
        include("libQnnHtp.so", "libQnnHtpV75Stub.so", "libQnnSystem.so")
    }
    from(qairtRoot.map { file("$it/lib/hexagon-v75/unsigned") }) {
        include("libQnnHtpV75Skel.so")
    }
    onlyIf { localQairtRuntimeAvailable }
}

android {
    namespace = "com.linnan.blindassist.hftf.devicecanary"
    compileSdk = libs.versions.compileSdk.get().toInt()
    ndkVersion = "27.0.12077973"
    targetProjectPath = ":app"

    defaultConfig {
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-O3", "-ffast-math")
                arguments += "-DANDROID_STL=c++_shared"
                arguments += "-DQAIRT_ROOT=${qairtRoot.get()}"
            }
        }
    }

    buildTypes {
        create("dualLoopShadow") {
            initWith(getByName("debug"))
            matchingFallbacks += listOf("debug")
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    sourceSets.getByName("main").jniLibs.srcDir(
        layout.buildDirectory.dir("generated/qairtRuntimeJni"),
    )

    buildFeatures {
        prefab = true
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
        targetCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
    }
}

tasks.named("preBuild").configure {
    dependsOn(prepareQairtRuntimeJni)
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.fromTarget(libs.versions.jvmTarget.get()))
    }
}

dependencies {
    implementation(project(":core:assist"))
    implementation(project(":hftf-metric-depth-canary-core"))
    // Keep the instrumentation APK ABI-aligned with the target App. Without
    // this explicit constraint, androidx.test contributes lifecycle-common
    // 2.3.1 while the target App uses LifecycleRegistry 2.8.7.
    implementation(libs.androidx.lifecycle.common)
    implementation(libs.androidx.test.runner)
    implementation(libs.androidx.test.rules)
    implementation(libs.androidx.test.ext.junit)
    implementation(libs.onnxruntime.android)
    implementation(libs.tflite)
    if (localQairtRuntimeAvailable) {
        implementation(libs.qnn.litert.delegate) {
            exclude(group = "com.qualcomm.qti", module = "qnn-runtime")
        }
    } else {
        implementation(libs.qnn.runtime)
        implementation(libs.qnn.litert.delegate)
    }
    implementation("org.opencv:opencv:4.10.0")
}
