import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

// Teachable personal-object recognition (Graduation thesis line). Standalone demo; the obstacle app is not touched.
// Uses the local QAIRT SDK exactly like :app's optional DAV2 runtime: no SDK or model download.
val qairtRoot = providers.gradleProperty("qairtRoot")
    .orElse(providers.environmentVariable("QAIRT_ROOT"))
    .orElse("E:/codex-tools/qairt/2.47.0.260601")
if (!file("${qairtRoot.get()}/lib/aarch64-android/libSNPE.so").isFile) {
    throw GradleException("teachable-object-app requires a local QAIRT SDK: set -PqairtRoot=<SDK>")
}
val prepareSnpeRuntime by tasks.registering(Sync::class) {
    into(layout.buildDirectory.dir("generated/snpeJni/arm64-v8a"))
    from("${qairtRoot.get()}/lib/aarch64-android") {
        include("libSnpeHtpPrepare.so", "libSnpeHtpV75Stub.so", "libSnpeHtpV75CalculatorStub.so")
    }
    from("${qairtRoot.get()}/lib/hexagon-v75/unsigned") { include("libSnpeHtpV75Skel.so") }
}

android {
    namespace = "com.linnan.blindassist.teachable"
    compileSdk = libs.versions.compileSdk.get().toInt()
    ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "com.linnan.blindassist.teachable"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 1
        versionName = "0.1.0-m0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                abiFilters += "arm64-v8a"
                arguments += "-DQAIRT_ROOT=${qairtRoot.get()}"
                arguments += "-DANDROID_STL=c++_static"
            }
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    sourceSets.getByName("main").jniLibs.srcDir(layout.buildDirectory.dir("generated/snpeJni").get().asFile)
    // Extracted native libs: the HTP skeleton is then found in nativeLibraryDir through ADSP_LIBRARY_PATH.
    packaging { jniLibs { useLegacyPackaging = true } }
    compileOptions {
        sourceCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
        targetCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.fromTarget(libs.versions.jvmTarget.get())) }
}

tasks.named("preBuild").configure { dependsOn(prepareSnpeRuntime) }

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.androidx.test.rules)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.junit4)
}
