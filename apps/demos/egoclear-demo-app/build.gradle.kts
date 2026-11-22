import org.gradle.api.tasks.Sync
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

val qairtRoot = providers.gradleProperty("qairtRoot")
    .orElse(providers.environmentVariable("QAIRT_ROOT"))
    .orElse("E:/codex-tools/qairt/2.47.0.260601")
val snpeLibFile = file("${qairtRoot.get()}/lib/aarch64-android/libSNPE.so")
val snpeEnabled = providers.gradleProperty("enableEgoClearSnpe")
    .map(String::toBoolean)
    .orElse(snpeLibFile.isFile)
    .get()

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

val prepareSharedEgoClearSources = tasks.register<Sync>("prepareSharedEgoClearSources") {
    into(layout.buildDirectory.dir("generated/egoclearSharedSrc"))
    from(rootProject.file("apps/canaries/egoclear-device-canary/src/main/java")) {
        include(
            "com/linnan/blindassist/egoclear/EgoClearSnpeEncoder.kt",
            "com/linnan/blindassist/egoclear/EgoClearStreamingEngine.kt",
        )
    }
}

val prepareEgoClearDemoSnpeJni = tasks.register<Sync>("prepareEgoClearDemoSnpeJni") {
    into(layout.buildDirectory.dir("generated/egoclearDemoSnpeJni/arm64-v8a"))
    from("${qairtRoot.get()}/lib/aarch64-android") {
        include("libSNPE.so", "libSnpeHtpPrepare.so", "libSnpeHtpV75Stub.so", "libSnpeHtpV75CalculatorStub.so")
    }
    from("${qairtRoot.get()}/lib/hexagon-v75/unsigned") {
        include("libSnpeHtpV75Skel.so")
    }
    onlyIf { snpeEnabled }
}

android {
    namespace = "com.linnan.blindassist.egoclear.demo"
    compileSdk = libs.versions.compileSdk.get().toInt()
    if (snpeEnabled) ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "com.linnan.blindassist.egoclear.demo"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 1
        versionName = "0.66-npu-demo"
        if (snpeEnabled) {
            ndk {
                abiFilters += "arm64-v8a"
            }
            externalNativeBuild {
                cmake {
                    abiFilters += "arm64-v8a"
                    arguments += "-DQAIRT_ROOT=${qairtRoot.get()}"
                    arguments += "-DANDROID_STL=c++_static"
                }
            }
        }
    }

    if (snpeEnabled) {
        externalNativeBuild {
            cmake {
                path = rootProject.file("apps/canaries/egoclear-device-canary/src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
        sourceSets.getByName("main").jniLibs.srcDir(
            layout.buildDirectory.dir("generated/egoclearDemoSnpeJni").get().asFile,
        )
    }

    sourceSets.getByName("main").java.srcDir(
        layout.buildDirectory.dir("generated/egoclearSharedSrc").get().asFile,
    )

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
        targetCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
    }
}

tasks.named("preBuild").configure {
    dependsOn(prepareSharedEgoClearSources)
    if (snpeEnabled) {
        dependsOn(prepareEgoClearDemoSnpeJni)
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.fromTarget(libs.versions.jvmTarget.get()))
    }
}

dependencies {
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.lifecycle.common)
    implementation(libs.onnxruntime.android)
}
