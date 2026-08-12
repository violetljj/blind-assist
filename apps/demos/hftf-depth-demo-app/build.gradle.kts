import java.nio.file.Path
import org.gradle.api.tasks.Sync
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

val qairtRoot = providers.gradleProperty("qairtRoot")
    .orElse(providers.environmentVariable("QAIRT_ROOT"))
    .orElse("E:/codex-tools/qairt/2.47.0.260601")
val localQairtRuntimeAvailable = Path.of(qairtRoot.get())
    .resolve("lib/aarch64-android/libQnnHtp.so")
    .toFile()
    .isFile
val dav2CachedDlc = providers.gradleProperty("dav2CachedDlcPath")
    .orElse(
        rootProject.file(
            "artifacts.local/work/dav2-qnn-native-cached-context-r0/model-sm8650-cached.dlc",
        ).absolutePath,
    )

plugins {
    alias(libs.plugins.android.application)
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

val prepareDav2DemoModelAsset by tasks.registering(Sync::class) {
    val generatedRoot = layout.buildDirectory.dir("generated/dav2DemoAssets")
    into(generatedRoot)
    from(dav2CachedDlc.map(::file)) {
        rename { "model-sm8650-cached.dlc" }
    }
    doFirst {
        check(file(dav2CachedDlc.get()).isFile) {
            "Missing local cached DLC: ${dav2CachedDlc.get()}"
        }
    }
}

android {
    namespace = "com.linnan.blindassist.depthdemo"
    compileSdk = libs.versions.compileSdk.get().toInt()
    ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "com.linnan.blindassist.depthdemo"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 2
        versionName = "0.2-r0"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-O3", "-fno-fast-math", "-ffp-contract=off")
                arguments += "-DANDROID_STL=c++_shared"
                arguments += "-DQAIRT_ROOT=${qairtRoot.get()}"
            }
        }
    }

    packaging { jniLibs { useLegacyPackaging = true } }
    sourceSets.getByName("main").jniLibs.srcDir(
        layout.buildDirectory.dir("generated/qairtRuntimeJni"),
    )
    sourceSets.getByName("main").assets.srcDir(
        layout.buildDirectory.dir("generated/dav2DemoAssets"),
    )
    androidResources { noCompress += "dlc" }
    buildFeatures { prefab = true }
    externalNativeBuild {
        cmake {
            path = rootProject.file("apps/canaries/hftf-device-canary/src/main/cpp/CMakeLists.txt")
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
    dependsOn(prepareDav2DemoModelAsset)
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.fromTarget(libs.versions.jvmTarget.get())) }
    sourceSets.getByName("main").kotlin.srcDir(
        rootProject.file("apps/canaries/hftf-device-canary/src/main/java"),
    )
    sourceSets.getByName("main").kotlin.include(
        "com/linnan/blindassist/hftf/Dav2Preprocessors.kt",
        "com/linnan/blindassist/hftf/Dav2QnnCachedContext.kt",
        "com/linnan/blindassist/hftf/Dav2Yuv420RgbConverter.kt",
        "com/linnan/blindassist/hftf/DepthExperienceActivity.kt",
    )
}

dependencies {
    implementation(project(":core:assist"))
    implementation(project(":core:device"))
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.lifecycle.common)
    implementation("org.opencv:opencv:4.10.0")
}
