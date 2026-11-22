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
    alias(libs.plugins.android.test)
    alias(libs.plugins.kotlin.android)
}

val prepareEgoClearSnpeJni by tasks.registering(Sync::class) {
    into(layout.buildDirectory.dir("generated/egoclearSnpeJni/arm64-v8a"))
    from("${qairtRoot.get()}/lib/aarch64-android") {
        include("libSNPE.so", "libSnpeHtpPrepare.so", "libSnpeHtpV75Stub.so", "libSnpeHtpV75CalculatorStub.so")
    }
    from("${qairtRoot.get()}/lib/hexagon-v75/unsigned") {
        include("libSnpeHtpV75Skel.so")
    }
    onlyIf { snpeEnabled }
}

android {
    namespace = "com.linnan.blindassist.egoclear.devicecanary"
    compileSdk = libs.versions.compileSdk.get().toInt()
    if (snpeEnabled) ndkVersion = "27.0.12077973"
    targetProjectPath = ":app"

    defaultConfig {
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
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
                path = file("src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
        sourceSets.getByName("main").jniLibs.srcDir(
            layout.buildDirectory.dir("generated/egoclearSnpeJni").get().asFile,
        )
    }

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

if (snpeEnabled) {
    tasks.named("preBuild").configure {
        dependsOn(prepareEgoClearSnpeJni)
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.fromTarget(libs.versions.jvmTarget.get()))
    }
}

dependencies {
    implementation(libs.androidx.lifecycle.common)
    implementation(libs.androidx.test.runner)
    implementation(libs.androidx.test.rules)
    implementation(libs.androidx.test.ext.junit)
    implementation(libs.onnxruntime.android)
}
