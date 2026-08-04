import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.test)
    alias(libs.plugins.kotlin.android)
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
    implementation(libs.qnn.runtime)
    implementation(libs.qnn.litert.delegate)
    implementation("org.opencv:opencv:4.10.0")
}
