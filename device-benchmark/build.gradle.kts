import org.gradle.api.tasks.Sync

plugins {
    alias(libs.plugins.android.test)
    alias(libs.plugins.kotlin.android)
}

val detectorBenchmarkAssetsDir = layout.buildDirectory.dir("generated/detectorBenchmarkAssets")
val depthBenchmarkAssetsDir = layout.buildDirectory.dir("generated/depthBenchmarkAssets")
val blindAssistEvalSetDir = providers
    .gradleProperty("blindAssistEvalSetDir")
    .orElse("test-artifacts.local/datasets/blindassist-evalset-20260527-impl")
val depthBenchmarkModelPath = providers
    .gradleProperty("depthBenchmarkModelPath")
    .orElse(".downloads/depth-lab/exports/depth_anything_v2_small_fp32.tflite")
val depthBenchmarkModelAssetName = providers
    .gradleProperty("depthBenchmarkModelAssetName")
    .orElse("depth_anything_v2_small_fp32.tflite")

val prepareDetectorBenchmarkAssets = tasks.register<Sync>("prepareDetectorBenchmarkAssets") {
    from(rootProject.file("app/src/main/assets")) {
        include("coco_labels.txt")
        include("yolo11n_fp16_320.tflite")
    }
    from(rootProject.file(".downloads/detector-lab/exports")) {
        include("yolo26n_fp16_320.tflite")
    }
    from(rootProject.file(".downloads/detector-lab/datasets/coco100")) {
        include("coco100_manifest.json")
        include("coco100_annotations.json")
        include("images/**")
    }
    from(blindAssistEvalSetDir.map { rootProject.file(it) }) {
        include("dataset_spec.json")
        include("manifest.jsonl")
        include("images/test/**")
        include("source_masks/test/**")
        into("blindassist_evalset")
    }
    into(detectorBenchmarkAssetsDir)
}

val prepareDepthBenchmarkAssets = tasks.register<Sync>("prepareDepthBenchmarkAssets") {
    from(depthBenchmarkModelPath.map { rootProject.file(it) }) {
        rename { depthBenchmarkModelAssetName.get() }
        into("depth")
    }
    into(depthBenchmarkAssetsDir)
}

android {
    namespace = "com.linnan.blindassist.benchmark"
    compileSdk = libs.versions.compileSdk.get().toInt()
    targetProjectPath = ":app"
    targetVariant = "debug"

    defaultConfig {
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
        targetCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
    }

    kotlinOptions {
        jvmTarget = libs.versions.jvmTarget.get()
    }

    androidResources {
        noCompress += "tflite"
    }

    sourceSets {
        getByName("main") {
            assets.srcDir(detectorBenchmarkAssetsDir)
            assets.srcDir(depthBenchmarkAssetsDir)
        }
    }
}

tasks.matching {
    it.name in setOf("mergeDebugAssets", "generateDebugLintModel", "generateDebugAndroidTestLintModel")
}.configureEach {
    dependsOn(prepareDetectorBenchmarkAssets)
    dependsOn(prepareDepthBenchmarkAssets)
}

dependencies {
    implementation(project(":core:assist"))
    implementation(project(":core:vision"))
    implementation(libs.androidx.test.runner)
    implementation(libs.androidx.test.rules)
    implementation(libs.androidx.test.ext.junit)
    implementation(libs.androidx.lifecycle.common)
    implementation(libs.tflite)
}
