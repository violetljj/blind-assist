import org.gradle.api.tasks.Sync
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.test)
    alias(libs.plugins.kotlin.android)
}

val detectorBenchmarkAssetsDir = layout.buildDirectory.dir("generated/detectorBenchmarkAssets")
val depthBenchmarkAssetsDir = layout.buildDirectory.dir("generated/depthBenchmarkAssets")
val segmentationBenchmarkAssetsDir = layout.buildDirectory.dir("generated/segmentationBenchmarkAssets")
val risksegPidnetPreflightAssetsDir = layout.buildDirectory.dir("generated/risksegPidnetPreflightAssets")
val sparseLkBenchmarkAssetsDir = layout.buildDirectory.dir("generated/sparseLkBenchmarkAssets")
val eventHeadBenchmarkAssetsDir = layout.buildDirectory.dir("generated/eventHeadBenchmarkAssets")
val ustrfR12DetectorAssetsDir = layout.buildDirectory.dir("generated/ustrfR12DetectorAssets")
val qnnPreprocessCandidateDir = providers
    .gradleProperty("qnnPreprocessCandidateDir")
    .orElse("artifacts.local/experiments/qnn-preprocess-fusion-v1")
val blindAssistEvalSetDir = providers
    .gradleProperty("blindAssistEvalSetDir")
    .orElse("test-artifacts.local/datasets/blindassist-evalset-20260527-impl")
val eventLifecycleDatasetDir = providers
    .gradleProperty("eventLifecycleDatasetDir")
    .orElse("artifacts.local/evidence/datasets/sanpo-v3-regression-90f")
val publicVideoInferenceDir = providers
    .gradleProperty("publicVideoInferenceDir")
    .orElse("artifacts.local/evidence/public-video-edge-inference/empty")
val depthBenchmarkModelPath = providers
    .gradleProperty("depthBenchmarkModelPath")
    .orElse(".downloads/depth-lab/exports/depth_anything_v2_small_fp32.tflite")
val depthBenchmarkModelAssetName = providers
    .gradleProperty("depthBenchmarkModelAssetName")
    .orElse("depth_anything_v2_small_fp32.tflite")
val segmentationBenchmarkModelPath = providers
    .gradleProperty("segmentationBenchmarkModelPath")
    .orElse(".downloads/traversability-lab/exports/mobilenetv3_lraspp_int8_256.tflite")
val segmentationBenchmarkModelAssetName = providers
    .gradleProperty("segmentationBenchmarkModelAssetName")
    .orElse("mobilenetv3_lraspp_int8_256.tflite")
val risksegPidnetPreflightModelPath = providers
    .gradleProperty("risksegPidnetPreflightModelPath")
    .orElse(
        "artifacts.local/evidence/riskseg-r0/pidnet-preflight-v1/host/" +
            "tflite_export_v3/pidnet_s_512x288_4class_preflight_full_integer_quant.tflite"
    )
val risksegPidnetPreflightCanaryPath = providers
    .gradleProperty("risksegPidnetPreflightCanaryPath")
    .orElse(
        "artifacts.local/evidence/datasets/sanpo-v4-real-canonical-r3-20260713/" +
            "images/blind/sanpo_real_v0_5LlqRK-hWoDLSW5MmoLjKj6uQtZMKjb9_000000.png"
    )
val ustrfR12DetectorModelPath = providers.gradleProperty("ustrfR12DetectorModelPath")
    .orElse("artifacts.local/evidence/ustrf-crosscam-codex/r12-detector-export/yoloe11s_marker_static3_fp16_640.tflite")
val ustrfR12DetectorModelAssetName = providers.gradleProperty("ustrfR12DetectorModelAssetName")
    .orElse("yoloe11s_marker_static3_fp16_640.tflite")
val ustrfR12DetectorLabelsPath = providers.gradleProperty("ustrfR12DetectorLabelsPath")
    .orElse("artifacts.local/evidence/ustrf-crosscam-codex/r12-detector-export/marker_labels.txt")
val ustrfR12DetectorCanaryDir = providers.gradleProperty("ustrfR12DetectorCanaryDir")
    .orElse("artifacts.local/evidence/ustrf-crosscam-codex/r12-detector-export/android-canary")
val benchmarkTargetProject = providers.gradleProperty("benchmarkTargetProject").orElse(":app")

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
    from(eventLifecycleDatasetDir.map { rootProject.file(it) }) {
        include("manifest.jsonl")
        include("images/test/**")
        include("source_masks/test/**")
        into("sanpo_event_lifecycle")
    }
    from(publicVideoInferenceDir.map { rootProject.file(it) }) {
        include("dataset_spec.json")
        include("manifest.jsonl")
        include("images/**")
        into("public_video_inference")
    }
    from(qnnPreprocessCandidateDir.map { rootProject.file(it) }) {
        include("rgba640x480_rot90_letterbox320.tflite")
        include("contract.json")
        into("qnn_preprocess")
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

val prepareSegmentationBenchmarkAssets = tasks.register<Sync>("prepareSegmentationBenchmarkAssets") {
    from(segmentationBenchmarkModelPath.map { rootProject.file(it) }) {
        rename { segmentationBenchmarkModelAssetName.get() }
        into("segmentation")
    }
    into(segmentationBenchmarkAssetsDir)
}

val prepareRisksegPidnetPreflightAssets =
    tasks.register<Sync>("prepareRisksegPidnetPreflightAssets") {
        from(risksegPidnetPreflightModelPath.map { rootProject.file(it) }) {
            rename { "pidnet_s_512x288_4class_preflight_full_integer_quant.tflite" }
            into("riskseg_pidnet")
        }
        from(risksegPidnetPreflightCanaryPath.map { rootProject.file(it) }) {
            rename { "train_rgb_non_eval.png" }
            into("riskseg_pidnet")
        }
        into(risksegPidnetPreflightAssetsDir)
    }

val prepareSparseLkBenchmarkAssets = tasks.register<Sync>("prepareSparseLkBenchmarkAssets") {
    from(rootProject.file("artifacts.local/evidence/datasets/sanpo-boundary-aux-wbp-20260715/whole_object_redacted_rgb")) {
        include("machine_redaction_receipt.json")
        include("images/*.png")
        into("sparse_lk_sanpo")
    }
    into(sparseLkBenchmarkAssetsDir)
}

val prepareEventHeadBenchmarkAssets = tasks.register<Sync>("prepareEventHeadBenchmarkAssets") {
    from(rootProject.file("artifacts.local/experiments/secondary-corridor-causal/event-head-tcn-int8-v0-20260718/android/app/src/main/assets")) {
        include("corridor_causal_tcn_int8_v0.tflite")
        include("corridor_causal_tcn_int8_v0_contract.json")
        include("corridor_causal_tcn_int8_v0_golden.json")
    }
    into(eventHeadBenchmarkAssetsDir)
}

val prepareUstrfR12DetectorAssets = tasks.register<Sync>("prepareUstrfR12DetectorAssets") {
    from(ustrfR12DetectorModelPath.map { rootProject.file(it) }) {
        rename { ustrfR12DetectorModelAssetName.get() }
        into("ustrf_r12_detector")
    }
    from(ustrfR12DetectorLabelsPath.map { rootProject.file(it) }) {
        rename { "marker_labels.txt" }
        into("ustrf_r12_detector")
    }
    from(ustrfR12DetectorCanaryDir.map { rootProject.file(it) }) {
        include("canary_manifest.json")
        include("images/**")
        into("ustrf_r12_detector")
    }
    into(ustrfR12DetectorAssetsDir)
}

android {
    namespace = "com.linnan.blindassist.benchmark"
    compileSdk = libs.versions.compileSdk.get().toInt()
    targetProjectPath = benchmarkTargetProject.get()

    defaultConfig {
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
        targetCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
    }

    androidResources {
        noCompress += "tflite"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    sourceSets {
        getByName("main") {
            assets.srcDir(detectorBenchmarkAssetsDir)
            assets.srcDir(depthBenchmarkAssetsDir)
            assets.srcDir(segmentationBenchmarkAssetsDir)
            assets.srcDir(risksegPidnetPreflightAssetsDir)
            assets.srcDir(sparseLkBenchmarkAssetsDir)
            assets.srcDir(eventHeadBenchmarkAssetsDir)
            assets.srcDir(ustrfR12DetectorAssetsDir)
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.fromTarget(libs.versions.jvmTarget.get()))
    }
}

tasks.matching {
    it.name in setOf("mergeDebugAssets", "generateDebugLintModel", "generateDebugAndroidTestLintModel")
}.configureEach {
    dependsOn(prepareDetectorBenchmarkAssets)
    dependsOn(prepareDepthBenchmarkAssets)
    dependsOn(prepareSegmentationBenchmarkAssets)
    dependsOn(prepareRisksegPidnetPreflightAssets)
    dependsOn(prepareSparseLkBenchmarkAssets)
    dependsOn(prepareEventHeadBenchmarkAssets)
    dependsOn(prepareUstrfR12DetectorAssets)
}

dependencies {
    implementation(project(":core:assist"))
    implementation(project(":core:device"))
    implementation(project(":core:ustrf"))
    implementation(project(":core:vision"))
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.test.runner)
    implementation(libs.androidx.test.rules)
    implementation(libs.androidx.test.ext.junit)
    implementation(libs.androidx.lifecycle.common)
    implementation(libs.tflite)
    implementation(libs.tflite.gpu)
    implementation(libs.tflite.gpu.api)
    implementation(libs.qnn.runtime)
    implementation(libs.qnn.litert.delegate)
    implementation("org.opencv:opencv:4.10.0")
}
