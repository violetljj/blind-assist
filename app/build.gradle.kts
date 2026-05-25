import java.util.Properties

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

gradle.taskGraph.whenReady {
    val releasePackageTaskRequested = allTasks.any { task ->
        task.path in setOf(":app:assembleRelease", ":app:bundleRelease") ||
            task.path.contains("packageRelease", ignoreCase = true)
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

    defaultConfig {
        applicationId = "com.linnan.blindassist"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 29
        versionName = "8.1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
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
    }

    compileOptions {
        sourceCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
        targetCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
    }

    kotlinOptions {
        jvmTarget = libs.versions.jvmTarget.get()
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }
}

dependencies {
    implementation(project(":feature:assist"))
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
    kapt(libs.hilt.compiler)

    debugImplementation(libs.androidx.compose.ui.test.manifest)

    testImplementation(libs.junit4)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.rules)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.espresso.core)
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
}
