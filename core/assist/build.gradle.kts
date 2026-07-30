import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
}

java {
    sourceCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
    targetCompatibility = JavaVersion.toVersion(libs.versions.jvmTarget.get())
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.fromTarget(libs.versions.jvmTarget.get()))
    }
}

dependencies {
    testImplementation(libs.junit4)
}

tasks.register<JavaExec>("runDualLoopJrdbShadowReplay") {
    group = "verification"
    description = "Runs the explicit diagnostic-only JRDB dual-loop shadow replay."
    dependsOn(tasks.named("testClasses"))
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.linnan.blindassist.session.DualLoopJrdbShadowReplayMain")
}
