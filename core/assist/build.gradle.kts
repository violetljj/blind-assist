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

tasks.register<JavaExec>("runHftfD34DetectorTrackParity") {
    group = "verification"
    description = "Runs source-only HFTF D34 Kotlin/Python track-state parity."
    dependsOn(tasks.named("testClasses"))
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.linnan.blindassist.session.HftfD34DetectorTrackParityMain")
    args(
        providers.gradleProperty("d34Input").orElse(
            rootProject.layout.projectDirectory.file(
                "artifacts.local/evidence/hftf/" +
                    "stage-c-d34-kotlin-shadow-state-parity-v0/parity_input.tsv"
            ).asFile.absolutePath
        ).get(),
        providers.gradleProperty("d34Output").orElse(
            rootProject.layout.projectDirectory.file(
                "artifacts.local/evidence/hftf/" +
                    "stage-c-d34-kotlin-shadow-state-parity-v0/report.json"
            ).asFile.absolutePath
        ).get()
    )
}
