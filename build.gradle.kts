import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.attribute.BasicFileAttributes

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.android.test) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.kotlin.kapt) apply false
    alias(libs.plugins.hilt.android) apply false
}

val localArtifactsRoot = rootProject.file("artifacts.local")
val localArtifactsPath = localArtifactsRoot.toPath()
val externalLocalArtifacts =
    localArtifactsRoot.isDirectory &&
        (
            Files.isSymbolicLink(localArtifactsPath) ||
                Files.readAttributes(
                    localArtifactsPath,
                    BasicFileAttributes::class.java,
                    LinkOption.NOFOLLOW_LINKS,
                ).isOther
        )
val localGradleBuildRoot = localArtifactsRoot.resolve("work/gradle-build")

// Keep CI and ordinary clones on Gradle's standard module build paths. On the
// managed Windows workspace, artifacts.local is a junction to the artifact
// volume, so generated build payloads belong there as well.
if (externalLocalArtifacts) {
    layout.buildDirectory.set(localGradleBuildRoot.resolve("_root"))
}

subprojects {
    if (externalLocalArtifacts) {
        val localProjectBuildPath = path.removePrefix(":").replace(':', '/')
        layout.buildDirectory.set(localGradleBuildRoot.resolve(localProjectBuildPath))
    }

    configurations.configureEach {
        resolutionStrategy.failOnNonReproducibleResolution()
    }
}
