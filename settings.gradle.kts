pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "BlindAssist"
include(":app")
include(":device-benchmark")
include(":core:assist")
include(":core:ustrf")
include(":ustrf-shadow-benchmark")
include(":core:vision")
include(":core:device")
include(":core:ui")
include(":feature:assist")
