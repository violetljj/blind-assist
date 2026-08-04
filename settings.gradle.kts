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
include(":npu-candidate")
include(":device-benchmark")
include(":hftf-device-canary")
include(":hftf-metric-depth-canary-core")
include(":core:assist")
include(":core:ustrf")
include(":ustrf-shadow-benchmark")
include(":known-height-capture-app")
include(":core:vision")
include(":core:device")
include(":core:ui")
include(":feature:assist")
