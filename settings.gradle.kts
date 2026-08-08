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
include(":hftf-depth-demo-app")
include(":hftf-metric-depth-canary-core")
include(":core:assist")
include(":core:ustrf")
include(":ustrf-shadow-benchmark")
include(":known-height-capture-app")
include(":core:vision")
include(":core:device")
include(":core:ui")
include(":feature:assist")

project(":device-benchmark").projectDir = file("apps/benchmarks/device-benchmark")
project(":ustrf-shadow-benchmark").projectDir = file("apps/benchmarks/ustrf-shadow-benchmark")
project(":hftf-device-canary").projectDir = file("apps/canaries/hftf-device-canary")
project(":hftf-metric-depth-canary-core").projectDir = file("apps/canaries/hftf-metric-depth-canary-core")
project(":hftf-depth-demo-app").projectDir = file("apps/demos/hftf-depth-demo-app")
project(":known-height-capture-app").projectDir = file("apps/demos/known-height-capture-app")
project(":npu-candidate").projectDir = file("apps/candidates/npu-candidate")
