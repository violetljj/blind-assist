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
include(":core:assist")
include(":core:vision")
include(":core:device")
include(":core:ui")
include(":feature:assist")
