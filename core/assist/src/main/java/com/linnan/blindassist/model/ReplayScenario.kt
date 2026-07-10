package com.linnan.blindassist.model

/** Stable offline scenes bundled with debug builds for replaying the vision pipeline. */
enum class ReplayScenario(val assetPath: String) {
    HIGH_CENTER("replay/000000001000.jpg"),
    MEDIUM_RIGHT("replay/000000019402.jpg"),
    LOW_CENTER("replay/000000574520.jpg"),
    NONE("replay/000000015272.jpg")
}
