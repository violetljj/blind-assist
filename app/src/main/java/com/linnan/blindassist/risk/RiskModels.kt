package com.linnan.blindassist.risk

import com.linnan.blindassist.model.Detection

enum class RiskLevel {
    NONE,
    LOW,
    MEDIUM,
    HIGH
}

enum class RiskDirection {
    NONE,
    LEFT,
    CENTER,
    RIGHT
}

data class RiskResult(
    val level: RiskLevel,
    val direction: RiskDirection,
    val message: String,
    val sourceDetection: Detection? = null
)
