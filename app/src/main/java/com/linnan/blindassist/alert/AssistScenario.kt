package com.linnan.blindassist.alert

enum class AssistScenario(
    val storageValue: String,
    val displayName: String,
    val description: String
) {
    GENERAL("general", "通用", "保持当前标准策略"),
    INDOOR("indoor", "室内慢行", "减少室内近距误触发"),
    CORRIDOR("corridor", "走廊通行", "更早关注正前方持续风险"),
    CROWDED("crowded", "密集区域", "降低密集环境提醒疲劳"),
    OUTDOOR_SLOW("outdoor_slow", "户外慢行", "让户外慢行提醒更清晰");

    fun next(): AssistScenario {
        val all = values()
        return all[(ordinal + 1) % all.size]
    }

    companion object {
        fun fromStorageValue(value: String?): AssistScenario {
            return values().firstOrNull { it.storageValue == value } ?: GENERAL
        }
    }
}
