package com.linnan.blindassist.localization

enum class AppLanguage(
    val storageValue: String,
    val chineseName: String,
    val englishName: String
) {
    ZH("zh", "中文", "Chinese"),
    EN("en", "English", "English");

    fun displayName(current: AppLanguage = this): String {
        return if (current == EN) englishName else chineseName
    }

    companion object {
        fun fromStorageValue(value: String?): AppLanguage {
            return values().firstOrNull { it.storageValue == value } ?: ZH
        }
    }
}
