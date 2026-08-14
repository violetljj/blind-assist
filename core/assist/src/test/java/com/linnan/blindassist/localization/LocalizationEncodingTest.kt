package com.linnan.blindassist.localization

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalizationEncodingTest {
    @Test
    fun chineseRiskAndSpeechPromptsStayReadable() {
        val frame = FrameSize(1000, 1000)
        val analyzer = RiskAnalyzer()
        val risk = analyzer.analyze(
            listOf(
                Detection(
                    classId = 0,
                    label = "person",
                    confidence = 0.9f,
                    boundingBox = BoundingBox(390f, 120f, 610f, 780f),
                    frameSize = frame
                )
            ),
            frame
        )

        assertEquals("前方很近，放慢", risk.message)
        assertEquals("正前方很近", SpeechStyle.BRIEF.messageFor(risk, AppLanguage.ZH))
        assertEquals("前方很近，放慢", SpeechStyle.STANDARD.messageFor(risk, AppLanguage.ZH))
        assertEquals("正前方有人迫近，请立刻放慢", SpeechStyle.DETAILED.messageFor(risk, AppLanguage.ZH))
    }

    @Test
    fun chineseLocalizationCatalogDoesNotContainMojibake() {
        val catalog = buildList {
            AlertProfile.values().forEach { profile ->
                add(profile.displayName(AppLanguage.ZH))
            }
            AssistScenario.values().forEach { scenario ->
                add(scenario.displayName(AppLanguage.ZH))
                add(scenario.description(AppLanguage.ZH))
            }
            SpeechStyle.values().forEach { style ->
                add(style.displayName(AppLanguage.ZH))
                add(style.description(AppLanguage.ZH))
            }
            VibrationStrength.values().forEach { strength ->
                add(strength.displayName(AppLanguage.ZH))
                add(strength.description(AppLanguage.ZH))
            }
            DailyUsageMode.values().forEach { mode ->
                add(mode.displayName(AppLanguage.ZH))
                add(mode.description(AppLanguage.ZH))
                add(mode.accessibilitySummary(AppLanguage.ZH))
            }
            FeedbackReason.values().forEach { reason ->
                add(reason.displayText(AppLanguage.ZH))
            }
            RiskLevel.values().forEach { level ->
                add(LocalizedText.level(level, AppLanguage.ZH))
            }
            RiskDirection.values().forEach { direction ->
                add(LocalizedText.direction(direction, AppLanguage.ZH))
                add(LocalizedText.direction(direction, AppLanguage.ZH, short = true))
            }
            ProximityBand.values().forEach { proximity ->
                add(LocalizedText.proximity(proximity, AppLanguage.ZH))
            }
            listOf(null, "person", "car", "traffic light", "stop sign", "chair").forEach { label ->
                LocalizedText.objectName(label, AppLanguage.ZH)?.let(::add)
            }
            add(LocalizedText.enabled(true, AppLanguage.ZH))
            add(LocalizedText.enabled(false, AppLanguage.ZH))
            add(LocalizedText.durationText(hasStarted = false, durationMs = 0L, language = AppLanguage.ZH))
            add(LocalizedText.durationText(hasStarted = true, durationMs = 65_000L, language = AppLanguage.ZH))
            add(SpeechStyle.STANDARD.messageFor(RiskResult(RiskLevel.NONE, RiskDirection.NONE, "持续检测中"), AppLanguage.ZH))
        }

        catalog.forEach { text ->
            assertFalse("Mojibake in: $text", containsMojibake(text))
        }
    }

    @Test
    fun bilingualDailyUsageAccessibilityCatalogStaysUsable() {
        AppLanguage.values().forEach { language ->
            DailyUsageMode.values().forEach { mode ->
                val name = mode.displayName(language)
                val description = mode.description(language)
                val summary = mode.accessibilitySummary(language)
                listOf(name, description, summary).forEach { text ->
                    assertFalse("Blank localization for $language", text.isBlank())
                    assertFalse("Replacement character in: $text", text.contains('\uFFFD'))
                    assertFalse("Unexpected control character in: $text", text.any { it.isISOControl() && it != '\n' && it != '\t' })
                }
                assertTrue(summary.contains(name))
            }
        }
    }
    private fun containsMojibake(text: String): Boolean {
        return mojibakeMarkers.any(text::contains)
    }

    private companion object {
        val mojibakeMarkers = listOf(
            "鍓",
            "妯",
            "绠",
            "瀹",
            "鏍",
            "闅",
            "杞",
            "閫",
            "锛",
            "銆",
            "鈥",
            "紝",
            "€"
        )
    }
}
