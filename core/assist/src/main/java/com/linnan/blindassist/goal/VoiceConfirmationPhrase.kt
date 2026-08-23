package com.linnan.blindassist.goal

import java.text.Normalizer

/**
 * Narrow text port for an upstream speech recognizer. This class does not capture audio or run ASR.
 */
object VoiceConfirmationPhrase {
    const val EXPLICIT_PHRASE = "找到了"

    fun normalize(rawPhrase: String): String {
        val compatibilityNormalized = Normalizer.normalize(rawPhrase, Normalizer.Form.NFKC)
        return compatibilityNormalized
            .trim(::isUnicodeSpace)
            .trimEnd { it == '。' || it == '!' }
            .trimEnd(::isUnicodeSpace)
    }

    fun isExplicitConfirmation(rawPhrase: String): Boolean {
        return normalize(rawPhrase) == EXPLICIT_PHRASE
    }

    private fun isUnicodeSpace(char: Char): Boolean {
        return char.isWhitespace() || Character.isSpaceChar(char)
    }
}
