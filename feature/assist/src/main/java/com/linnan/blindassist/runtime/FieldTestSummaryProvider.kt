package com.linnan.blindassist.runtime

import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.session.SessionSummary
import com.linnan.blindassist.ui.FieldTestSummaryMapper
import com.linnan.blindassist.ui.compose.FieldTestSummaryUiState

internal class FieldTestSummaryProvider(
    private val coordinator: AssistSessionCoordinator
) {
    fun current(active: Boolean, runtimeConfig: AssistRuntimeConfig): FieldTestSummaryUiState {
        return fromSummary(coordinator.sessionSummary(), active, runtimeConfig)
    }

    fun fromSummary(
        summary: SessionSummary,
        active: Boolean,
        runtimeConfig: AssistRuntimeConfig
    ): FieldTestSummaryUiState {
        return FieldTestSummaryMapper.fromSummary(
            summary,
            active,
            runtimeConfig.alertProfile,
            runtimeConfig.assistScenario,
            runtimeConfig.appLanguage
        )
    }
}
