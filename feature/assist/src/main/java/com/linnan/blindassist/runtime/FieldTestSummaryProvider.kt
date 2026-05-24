package com.linnan.blindassist.runtime

import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.ui.FieldTestSummaryMapper
import com.linnan.blindassist.ui.compose.FieldTestSummaryUiState

internal class FieldTestSummaryProvider(
    private val coordinator: AssistSessionCoordinator
) {
    fun current(active: Boolean, runtimeConfig: AssistRuntimeConfig): FieldTestSummaryUiState {
        return FieldTestSummaryMapper.fromSummary(
            coordinator.sessionSummary(),
            active,
            runtimeConfig.alertProfile,
            runtimeConfig.assistScenario,
            runtimeConfig.appLanguage
        )
    }
}
