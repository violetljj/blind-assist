package com.linnan.blindassist.feedback

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.risk.RiskResult

interface FeedbackGateway {
    fun resetSession()

    fun notify(
        risk: RiskResult,
        profile: AlertProfile,
        scenario: AssistScenario
    ): FeedbackDecision
}
