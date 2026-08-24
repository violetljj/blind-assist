package com.linnan.blindassist.ui.compose

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.linnan.blindassist.goal.GoalHandoffState
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.GoalHandoffLocalizedText

/** Stateless product surface. State flows down and explicit user-confirmation events flow up. */
@Composable
fun GoalHandoffCard(
    state: GoalHandoffState,
    language: AppLanguage,
    modifier: Modifier = Modifier,
    onUserConfirmed: () -> Unit
) {
    if (state is GoalHandoffState.Inactive) return

    val message = requireNotNull(GoalHandoffLocalizedText.message(state, language))
    val stateDescription = requireNotNull(
        GoalHandoffLocalizedText.stateDescription(state, language)
    )
    Card(
        modifier = modifier
            .fillMaxWidth()
            .testTag("goal_handoff_card")
            .semantics {
                liveRegion = LiveRegionMode.Polite
                this.stateDescription = stateDescription
            },
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer
        ),
        shape = RoundedCornerShape(22.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = message,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.semantics { heading() }
            )
            if (state is GoalHandoffState.HandoffReady) {
                Spacer(Modifier.height(16.dp))
                Button(
                    onClick = onUserConfirmed,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 64.dp)
                        .testTag("goal_handoff_confirm_button")
                        .semantics {
                            contentDescription = GoalHandoffLocalizedText
                                .confirmationButtonDescription(language)
                        }
                ) {
                    Text(
                        text = GoalHandoffLocalizedText.confirmationButton(language),
                        style = MaterialTheme.typography.titleLarge
                    )
                }
            }
        }
    }
}
