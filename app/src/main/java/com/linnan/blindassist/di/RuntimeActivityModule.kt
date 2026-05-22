package com.linnan.blindassist.di

import android.content.Context
import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.vision.ObjectDetector
import com.linnan.blindassist.vision.TfliteYoloDetector
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.components.ActivityComponent
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.android.scopes.ActivityScoped

@Module
@InstallIn(ActivityComponent::class)
object RuntimeActivityModule {
    @Provides
    @ActivityScoped
    fun provideFeedbackController(@ApplicationContext context: Context): FeedbackController {
        return FeedbackController(context)
    }

    @Provides
    @ActivityScoped
    fun provideObjectDetector(@ApplicationContext context: Context): ObjectDetector {
        return TfliteYoloDetector(context)
    }

    @Provides
    @ActivityScoped
    fun provideAssistSessionCoordinator(
        feedbackController: FeedbackController
    ): AssistSessionCoordinator {
        return AssistSessionCoordinator(feedbackGateway = feedbackController)
    }
}
