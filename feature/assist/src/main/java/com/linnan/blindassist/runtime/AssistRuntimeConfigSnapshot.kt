package com.linnan.blindassist.runtime

import java.util.concurrent.atomic.AtomicReference

internal class AssistRuntimeConfigSnapshot(initialConfig: AssistRuntimeConfig) {
    private val reference = AtomicReference(initialConfig)

    fun get(): AssistRuntimeConfig = reference.get()

    fun update(config: AssistRuntimeConfig): AssistRuntimeConfig {
        reference.set(config)
        return config
    }
}
