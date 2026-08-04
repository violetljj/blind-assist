package com.linnan.blindassist.vision

/** Thread-safe latest-result holder that fails closed after its capture-time TTL. */
class ExpiringLatestResult<T>(private val ttlNanos: Long) {
    init { require(ttlNanos >= 0L) }

    fun update(value: T, capturedAtNanos: Long, completedAtNanos: Long) {
        require(capturedAtNanos >= 0L && completedAtNanos >= capturedAtNanos)
        synchronized(lock) { latest = Entry(value, capturedAtNanos, completedAtNanos) }
    }

    fun readAt(nowNanos: Long): State<T> = synchronized(lock) {
        require(nowNanos >= 0L)
        val entry = latest ?: return@synchronized State.Unknown(UnknownReason.UNAVAILABLE)
        val age = if (nowNanos <= entry.capturedAtNanos) 0L else nowNanos - entry.capturedAtNanos
        if (age > ttlNanos) State.Unknown(UnknownReason.EXPIRED)
        else State.Fresh(entry.value, entry.capturedAtNanos, entry.completedAtNanos, age)
    }

    sealed interface State<out T> {
        data class Fresh<T>(val value: T, val capturedAtNanos: Long,
            val completedAtNanos: Long, val ageNanos: Long) : State<T>
        data class Unknown(val reason: UnknownReason) : State<Nothing>
    }

    enum class UnknownReason { UNAVAILABLE, EXPIRED }
    private data class Entry<T>(val value: T, val capturedAtNanos: Long, val completedAtNanos: Long)
    private val lock = Any()
    private var latest: Entry<T>? = null
}
