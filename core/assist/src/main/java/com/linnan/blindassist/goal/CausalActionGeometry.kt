package com.linnan.blindassist.goal

import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import kotlin.math.PI
import kotlin.math.acos
import kotlin.math.sqrt

enum class ActionGeometryBeliefState {
    UNKNOWN,
    SET_VALUED,
    LOCKED
}

enum class ActionMotionType {
    TRANSLATION,
    ROTATION
}

data class ActionVector3(
    val x: Double,
    val y: Double,
    val z: Double
) {
    init {
        require(x.isFinite() && y.isFinite() && z.isFinite())
    }

    internal operator fun plus(other: ActionVector3) =
        ActionVector3(x + other.x, y + other.y, z + other.z)

    internal operator fun minus(other: ActionVector3) =
        ActionVector3(x - other.x, y - other.y, z - other.z)

    internal operator fun times(scale: Double) = ActionVector3(x * scale, y * scale, z * scale)

    internal fun dot(other: ActionVector3): Double = x * other.x + y * other.y + z * other.z

    internal fun norm(): Double = sqrt(dot(this))

    internal fun normalizedOrNull(): ActionVector3? {
        val magnitude = norm()
        return if (magnitude.isFinite() && magnitude > NUMERIC_EPSILON) this * (1.0 / magnitude) else null
    }

    companion object {
        internal const val NUMERIC_EPSILON = 1e-12
        internal val ZERO = ActionVector3(0.0, 0.0, 0.0)
    }
}

data class ActionPointPair(
    /** Metres in the parent entity's coordinate frame before the micro-motion. */
    val before: ActionVector3,
    /** Metres in the same coordinate frame after the micro-motion. */
    val after: ActionVector3
)

data class CausalActionGeometryEstimate(
    val state: ActionGeometryBeliefState,
    val motionType: ActionMotionType? = null,
    val axis: ActionVector3? = null,
    /** Closest point on the rotation axis to the parent-frame origin; null for translation. */
    val pivotLinePointMeters: ActionVector3? = null,
    val rotationRadians: Double? = null,
    val translationMeters: Double? = null,
    val rmsResidualMeters: Double? = null,
    val pairCount: Int
)

/**
 * Converts a causal before/after RGB-D correspondence set into an action belief.
 *
 * A Horn quaternion fit supplies the rigid transform. Motion below the observability
 * floor and non-rigid/high-residual fits remain set-valued; they never become a
 * guessed action axis.
 */
class CausalActionGeometryEstimator(
    private val minimumPairs: Int = DEFAULT_MINIMUM_PAIRS,
    private val minimumTranslationMeters: Double = DEFAULT_MINIMUM_TRANSLATION_METERS,
    private val minimumRotationRadians: Double = DEFAULT_MINIMUM_ROTATION_DEGREES * PI / 180.0,
    private val maximumRmsResidualMeters: Double = DEFAULT_MAXIMUM_RMS_RESIDUAL_METERS
) {
    init {
        require(minimumPairs >= 3)
        require(minimumTranslationMeters.isFinite() && minimumTranslationMeters > 0.0)
        require(minimumRotationRadians.isFinite() && minimumRotationRadians > 0.0)
        require(maximumRmsResidualMeters.isFinite() && maximumRmsResidualMeters > 0.0)
    }

    fun estimate(pairs: List<ActionPointPair>): CausalActionGeometryEstimate {
        if (pairs.size < minimumPairs) return unknown(pairs.size)

        val beforeCentroid = centroid(pairs.map(ActionPointPair::before))
        val afterCentroid = centroid(pairs.map(ActionPointPair::after))
        val before = pairs.map { it.before - beforeCentroid }
        val after = pairs.map { it.after - afterCentroid }
        val spatialEnergy = before.sumOf { it.dot(it) } / pairs.size
        if (!spatialEnergy.isFinite() || spatialEnergy <= MINIMUM_SPATIAL_ENERGY) {
            return unknown(pairs.size)
        }

        val rotation = fitRotation(before, after) ?: return unknown(pairs.size)
        val translation = afterCentroid - rotation.apply(beforeCentroid)
        val residual = sqrt(
            pairs.indices.sumOf { index ->
                val error = rotation.apply(pairs[index].before) + translation - pairs[index].after
                error.dot(error)
            } / pairs.size
        )
        val quaternion = rotation.quaternion
        val clampedW = quaternion[0].coerceIn(-1.0, 1.0)
        val rotationRadians = 2.0 * acos(kotlin.math.abs(clampedW))
        val translationMeters = translation.norm()

        if (!residual.isFinite() || residual > maximumRmsResidualMeters) {
            return setValued(pairs.size, rotationRadians, translationMeters, residual)
        }
        if (rotationRadians < minimumRotationRadians) {
            val axis = translation.normalizedOrNull()
            return if (translationMeters >= minimumTranslationMeters && axis != null) {
                CausalActionGeometryEstimate(
                    state = ActionGeometryBeliefState.LOCKED,
                    motionType = ActionMotionType.TRANSLATION,
                    axis = axis,
                    rotationRadians = rotationRadians,
                    translationMeters = translationMeters,
                    rmsResidualMeters = residual,
                    pairCount = pairs.size
                )
            } else {
                setValued(pairs.size, rotationRadians, translationMeters, residual)
            }
        }

        val vectorPart = ActionVector3(quaternion[1], quaternion[2], quaternion[3])
        val axis = vectorPart.normalizedOrNull()
            ?: return setValued(pairs.size, rotationRadians, translationMeters, residual)
        val pivot = closestRotationAxisPoint(rotation.matrix, translation, axis)
            ?: return setValued(pairs.size, rotationRadians, translationMeters, residual)
        return CausalActionGeometryEstimate(
            state = ActionGeometryBeliefState.LOCKED,
            motionType = ActionMotionType.ROTATION,
            axis = axis,
            pivotLinePointMeters = pivot,
            rotationRadians = rotationRadians,
            translationMeters = translationMeters,
            rmsResidualMeters = residual,
            pairCount = pairs.size
        )
    }

    private fun centroid(points: List<ActionVector3>): ActionVector3 =
        points.fold(ActionVector3.ZERO, ActionVector3::plus) * (1.0 / points.size)

    private fun fitRotation(
        before: List<ActionVector3>,
        after: List<ActionVector3>
    ): RotationFit? {
        val covariance = Array(3) { DoubleArray(3) }
        before.indices.forEach { index ->
            val p = before[index]
            val q = after[index]
            val pv = doubleArrayOf(p.x, p.y, p.z)
            val qv = doubleArrayOf(q.x, q.y, q.z)
            for (row in 0..2) for (column in 0..2) covariance[row][column] += pv[row] * qv[column]
        }
        val sxx = covariance[0][0]
        val sxy = covariance[0][1]
        val sxz = covariance[0][2]
        val syx = covariance[1][0]
        val syy = covariance[1][1]
        val syz = covariance[1][2]
        val szx = covariance[2][0]
        val szy = covariance[2][1]
        val szz = covariance[2][2]
        val horn = arrayOf(
            doubleArrayOf(sxx + syy + szz, syz - szy, szx - sxz, sxy - syx),
            doubleArrayOf(syz - szy, sxx - syy - szz, sxy + syx, szx + sxz),
            doubleArrayOf(szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy),
            doubleArrayOf(sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz)
        )
        // Shift just past the spectral floor: a unit shift would dominate metre-scale
        // covariance and make power iteration converge to the seed instead of the fit.
        val shift = sqrt(horn.sumOf { row -> row.sumOf { it * it } }) +
            ActionVector3.NUMERIC_EPSILON
        var q = doubleArrayOf(1.0, 0.17, -0.11, 0.07)
        repeat(80) {
            val next = DoubleArray(4) { row ->
                horn[row].indices.sumOf { column -> horn[row][column] * q[column] } + shift * q[row]
            }
            val norm = sqrt(next.sumOf { it * it })
            if (!norm.isFinite() || norm <= ActionVector3.NUMERIC_EPSILON) return null
            q = DoubleArray(4) { next[it] / norm }
        }
        if (q[0] < 0.0) q = DoubleArray(4) { -q[it] }
        val w = q[0]
        val x = q[1]
        val y = q[2]
        val z = q[3]
        val matrix = arrayOf(
            doubleArrayOf(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            doubleArrayOf(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            doubleArrayOf(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))
        )
        return RotationFit(q, matrix)
    }

    /** Gauge-fixes the rank-2 pivot equation with the point closest to the origin. */
    private fun closestRotationAxisPoint(
        rotation: Array<DoubleArray>,
        translation: ActionVector3,
        axis: ActionVector3
    ): ActionVector3? {
        val a = Array(3) { row -> DoubleArray(3) { column -> (if (row == column) 1.0 else 0.0) - rotation[row][column] } }
        val normal = Array(3) { row ->
            DoubleArray(3) { column ->
                (0..2).sumOf { k -> a[k][row] * a[k][column] } + component(axis, row) * component(axis, column)
            }
        }
        val tv = doubleArrayOf(translation.x, translation.y, translation.z)
        val right = DoubleArray(3) { row -> (0..2).sumOf { k -> a[k][row] * tv[k] } }
        val solution = solve3x3(normal, right) ?: return null
        return ActionVector3(solution[0], solution[1], solution[2])
    }

    private fun solve3x3(matrix: Array<DoubleArray>, right: DoubleArray): DoubleArray? {
        val augmented = Array(3) { row -> DoubleArray(4) { column -> if (column < 3) matrix[row][column] else right[row] } }
        for (column in 0..2) {
            val pivot = (column..2).maxBy { row -> kotlin.math.abs(augmented[row][column]) }
            if (kotlin.math.abs(augmented[pivot][column]) <= ActionVector3.NUMERIC_EPSILON) return null
            val swap = augmented[column]
            augmented[column] = augmented[pivot]
            augmented[pivot] = swap
            val divisor = augmented[column][column]
            for (j in column..3) augmented[column][j] /= divisor
            for (row in 0..2) {
                if (row == column) continue
                val factor = augmented[row][column]
                for (j in column..3) augmented[row][j] -= factor * augmented[column][j]
            }
        }
        return DoubleArray(3) { augmented[it][3] }
    }

    private fun component(vector: ActionVector3, index: Int): Double =
        when (index) {
            0 -> vector.x
            1 -> vector.y
            else -> vector.z
        }

    private fun unknown(pairCount: Int) = CausalActionGeometryEstimate(
        state = ActionGeometryBeliefState.UNKNOWN,
        pairCount = pairCount
    )

    private fun setValued(
        pairCount: Int,
        rotationRadians: Double,
        translationMeters: Double,
        residualMeters: Double
    ) = CausalActionGeometryEstimate(
        state = ActionGeometryBeliefState.SET_VALUED,
        rotationRadians = rotationRadians,
        translationMeters = translationMeters,
        rmsResidualMeters = residualMeters,
        pairCount = pairCount
    )

    private data class RotationFit(
        val quaternion: DoubleArray,
        val matrix: Array<DoubleArray>
    ) {
        fun apply(vector: ActionVector3): ActionVector3 = ActionVector3(
            x = matrix[0][0] * vector.x + matrix[0][1] * vector.y + matrix[0][2] * vector.z,
            y = matrix[1][0] * vector.x + matrix[1][1] * vector.y + matrix[1][2] * vector.z,
            z = matrix[2][0] * vector.x + matrix[2][1] * vector.y + matrix[2][2] * vector.z
        )
    }

    companion object {
        const val DEFAULT_MINIMUM_PAIRS = 6
        const val DEFAULT_MINIMUM_TRANSLATION_METERS = 0.005
        const val DEFAULT_MINIMUM_ROTATION_DEGREES = 2.0
        const val DEFAULT_MAXIMUM_RMS_RESIDUAL_METERS = 0.015
        private const val MINIMUM_SPATIAL_ENERGY = 1e-6
    }
}

data class CausalActionGeometryEvidence(
    val sourceContractId: String,
    val sourceId: String,
    val goalId: String,
    val sessionId: String,
    val parentBindingId: String,
    val previousFrame: FrameStamp,
    val currentFrame: FrameStamp,
    val availableAtNs: Long,
    val validUntilNs: Long,
    val availabilityClockDomain: FrameClockDomain,
    val pairs: List<ActionPointPair>
)

enum class CausalActionGeometryDisposition {
    EVIDENCE_ABSENT,
    SOURCE_NOT_ADMITTED,
    IDENTITY_MISMATCH,
    CURRENT_FRAME_MISMATCH,
    FRAME_PAIR_INVALID,
    CLOCK_DOMAIN_MISMATCH,
    EVIDENCE_NOT_AVAILABLE,
    EVIDENCE_STALE,
    ADMITTED
}

data class CausalActionGeometryObservation(
    val disposition: CausalActionGeometryDisposition,
    val state: ActionGeometryBeliefState = ActionGeometryBeliefState.UNKNOWN,
    val goalId: String? = null,
    val sessionId: String? = null,
    val parentBindingId: String? = null,
    val currentFrame: FrameStamp? = null,
    val sourceId: String? = null,
    val estimate: CausalActionGeometryEstimate? = null
) {
    val admitted: Boolean get() = disposition == CausalActionGeometryDisposition.ADMITTED
}

data class CausalActionGeometrySourceIdentity(
    val sourceContractId: String,
    val sourceId: String
)

/** Fail-closed identity, frame, clock and freshness admission before motion fitting. */
class CausalActionGeometryAdmitter(
    admittedSources: Set<CausalActionGeometrySourceIdentity>,
    private val estimator: CausalActionGeometryEstimator = CausalActionGeometryEstimator()
) {
    private val admittedSources = admittedSources.toSet()

    fun evaluate(
        evidence: CausalActionGeometryEvidence?,
        goalId: String,
        sessionId: String,
        parentBindingId: String,
        currentFrame: FrameStamp,
        decisionAtNs: Long,
        decisionClockDomain: FrameClockDomain
    ): CausalActionGeometryObservation {
        if (evidence == null) return observation(CausalActionGeometryDisposition.EVIDENCE_ABSENT)
        if (CausalActionGeometrySourceIdentity(evidence.sourceContractId, evidence.sourceId) !in admittedSources) {
            return observation(CausalActionGeometryDisposition.SOURCE_NOT_ADMITTED, evidence)
        }
        if (evidence.goalId != goalId || evidence.sessionId != sessionId ||
            evidence.parentBindingId != parentBindingId || goalId.isBlank() || sessionId.isBlank() ||
            parentBindingId.isBlank()
        ) {
            return observation(CausalActionGeometryDisposition.IDENTITY_MISMATCH, evidence)
        }
        if (evidence.currentFrame != currentFrame) {
            return observation(CausalActionGeometryDisposition.CURRENT_FRAME_MISMATCH, evidence)
        }
        if (!validFramePair(evidence.previousFrame, evidence.currentFrame)) {
            return observation(CausalActionGeometryDisposition.FRAME_PAIR_INVALID, evidence)
        }
        if (evidence.availabilityClockDomain != decisionClockDomain ||
            evidence.currentFrame.clockDomain != decisionClockDomain ||
            decisionClockDomain == FrameClockDomain.CAMERA_HARDWARE_UNMAPPED
        ) {
            return observation(CausalActionGeometryDisposition.CLOCK_DOMAIN_MISMATCH, evidence)
        }
        if (evidence.availableAtNs < evidence.currentFrame.capturedAtNs || decisionAtNs < evidence.availableAtNs) {
            return observation(CausalActionGeometryDisposition.EVIDENCE_NOT_AVAILABLE, evidence)
        }
        if (evidence.validUntilNs < evidence.availableAtNs || decisionAtNs > evidence.validUntilNs) {
            return observation(CausalActionGeometryDisposition.EVIDENCE_STALE, evidence)
        }
        val estimate = estimator.estimate(evidence.pairs)
        return observation(CausalActionGeometryDisposition.ADMITTED, evidence, estimate)
    }

    private fun validFramePair(previous: FrameStamp, current: FrameStamp): Boolean =
        previous.sourceId == current.sourceId &&
            previous.coordinateFrame == current.coordinateFrame &&
            previous.clockDomain == current.clockDomain &&
            previous.frameId < current.frameId &&
            previous.capturedAtNs < current.capturedAtNs

    private fun observation(
        disposition: CausalActionGeometryDisposition,
        evidence: CausalActionGeometryEvidence? = null,
        estimate: CausalActionGeometryEstimate? = null
    ) = CausalActionGeometryObservation(
        disposition = disposition,
        state = estimate?.state ?: ActionGeometryBeliefState.UNKNOWN,
        goalId = evidence?.goalId,
        sessionId = evidence?.sessionId,
        parentBindingId = evidence?.parentBindingId,
        currentFrame = evidence?.currentFrame,
        sourceId = evidence?.sourceId,
        estimate = estimate
    )

    companion object {
        const val CONTRACT_ID = "blindassist_causal_action_geometry_input_v1"
        const val PAIRED_RGBD_SOURCE_ID = "paired_rgbd_functional_part_micro_motion_v1"

        fun pairedRgbdSource() = CausalActionGeometryAdmitter(
            admittedSources = setOf(
                CausalActionGeometrySourceIdentity(CONTRACT_ID, PAIRED_RGBD_SOURCE_ID)
            )
        )
    }
}

enum class GoalEndpointCondition {
    UNKNOWN,
    NOT_READY,
    READY
}

data class GoalEndpointEvidence(
    val sourceContractId: String,
    val goalId: String,
    val sessionId: String,
    val parentBindingId: String,
    val currentFrame: FrameStamp,
    val availableAtNs: Long,
    val validUntilNs: Long,
    val availabilityClockDomain: FrameClockDomain,
    val position: GoalEndpointCondition,
    val visibility: GoalEndpointCondition,
    val grounding: GoalEndpointCondition,
    val orientation: GoalEndpointCondition,
    val reachability: GoalEndpointCondition,
    val actionGeometry: CausalActionGeometryObservation
)

enum class GoalHandoffReadinessBlock {
    SOURCE_NOT_ADMITTED,
    IDENTITY_MISMATCH,
    CURRENT_FRAME_MISMATCH,
    CLOCK_DOMAIN_MISMATCH,
    EVIDENCE_NOT_AVAILABLE,
    EVIDENCE_STALE,
    POSITION_NOT_READY,
    VISIBILITY_NOT_READY,
    GROUNDING_NOT_READY,
    ORIENTATION_NOT_READY,
    REACHABILITY_NOT_READY,
    ACTION_GEOMETRY_NOT_LOCKED
}

data class GoalHandoffReadinessReceipt(
    val sourceContractId: String,
    val goalId: String,
    val sessionId: String,
    val parentBindingId: String,
    val currentFrameId: Long,
    val actionSourceId: String,
    val actionMotionType: ActionMotionType,
    val evaluatedAtNs: Long
)

sealed interface GoalHandoffReadinessDecision {
    data class Ready(val receipt: GoalHandoffReadinessReceipt) : GoalHandoffReadinessDecision
    data class Blocked(val reason: GoalHandoffReadinessBlock) : GoalHandoffReadinessDecision
}

/** Joins the five endpoint conditions without giving any of them completion authority. */
object GoalHandoffReadinessGuard {
    fun evaluate(
        evidence: GoalEndpointEvidence,
        expectedGoalId: String,
        expectedSessionId: String,
        expectedParentBindingId: String,
        currentFrame: FrameStamp,
        decisionAtNs: Long,
        decisionClockDomain: FrameClockDomain
    ): GoalHandoffReadinessDecision {
        if (evidence.sourceContractId != CONTRACT_ID) {
            return blocked(GoalHandoffReadinessBlock.SOURCE_NOT_ADMITTED)
        }
        if (evidence.goalId != expectedGoalId ||
            evidence.sessionId != expectedSessionId || evidence.parentBindingId != expectedParentBindingId ||
            expectedGoalId.isBlank() || expectedSessionId.isBlank() || expectedParentBindingId.isBlank()
        ) return blocked(GoalHandoffReadinessBlock.IDENTITY_MISMATCH)
        if (evidence.currentFrame != currentFrame) {
            return blocked(GoalHandoffReadinessBlock.CURRENT_FRAME_MISMATCH)
        }
        if (evidence.availabilityClockDomain != decisionClockDomain ||
            evidence.currentFrame.clockDomain != decisionClockDomain ||
            decisionClockDomain == FrameClockDomain.CAMERA_HARDWARE_UNMAPPED
        ) return blocked(GoalHandoffReadinessBlock.CLOCK_DOMAIN_MISMATCH)
        if (evidence.availableAtNs < evidence.currentFrame.capturedAtNs || decisionAtNs < evidence.availableAtNs) {
            return blocked(GoalHandoffReadinessBlock.EVIDENCE_NOT_AVAILABLE)
        }
        if (evidence.validUntilNs < evidence.availableAtNs || decisionAtNs > evidence.validUntilNs) {
            return blocked(GoalHandoffReadinessBlock.EVIDENCE_STALE)
        }
        listOf(
            GoalHandoffReadinessBlock.POSITION_NOT_READY to evidence.position,
            GoalHandoffReadinessBlock.VISIBILITY_NOT_READY to evidence.visibility,
            GoalHandoffReadinessBlock.GROUNDING_NOT_READY to evidence.grounding,
            GoalHandoffReadinessBlock.ORIENTATION_NOT_READY to evidence.orientation,
            GoalHandoffReadinessBlock.REACHABILITY_NOT_READY to evidence.reachability
        ).firstOrNull { (_, state) -> state != GoalEndpointCondition.READY }
            ?.let { (reason, _) -> return blocked(reason) }

        val action = evidence.actionGeometry
        val estimate = action.estimate
        if (!action.admitted || action.state != ActionGeometryBeliefState.LOCKED ||
            action.goalId != evidence.goalId || action.sessionId != evidence.sessionId ||
            action.parentBindingId != evidence.parentBindingId ||
            action.currentFrame != evidence.currentFrame ||
            action.sourceId.isNullOrBlank() || estimate?.motionType == null
        ) return blocked(GoalHandoffReadinessBlock.ACTION_GEOMETRY_NOT_LOCKED)

        return GoalHandoffReadinessDecision.Ready(
            GoalHandoffReadinessReceipt(
                sourceContractId = evidence.sourceContractId,
                goalId = evidence.goalId,
                sessionId = evidence.sessionId,
                parentBindingId = evidence.parentBindingId,
                currentFrameId = evidence.currentFrame.frameId,
                actionSourceId = requireNotNull(action.sourceId),
                actionMotionType = estimate.motionType,
                evaluatedAtNs = decisionAtNs
            )
        )
    }

    private fun blocked(reason: GoalHandoffReadinessBlock) = GoalHandoffReadinessDecision.Blocked(reason)

    const val CONTRACT_ID = "blindassist_goal_endpoint_readiness_input_v1"
}
