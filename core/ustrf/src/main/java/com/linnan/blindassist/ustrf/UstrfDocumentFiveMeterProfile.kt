package com.linnan.blindassist.ustrf

/**
 * The document-target local BEV profile: 0–5 m ahead at 0.5 m cells, with a 5-candidate
 * body-envelope corridor search. It is deliberately an offline/shadow factory, not an Android
 * production configuration.
 */
object UstrfDocumentFiveMeterProfile {
    const val CELL_METERS = .5f
    const val HALF_WIDTH_CELLS = 2
    const val HORIZON_CELLS = 10
    const val BODY_CAPSULE_HALF_WIDTH_CELLS = 1

    fun safetySession(): UstrfSafetySession = UstrfSafetySession(
        fieldBuilder = UstrfRiskFieldBuilder(
            UstrfRiskFieldConfig(
                halfWidthCells = HALF_WIDTH_CELLS,
                horizonCells = HORIZON_CELLS,
                cellMeters = CELL_METERS
            )
        ),
        planner = UstrfCorridorPlanner(
            horizonCells = HORIZON_CELLS,
            capsuleHalfWidthCells = BODY_CAPSULE_HALF_WIDTH_CELLS,
            fixedCandidateOffsets = listOf(-2, -1, 0, 1, 2)
        ),
        supervisor = UstrfSafetySupervisor(centralHorizonCells = HORIZON_CELLS),
        structuredOutputMapper = UstrfStructuredSafetyOutputMapper(
            cellMeters = CELL_METERS,
            lookaheadMeters = HORIZON_CELLS * CELL_METERS
        )
    )

    fun perceptionAssembler(): UstrfPerceptionAssembler = UstrfPerceptionAssembler(
        geometryProjector = UstrfGeometryProjector(
            cellSizeMeters = CELL_METERS,
            halfWidthCells = HALF_WIDTH_CELLS,
            horizonCells = HORIZON_CELLS
        )
    )

    fun egoMotionPromoter(): UstrfEgoCompensatedMotionPromoter = UstrfEgoCompensatedMotionPromoter(
        cellMeters = CELL_METERS,
        halfWidthCells = HALF_WIDTH_CELLS,
        horizonCells = HORIZON_CELLS
    )
}
