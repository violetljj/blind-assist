package com.linnan.blindassist.ustrf

/**
 * The document-target local BEV profile: 0–5 m ahead at 0.5 m cells, with a 5-candidate
 * body-envelope corridor search. It is deliberately an offline/shadow factory, not an Android
 * production configuration.
 */
object UstrfDocumentFiveMeterProfile {
    val GRID_SPEC: UstrfGridSpec = UstrfGridSpec.DOCUMENT_FIVE_METER
    const val CELL_METERS = .5f
    const val HALF_WIDTH_CELLS = 3
    const val HORIZON_CELLS = 10
    const val BODY_CAPSULE_HALF_WIDTH_CELLS = 1

    fun safetySession(): UstrfSafetySession = UstrfSafetySession(
        fieldBuilder = UstrfRiskFieldBuilder(
            UstrfRiskFieldConfig(gridSpec = GRID_SPEC)
        ),
        planner = UstrfCorridorPlanner(gridSpec = GRID_SPEC),
        supervisor = UstrfSafetySupervisor(centralHorizonCells = HORIZON_CELLS),
        structuredOutputMapper = UstrfStructuredSafetyOutputMapper(gridSpec = GRID_SPEC)
    )

    fun perceptionAssembler(): UstrfPerceptionAssembler = UstrfPerceptionAssembler(
        geometryProjector = UstrfGeometryProjector(gridSpec = GRID_SPEC)
    )

    fun egoMotionPromoter(): UstrfEgoCompensatedMotionPromoter =
        UstrfEgoCompensatedMotionPromoter(gridSpec = GRID_SPEC)
}
