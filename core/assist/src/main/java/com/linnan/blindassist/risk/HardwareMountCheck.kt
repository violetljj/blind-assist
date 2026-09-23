package com.linnan.blindassist.risk

import kotlin.math.abs

/** Session-only rough position check; never changes geometry or claims calibration. */
object HardwareMountCheck {
    data class Capture(val frames: List<List<DemoTofCell>>)
    data class Position(val row: Double, val column: Double, val changedZones: Int, val pairedZones: Int)
    data class Result(val consistent: Boolean, val message: String, val positions: List<Position>)

    fun evaluate(baseline: Capture, center: Capture, left: Capture, right: Capture): Result {
        val captures = listOf(baseline, center, left, right)
        if (captures.any { it.frames.size < 5 }) return Result(false, "数据不足：每个位置需要至少 5 个不同的新帧", emptyList())
        val base = ranges(baseline)
        val positions = mutableListOf<Position>()
        for (capture in captures.drop(1)) {
            val current = ranges(capture)
            val paired = base.keys.intersect(current.keys)
            val changes = paired.mapNotNull { zone ->
                val delta = base.getValue(zone) - current.getValue(zone)
                if (delta >= 200.0) zone to delta else null
            }
            if (paired.size < 8 || changes.size < 2) return Result(false,
                "数据不足：有效配对 ${paired.size}/64，接近至少 200 mm 的区域 ${changes.size}；保持背景不动并重试", positions)
            val weight = changes.sumOf { it.second }
            positions += Position(changes.sumOf { (it.first / 8 / 7.0 * 2 - 1) * it.second } / weight,
                changes.sumOf { (it.first % 8 / 7.0 * 2 - 1) * it.second } / weight, changes.size, paired.size)
        }
        val (c, l, r) = positions
        val good = abs(c.row) <= .35 && abs(c.column) <= .5 && l.row >= .25 && r.row <= -.25 &&
            l.row - c.row >= .25 && c.row - r.row >= .25
        return Result(good, if (good) "粗位置一致：与您声明的中、左、右位置相符。未完成标定。" else
            "位置不一致：检查安装朝向、目标是否进入测距范围，或背景是否移动。", positions)
    }

    private fun ranges(capture: Capture): Map<Int, Double> = (0 until 64).mapNotNull { zone ->
        val values = capture.frames.mapNotNull { frame ->
            if (frame.size != 64 || frame.map { it.zone }.toSet().size != 64) null else
                frame.singleOrNull { it.zone == zone }?.takeIf {
                    it.status == 5 && it.targets > 0 && it.quality == "KNOWN" && it.rangeMm.isFinite() && it.rangeMm > 3
                }?.rangeMm
        }.sorted()
        if (values.size < 5) null else zone to values[values.size / 2]
    }.toMap()
}
