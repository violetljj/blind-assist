package com.linnan.blindassist.benchmark

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video

/** Test-APK-only complete five-channel implementation of the Sparse-LK candidate contract. */
data class SparseLkGeometryVector(
    val success: Boolean,
    val inlierRatio: Float,
    val validCorridorFraction: Float,
    val corridorResidual: Float,
    val lowerCorridorResidual: Float
)

class SparseLkGeometryProbe(
    private val size: Int = SIZE,
    private val maxCorners: Int = MAX_CORNERS
) : AutoCloseable {
    private val corridorMask: Mat
    private val lowerMask: Mat
    private val corridorPixelCount: Int

    init {
        require(maxCorners >= MIN_POINTS) { "maxCorners must support homography" }
        val corridor = ByteArray(size * size)
        val lower = ByteArray(size * size)
        for (yIndex in 0 until size) for (xIndex in 0 until size) {
            val y = yIndex.toFloat() / (size - 1).coerceAtLeast(1); val x = xIndex.toFloat() / (size - 1).coerceAtLeast(1)
            val halfWidth = 0.16f + (((y - 0.35f) / 0.65f).coerceIn(0f, 1f) * 0.34f)
            if (y >= 0.35f && kotlin.math.abs(x - 0.5f) <= halfWidth) {
                corridor[yIndex * size + xIndex] = 0xFF.toByte()
                if (y >= 0.58f) lower[yIndex * size + xIndex] = 0xFF.toByte()
            }
        }
        corridorMask = Mat(size, size, CvType.CV_8UC1).also { it.put(0, 0, corridor) }
        lowerMask = Mat(size, size, CvType.CV_8UC1).also { it.put(0, 0, lower) }
        corridorPixelCount = Core.countNonZero(corridorMask)
    }

    fun measure(previous: Mat, current: Mat): SparseLkGeometryVector {
        require(previous.rows() == size && previous.cols() == size && current.rows() == size && current.cols() == size) { "expected ${size}x$size luma Mats" }
        val corners = MatOfPoint(); val previousPoints = MatOfPoint2f(); val currentPoints = MatOfPoint2f(); val status = MatOfByte(); val error = MatOfFloat(); val mask = Mat()
        var inliers: Mat? = null; var homography: Mat? = null; var keptPrevious: MatOfPoint2f? = null; var keptCurrent: MatOfPoint2f? = null; var warped: Mat? = null; var valid: Mat? = null; var validBinary: Mat? = null; var residual: Mat? = null; var ones: Mat? = null; var effectiveCorridor: Mat? = null; var effectiveLower: Mat? = null
        try {
            Imgproc.goodFeaturesToTrack(previous, corners, maxCorners, QUALITY_LEVEL, MIN_DISTANCE, mask, BLOCK_SIZE, false, HARRIS_K)
            if (corners.rows() < MIN_POINTS) return ZERO
            previousPoints.fromArray(*corners.toArray())
            Video.calcOpticalFlowPyrLK(previous, current, previousPoints, currentPoints, status, error, Size(WINDOW.toDouble(), WINDOW.toDouble()), MAX_LEVEL, TermCriteria(TermCriteria.COUNT or TermCriteria.EPS, TERM_COUNT, TERM_EPS), 0, MIN_EIGEN_THRESHOLD)
            val before = previousPoints.toArray(); val after = currentPoints.toArray(); val flags = status.toArray(); val keptBefore = ArrayList<Point>(); val keptAfter = ArrayList<Point>()
            flags.forEachIndexed { index, flag -> if (flag.toInt() != 0) { keptBefore += before[index]; keptAfter += after[index] } }
            if (keptBefore.size < MIN_POINTS) return ZERO
            keptPrevious = MatOfPoint2f(*keptBefore.toTypedArray()); keptCurrent = MatOfPoint2f(*keptAfter.toTypedArray()); inliers = Mat()
            homography = Calib3d.findHomography(keptPrevious!!, keptCurrent!!, Calib3d.RANSAC, RANSAC_REPROJECTION, inliers)
            if (homography!!.empty()) return ZERO
            warped = Mat(); valid = Mat(); residual = Mat()
            Imgproc.warpPerspective(previous, warped, homography, Size(size.toDouble(), size.toDouble()))
            ones = Mat.ones(previous.size(), previous.type())
            Imgproc.warpPerspective(ones, valid, homography, Size(size.toDouble(), size.toDouble()))
            Core.absdiff(warped, current, residual)
            validBinary = Mat(); effectiveCorridor = Mat(); effectiveLower = Mat()
            Imgproc.threshold(valid, validBinary, 0.0, 255.0, Imgproc.THRESH_BINARY)
            Core.bitwise_and(validBinary, corridorMask, effectiveCorridor)
            Core.bitwise_and(validBinary, lowerMask, effectiveLower)
            val validCorridorCount = Core.countNonZero(effectiveCorridor)
            val validLowerCount = Core.countNonZero(effectiveLower)
            if (validCorridorCount == 0 || validLowerCount == 0) return ZERO
            return SparseLkGeometryVector(
                success = true,
                inlierRatio = Core.countNonZero(inliers!!).toFloat() / keptBefore.size,
                validCorridorFraction = validCorridorCount.toFloat() / corridorPixelCount,
                corridorResidual = (Core.mean(residual, effectiveCorridor).`val`[0] / 255.0).toFloat(),
                lowerCorridorResidual = (Core.mean(residual, effectiveLower).`val`[0] / 255.0).toFloat()
            )
        } finally {
            corners.release(); previousPoints.release(); currentPoints.release(); status.release(); error.release(); mask.release(); inliers?.release(); homography?.release(); keptPrevious?.release(); keptCurrent?.release(); warped?.release(); valid?.release(); validBinary?.release(); residual?.release(); ones?.release(); effectiveCorridor?.release(); effectiveLower?.release()
        }
    }

    /**
     * The probe is deliberately test-APK-only, but it still owns native masks.
     * Make that ownership explicit so repeated device benchmark branches do not
     * retain OpenCV allocations between runs.
     */
    override fun close() {
        corridorMask.release()
        lowerMask.release()
    }

    private companion object {
        const val SIZE = 320; const val MAX_CORNERS = 300; const val QUALITY_LEVEL = 0.01; const val MIN_DISTANCE = 6.0; const val BLOCK_SIZE = 7; const val HARRIS_K = 0.04; const val MIN_POINTS = 8; const val WINDOW = 21; const val MAX_LEVEL = 3; const val TERM_COUNT = 20; const val TERM_EPS = 0.03; const val MIN_EIGEN_THRESHOLD = 1e-4; const val RANSAC_REPROJECTION = 3.0
        val ZERO = SparseLkGeometryVector(false, 0f, 0f, 0f, 0f)
    }
}
