package com.linnan.blindassist.goalcapture

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import java.time.Instant

@RunWith(JUnit4::class)
class GoalCaptureContractTest {
    @Test
    fun canonicalJsonMatchesPythonContractEncoding() {
        val value = mapOf(
            "b" to 1,
            "a" to "entrance",
            "c" to listOf(true, -2.5, null),
        )

        assertThat(CanonicalJson.encode(value)).isEqualTo("{\"a\":\"entrance\",\"b\":1,\"c\":[true,-2.5,null]}\n")
        assertThat(CanonicalJson.sha256(value)).isEqualTo("1523f69c462a1c424d8571e8ad18255e09cda1e2f0a0d0f82e2e3c3d4c573783")
    }

    @Test
    fun receiptRequiresFullFrozenRosterAndCarriesNoAuthorization() {
        val plan = plan()
        val captures = captures(plan)

        val receipt = CaptureReceiptBuilder.build(plan, captures, Instant.parse("2026-08-22T09:10:00Z"))

        assertThat(receipt["episode_count"]).isEqualTo(5)
        assertThat(receipt["truth_state_at_capture"]).isEqualTo("NOT_CREATED")
        assertThat(receipt["provider_state_at_capture"]).isEqualTo("NOT_STARTED")
        assertThat(receipt["recorder_authority"]).isEqualTo("DEVICE_OWNED_CONTINUOUS_VIDEO_RECORDER")
        assertThat(receipt).doesNotContainKey("pa3_inference_authorized")
        val declared = receipt.getValue("capture_body_sha256")
        val body = receipt.filterKeys { it != "capture_body_sha256" }
        assertThat(declared).isEqualTo(CanonicalJson.sha256(body))
        assertThat(declared).isEqualTo("0e5c857e19b053f5bbed7ad103cdc80b4b40eae359cc1c25b76f616402388c71")
    }

    @Test
    fun partialRosterCannotProduceReceipt() {
        val plan = plan()

        val error = runCatching {
            CaptureReceiptBuilder.build(plan, captures(plan).dropLast(1), Instant.parse("2026-08-22T09:10:00Z"))
        }.exceptionOrNull()

        assertThat(error).isInstanceOf(IllegalArgumentException::class.java)
        assertThat(error).hasMessageThat().contains("partial capture roster")
    }

    @Test
    fun reusedPhysicalMediaCannotProduceReceipt() {
        val plan = plan()
        val captures = captures(plan).toMutableList()
        captures[1] = captures[1].copy(mediaSha256 = captures[0].mediaSha256)

        val error = runCatching {
            CaptureReceiptBuilder.build(plan, captures, Instant.parse("2026-08-22T09:10:00Z"))
        }.exceptionOrNull()

        assertThat(error).isInstanceOf(IllegalArgumentException::class.java)
        assertThat(error).hasMessageThat().contains("media was reused")
    }

    @Test
    fun captureMustStartAfterArming() {
        val plan = plan()
        val captures = captures(plan).toMutableList()
        captures[0] = captures[0].copy(captureStartedAt = plan.armedAt.minusSeconds(1))

        val error = runCatching {
            CaptureReceiptBuilder.build(plan, captures, Instant.parse("2026-08-22T09:10:00Z"))
        }.exceptionOrNull()

        assertThat(error).isInstanceOf(IllegalArgumentException::class.java)
    }

    private fun plan(): CapturePlan = CapturePlan(
        bodySha256 = "a".repeat(64),
        goalReceiptBodySha256 = "b".repeat(64),
        armedAt = Instant.parse("2026-08-22T09:00:00Z"),
        episodes = (1..5).map { index ->
            CapturePlanEpisode("capture-%02d".format(index), "帮我找入口", "capture-%02d.mp4".format(index))
        },
        originalJson = "{}\n",
    )

    private fun captures(plan: CapturePlan): List<CompletedCapture> = plan.episodes.mapIndexed { index, episode ->
        val started = Instant.parse("2026-08-22T09:01:00Z").plusSeconds(index * 20L)
        CompletedCapture(
            episodeId = episode.episodeId,
            captureStartedAt = started,
            captureCompletedAt = started.plusSeconds(10),
            mediaPath = episode.mediaRelativePath,
            mediaSha256 = "%064x".format(index + 1),
            width = 1280,
            height = 720,
            durationSeconds = 10.0,
        )
    }
}
