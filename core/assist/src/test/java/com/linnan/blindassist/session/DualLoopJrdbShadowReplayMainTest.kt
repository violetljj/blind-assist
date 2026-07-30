package com.linnan.blindassist.session

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.security.MessageDigest
import org.junit.Assert.assertThrows
import org.junit.Test

class DualLoopJrdbShadowReplayMainTest {
    @Test
    fun producerReceiptMustBindExactReplayBytes() {
        val directory = Files.createTempDirectory("dual-loop-jrdb-receipt-test")
        val input = directory.resolve("replay.tsv")
        val receipt = directory.resolve("producer_receipt.json")
        Files.writeString(input, "header\nrow\n", StandardCharsets.UTF_8)
        Files.writeString(
            receipt,
            validProducerReceipt(outputSha256 = sha256(input)),
            StandardCharsets.UTF_8
        )

        DualLoopJrdbShadowReplayMain.validateProducerReceipt(receipt, input)

        Files.writeString(
            receipt,
            validProducerReceipt(outputSha256 = "0".repeat(64)),
            StandardCharsets.UTF_8
        )
        assertThrows(IllegalArgumentException::class.java) {
            DualLoopJrdbShadowReplayMain.validateProducerReceipt(receipt, input)
        }
    }

    @Test
    fun validlyHashedUndersizedReplayCannotReachKernel() {
        val directory = Files.createTempDirectory("dual-loop-jrdb-denominator-test")
        val input = directory.resolve("replay.tsv")
        val receipt = directory.resolve("producer_receipt.json")
        val output = directory.resolve("kernel_receipt.json")
        Files.writeString(
            input,
            """
                sequence	frame_index	image_timestamp_ns	available_at_ns	frame_width	frame_height	label_id	left	top	right	bottom	box_clamped	geometry_status	geometry_reason	previous_frame_index	previous_timestamp_ns	signed_approach_rate_per_s	quality
                clark-center-2019-02-28_0	1	1000000	1100000	3760	480	pedestrian:1	10	10	20	20	false	ELIGIBLE	OK	0	0	0.1	1.0
            """.trimIndent() + "\n",
            StandardCharsets.UTF_8
        )
        Files.writeString(
            receipt,
            validProducerReceipt(outputSha256 = sha256(input)),
            StandardCharsets.UTF_8
        )

        assertThrows(IllegalArgumentException::class.java) {
            DualLoopJrdbShadowReplayMain.main(
                arrayOf(input.toString(), receipt.toString(), output.toString())
            )
        }
        check(!Files.exists(output))
    }

    private fun validProducerReceipt(outputSha256: String): String = """
        {
          "schema": "blindassist.dual_loop_jrdb_shadow_producer_receipt.v1",
          "status": "COMPLETE",
          "source_id": "JRDB_ANNOTATION_CONDITIONED_LIDAR_CENTROID_REPLAY_V1",
          "claim_ceiling": "DIAGNOSTIC_ENGINEERING_ONLY",
          "output_sha256": "$outputSha256",
          "implementation_sha256": "dd72ece0e910363507c60b873abc82726ae0e825353a31c51e9da1f7ae3ef4aa",
          "sequence_count": 4,
          "frame_count": 480,
          "detection_rows": 10786,
          "source_eligible_rows": 8836,
          "protected_alert_outcomes_opened": false
        }
    """.trimIndent() + "\n"

    private fun sha256(path: java.nio.file.Path): String {
        val digest = MessageDigest.getInstance("SHA-256")
        Files.newInputStream(path).use { stream ->
            val buffer = ByteArray(1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }
}
