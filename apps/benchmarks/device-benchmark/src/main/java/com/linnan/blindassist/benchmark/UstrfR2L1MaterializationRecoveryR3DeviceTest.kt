package com.linnan.blindassist.benchmark

import android.os.Process
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.InputStream
import java.security.MessageDigest

/**
 * Outcome-unseen R3 transport canary.
 *
 * It verifies that a manifest and every bound image are readable from the target
 * app's internal files directory. It never loads TFLite or candidate code.
 */
@RunWith(AndroidJUnit4::class)
class UstrfR2L1MaterializationRecoveryR3DeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun verifyTargetPrivateTransportCanary() {
        if (arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() != true) {
            assumeTrue("R3 target-private transport canary was not requested", false)
            return
        }
        val outputFile = privateOutputFile(requireArgument(ARG_OUTPUT))
        val manifestRelative = requireArgument(ARG_INPUT)
        val expectedManifestSha = requireArgument(ARG_EXPECTED_MANIFEST_SHA)
        var status = "TRANSPORT_CANARY_FAILED"
        var error: String? = null
        var manifestSha: String? = null
        var manifestBytes: Long? = null
        var verifiedImages = 0
        var verifiedImageBytes = 0L
        var sourceId: String? = null
        var sequenceId: String? = null
        try {
            val manifestFile = privateFile(manifestRelative, "R3 manifest")
            manifestSha = sha256File(manifestFile)
            manifestBytes = manifestFile.length()
            check(manifestSha == expectedManifestSha) {
                "manifest_sha256_mismatch"
            }
            val manifest = JSONObject(manifestFile.readText(Charsets.UTF_8))
            check(
                manifest.getString("schema") ==
                    "blindassist_ustrf_route_target_l1e_device_shard_manifest_r1"
            )
            check(manifest.getString("stage") == "R2-L1E-RECOVERY-B1")
            check(manifest.getString("attempt_namespace") == "r2-l1e-recovery-b1")
            sourceId = manifest.getString("source_id")
            sequenceId = manifest.getString("sequence_id")
            val frames = manifest.getJSONArray("frames")
            check(frames.length() == manifest.getInt("frame_count"))
            check(frames.length() > 0)
            for (index in 0 until frames.length()) {
                val expected = frames.getJSONObject(index)
                val imageFile = privateChild(
                    manifestFile.parentFile,
                    expected.getString("image_path"),
                    "R3 masked frame"
                )
                check(sha256File(imageFile) == expected.getString("image_sha256")) {
                    "image_sha256_mismatch:${expected.getString("frame_id")}"
                }
                verifiedImages++
                verifiedImageBytes += imageFile.length()
            }
            status = "TARGET_PRIVATE_TRANSPORT_CANARY_PASS"
        } catch (failure: Throwable) {
            error = failure.javaClass.simpleName + ":" + (failure.message ?: "unknown")
        } finally {
            val receipt = JSONObject()
                .put(
                    "schema",
                    "blindassist_ustrf_route_target_l1e_materialization_recovery_r3_transport_canary"
                )
                .put("stage", "R2-L1E-RECOVERY-B1")
                .put("attempt_namespace", "r2-l1e-recovery-b1")
                .put("status", status)
                .put("error", error)
                .put("target_package", targetContext.packageName)
                .put("instrumentation_process_uid", Process.myUid())
                .put("resolved_storage", "target_context_internal_files")
                .put("manifest_relative_path", manifestRelative)
                .put("manifest_sha256", manifestSha)
                .put("manifest_bytes", manifestBytes)
                .put("source_id", sourceId)
                .put("sequence_id", sequenceId)
                .put("verified_image_count", verifiedImages)
                .put("verified_image_bytes", verifiedImageBytes)
                .put(
                    "candidate_execution",
                    JSONObject()
                        .put("tflite_loaded", false)
                        .put("c1_c2_c3_loaded", false)
                        .put("candidate_output_count", 0)
                )
                .put(
                    "authority",
                    JSONObject()
                        .put("input_transport_only", true)
                        .put("selection", false)
                        .put("android_shadow", false)
                        .put("h2", false)
                        .put("human_outcome", false)
                        .put("production", false)
                )
            outputFile.writeText(receipt.toString(), Charsets.UTF_8)
        }
        check(status == "TARGET_PRIVATE_TRANSPORT_CANARY_PASS") {
            error ?: "R3 target-private transport canary failed"
        }
    }

    private fun privateFile(relative: String, label: String): File {
        val root = targetContext.filesDir.canonicalFile
        val file = File(root, relative).canonicalFile
        check(file.toPath().startsWith(root.toPath())) { "$label escapes target private files" }
        check(file.isFile) { "$label is missing: $relative" }
        return file
    }

    private fun privateChild(parent: File?, relative: String, label: String): File {
        check(parent != null) { "$label parent is missing" }
        val root = targetContext.filesDir.canonicalFile
        val file = File(parent, relative).canonicalFile
        check(file.toPath().startsWith(root.toPath())) { "$label escapes target private files" }
        check(file.isFile) { "$label is missing: $relative" }
        return file
    }

    private fun privateOutputFile(relative: String): File {
        val root = targetContext.filesDir.canonicalFile
        val file = File(root, relative).canonicalFile
        check(file.toPath().startsWith(root.toPath())) { "output escapes target private files" }
        file.parentFile?.mkdirs()
        return file
    }

    private fun requireArgument(name: String): String =
        requireNotNull(arguments.getString(name)) { "missing instrumentation argument: $name" }
            .also {
                check(it.isNotBlank() && !it.startsWith('/') && !it.contains("..")) {
                    "unsafe instrumentation argument: $name"
                }
            }

    private fun sha256File(file: File) = file.inputStream().use(::digest)

    private fun digest(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val ARG_REQUIRED = "ustrfR2L1eRecoveryR3CanaryRequired"
        const val ARG_INPUT = "ustrfR2L1eRecoveryR3CanaryInput"
        const val ARG_EXPECTED_MANIFEST_SHA = "ustrfR2L1eRecoveryR3ExpectedManifestSha256"
        const val ARG_OUTPUT = "ustrfR2L1eRecoveryR3CanaryOutput"
    }
}
