package com.linnan.blindassist.vision

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.util.zip.CRC32
import java.util.zip.ZipFile

/** Materializes the SM8650 HTP skeleton for FastRPC when JNI libs stay inside the APK. */
internal object QnnHtpSkelInstaller {
    fun install(context: Context): File {
        val destinationDir = File(context.codeCacheDir, INSTALL_DIRECTORY).apply {
            check(isDirectory || mkdirs()) { "Unable to create QNN skeleton directory: $this" }
        }
        val destination = File(destinationDir, SKEL_FILE_NAME)
        ZipFile(context.applicationInfo.sourceDir).use { apk ->
            val entry = requireNotNull(apk.getEntry(APK_ENTRY)) {
                "APK is missing $APK_ENTRY"
            }
            require(entry.size in 1..MAX_SKEL_BYTES) {
                "Invalid QNN skeleton size: ${entry.size}"
            }
            if (!matches(destination, entry.size, entry.crc)) {
                val temporary = File(destinationDir, "$SKEL_FILE_NAME.tmp-${android.os.Process.myPid()}")
                temporary.delete()
                try {
                    apk.getInputStream(entry).use { input ->
                        FileOutputStream(temporary).use { output ->
                            input.copyTo(output)
                            output.fd.sync()
                        }
                    }
                    require(matches(temporary, entry.size, entry.crc)) {
                        "Extracted QNN skeleton failed size/CRC verification"
                    }
                    if (destination.exists()) check(destination.delete()) {
                        "Unable to replace stale QNN skeleton: $destination"
                    }
                    check(temporary.renameTo(destination)) {
                        "Unable to install QNN skeleton: $destination"
                    }
                } finally {
                    temporary.delete()
                }
            }
        }
        check(destination.setReadable(true, false)) { "Unable to make QNN skeleton readable" }
        check(destination.setExecutable(true, false)) { "Unable to make QNN skeleton executable" }
        return destinationDir
    }

    private fun matches(file: File, expectedSize: Long, expectedCrc: Long): Boolean {
        if (!file.isFile || file.length() != expectedSize) return false
        val crc = CRC32()
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(CRC_BUFFER_BYTES)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                crc.update(buffer, 0, read)
            }
        }
        return crc.value == expectedCrc
    }

    private const val INSTALL_DIRECTORY = "qnn-2.47-sm8650-v75-skel"
    private const val SKEL_FILE_NAME = "libQnnHtpV75Skel.so"
    private const val APK_ENTRY = "lib/arm64-v8a/$SKEL_FILE_NAME"
    private const val MAX_SKEL_BYTES = 64L * 1024L * 1024L
    private const val CRC_BUFFER_BYTES = 64 * 1024
}
