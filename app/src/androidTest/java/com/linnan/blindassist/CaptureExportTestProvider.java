package com.linnan.blindassist;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.os.SystemClock;
import android.provider.DocumentsContract;
import android.provider.DocumentsContract.Document;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

/** Test APK process has Android/JDK classes, but does not inherit the target's Kotlin runtime. */
public final class CaptureExportTestProvider extends ContentProvider {
    private File file(String id) {
        if (!id.matches("[a-zA-Z0-9_-]+")) throw new IllegalArgumentException("Invalid fixture id");
        File root = new File(getContext().getCacheDir(), "capture-export-provider");
        if (!root.isDirectory() && !root.mkdirs()) throw new IllegalStateException("Cannot create fixture directory");
        return new File(root, id);
    }

    @Override public boolean onCreate() { return true; }

    @Override public Cursor query(Uri uri, String[] projection, String selection,
            String[] selectionArgs, String sortOrder) {
        String id = DocumentsContract.getDocumentId(uri);
        File target = file(id);
        String[] columns = projection != null ? projection : new String[] {
            Document.COLUMN_DOCUMENT_ID, Document.COLUMN_DISPLAY_NAME,
            Document.COLUMN_MIME_TYPE, Document.COLUMN_FLAGS, Document.COLUMN_SIZE
        };
        MatrixCursor cursor = new MatrixCursor(columns);
        if (target.exists()) {
            Object[] row = new Object[columns.length];
            for (int i = 0; i < columns.length; i++) {
                switch (columns[i]) {
                    case Document.COLUMN_DOCUMENT_ID:
                    case Document.COLUMN_DISPLAY_NAME: row[i] = id; break;
                    case Document.COLUMN_MIME_TYPE: row[i] = "application/zip"; break;
                    case Document.COLUMN_FLAGS:
                        row[i] = Document.FLAG_SUPPORTS_WRITE | Document.FLAG_SUPPORTS_DELETE; break;
                    case Document.COLUMN_SIZE: row[i] = target.length(); break;
                    default: row[i] = null;
                }
            }
            cursor.addRow(row);
        }
        return cursor;
    }

    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        String id = DocumentsContract.getDocumentId(uri);
        File target = file(id);
        if (!mode.contains("w")) return ParcelFileDescriptor.open(target, ParcelFileDescriptor.MODE_READ_ONLY);
        ParcelFileDescriptor descriptor = ParcelFileDescriptor.open(target,
            ParcelFileDescriptor.MODE_CREATE | ParcelFileDescriptor.MODE_WRITE_ONLY | ParcelFileDescriptor.MODE_TRUNCATE);
        // Creation is the observable entry marker. Keep the service worker inside this
        // Binder call until the test explicitly creates the matching release file.
        long deadline = SystemClock.elapsedRealtime() + 30_000;
        try {
            while (id.startsWith("gated-") && !file("release-" + id).exists()
                    && SystemClock.elapsedRealtime() < deadline) Thread.sleep(20);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            try { descriptor.close(); } catch (IOException ignored) { }
            throw new FileNotFoundException("Fixture open interrupted");
        }
        return descriptor;
    }

    @Override public int delete(Uri uri, String selection, String[] selectionArgs) {
        File target = file(DocumentsContract.getDocumentId(uri));
        if (target.exists() && !target.delete()) throw new IllegalStateException("Cannot delete fixture");
        return 1;
    }

    @SuppressWarnings("deprecation")
    @Override public Bundle call(String method, String arg, Bundle extras) {
        if (!"android:deleteDocument".equals(method)) throw new UnsupportedOperationException(method);
        Uri uri = extras == null ? null : extras.getParcelable("uri");
        if (uri == null) throw new IllegalArgumentException("Missing document URI");
        delete(uri, null, null);
        return new Bundle();
    }

    @Override public String getType(Uri uri) { return "application/zip"; }
    @Override public Uri insert(Uri uri, ContentValues values) { throw new UnsupportedOperationException(); }
    @Override public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException();
    }
}
