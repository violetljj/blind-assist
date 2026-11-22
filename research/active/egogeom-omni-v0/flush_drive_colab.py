from google.colab import drive
print("Flushing and unmounting Google Drive...", flush=True)
drive.flush_and_unmount()
print("Drive cleanly flushed and unmounted!", flush=True)
