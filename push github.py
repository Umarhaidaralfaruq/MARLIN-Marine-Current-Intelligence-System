import subprocess
import datetime

from config import PATHS

# ==========================================
# KURIR GIT OTOMATIS
# ==========================================
print("Mulai mengirim data terbaru ke Dashboard Awan...")

# Path ke folder tempat repository Git lu berada
repo_dir = PATHS["Dashboard"]

# Bikin pesan commit dinamis sesuai jam update
waktu_sekarang = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
pesan_commit = f"Auto-update prediksi arus dan data validasi: {waktu_sekarang}"

try:
    # 1A. Menambahkan folder Arsip_Prediksi
    subprocess.run(["git", "add", "Arsip_Prediksi/."], cwd=repo_dir, check=True)
    
    # 1B. BARU: Menambahkan file Validasi HF Radar
    # Kalau lu mau nge-push se-foldernya sekalian, ganti jadi "Hasil_Validasi/."
    subprocess.run(["git", "add", "Hasil_Validasi/Validasi_Real_Max_12H.csv"], cwd=repo_dir, check=True)
    
    # 1C. Numpang push app.py (HAPUS BARIS INI SETELAH SUKSES PUSH)
    subprocess.run(["git", "add", "app.py"], cwd=repo_dir, check=True)
    
    # 1D. Numpang push app.py (HAPUS BARIS INI SETELAH SUKSES PUSH)
    subprocess.run(["git", "add", "config.py"], cwd=repo_dir, check=True)
    
    # 1D. Numpang push app.py (HAPUS BARIS INI SETELAH SUKSES PUSH)
    subprocess.run(["git", "add", "extractdata.py"], cwd=repo_dir, check=True)
    
    # 1F. Numpang push app.py (HAPUS BARIS INI SETELAH SUKSES PUSH)
    subprocess.run(["git", "add", "Spatial Pre-processing.py"], cwd=repo_dir, check=True)
    
    # 1G. Numpang push app.py (HAPUS BARIS INI SETELAH SUKSES PUSH)
    subprocess.run(["git", "add", "Data Prep.py"], cwd=repo_dir, check=True)
    
    # 1H. Numpang push app.py (HAPUS BARIS INI SETELAH SUKSES PUSH)
    subprocess.run(["git", "add", "Predict.py"], cwd=repo_dir, check=True)
    
    # 1I. Numpang push app.py (HAPUS BARIS INI SETELAH SUKSES PUSH)
    subprocess.run(["git", "add", "verification.py"], cwd=repo_dir, check=True)
    
    # 1J. Numpang push app.py (HAPUS BARIS INI SETELAH SUKSES PUSH)
    subprocess.run(["git", "add", "feature_engineering.py"], cwd=repo_dir, check=True)
    
    # 2. Membuat catatan/commit
    subprocess.run(["git", "commit", "-m", pesan_commit], cwd=repo_dir, check=True)
    
    # 3. Menerbangkan data ke GitHub
    subprocess.run(["git", "push"], cwd=repo_dir, check=True)
    
    print(f"✅ Data sukses diterbangkan ke Dashboard pada {waktu_sekarang}!")

except subprocess.CalledProcessError as e:
    # Kalau misalnya nggak ada file baru yang diupdate, dia bakal skip dengan aman
    print(f"⚠️ Kurir selesai (Tidak ada perubahan data baru atau terjadi error: {e})")