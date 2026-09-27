@echo off
title Pipeline Prediksi Arus Selat Bali

echo -------------------------------------------------
echo [1/3] Pindah ke direktori kerja...
echo -------------------------------------------------
cd /d "D:\Haidar\Upgrade Arus"

echo.
echo -------------------------------------------------
echo [2/3] Mengaktifkan environment Conda (umar309)...
echo -------------------------------------------------
:: Memanggil script aktivasi bawaan Anaconda, lalu nama environment-nya
call "C:\Users\stame\anaconda3\Scripts\activate.bat" umar309

echo.
echo -------------------------------------------------
echo [3/3] Mengeksekusi Script Python...
echo -------------------------------------------------

echo --- Extract Data Arus dan Pasut
python "extractdata.py"

echo --- Spatial Pre-processing
python "Spatial Pre-processing.py"

echo --- MEOF Analysis
python "MEOF Analysis.py"

echo --- Training
python "Training.py"

echo --- Reconstruction
python "reconstruction.py"

echo --- Mengunggah Model Baru dan Artefak ke Git lokal...
git add "01 Artefak/Aset/*.joblib"
git add "01 Artefak/Aset/*.h5"
git add *.joblib
git add *.h5

:: Tambahkan baris ini agar skrip ikut tersimpan dan tidak ter-stash!
git add "Prediksi Arus With Training.bat"

:: 1. Commit khusus file model, artefak, dan skrip bat ini
git commit -m "Auto-update: Model LSTM, artefak PCA, dan hasil validasi terbaru" || echo --- Tidak ada perubahan pada model untuk di-commit.

echo --- Mengamankan file lain yang sedang berubah...
:: 2. Sembunyikan sisa file (seperti .py, .nc, .csv) ke penyimpanan sementara
git stash

echo --- Sinkronisasi dengan GitHub...
:: 3. Rebase sekarang akan berjalan mulus tanpa error
git pull --rebase origin main

echo --- Mengembalikan file yang diamankan...
:: 4. Keluarkan lagi file sisa tadi agar foldermu kembali seperti semula
git stash pop || echo --- Tidak ada file di penyimpanan sementara.

echo --- Mengirim ke GitHub...
:: 5. Push model terbaru ke server
git push origin main

echo --- Selesai! Model dan artefak siap digunakan oleh Robot GitHub!

echo.
echo =================================================
echo 🎉 SEMUA PROSES SELESAI!
echo =================================================
pause