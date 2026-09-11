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

echo =^> Extract Data Arus dan Pasut
python "extractdata.py"

echo =^> Spatial Pre-processing
python "Spatial Pre-processing.py"

echo =^> MEOF Analysis
python "MEOF Analysis.py"

echo =^> Training
python "Training.py"

echo =^> Reconstruction
python "reconstruction.py"

echo =^> cek_nc
python "cek_nc.py"

echo => Sinkronisasi dengan GitHub terlebih dahulu...
git pull --rebase origin main

echo => Mengunggah Model Baru dan Artefak ke GitHub...
git add "01 Artefak/Aset/*.joblib"
git add "01 Artefak/Aset/*.h5"
git add *.joblib
git add *.h5

:: 3. Buat commit (|| echo memastikan skrip lanjut meski tidak ada perubahan)
git commit -m "Auto-update: Model LSTM, artefak PCA, dan hasil validasi terbaru" || echo => Tidak ada perubahan pada model untuk di-commit.

:: 4. Push ke GitHub
git push origin main

echo => Selesai! Model dan artefak siap digunakan oleh Robot GitHub!

echo.
echo =================================================
echo 🎉 SEMUA PROSES SELESAI!
echo =================================================
pause