@echo off
title Pipeline Prediksi Arus Selat Bali (Local)

echo -------------------------------------------------
echo [1/4] Pindah ke direktori kerja ^& Git Pull...
echo -------------------------------------------------
cd /d "D:\Haidar\Upgrade Arus"

echo =^> Menarik data terbaru dari repository...
git pull origin main

echo.
echo -------------------------------------------------
echo [2/4] Mengaktifkan environment Conda (umar309)...
echo -------------------------------------------------
:: Memanggil script aktivasi bawaan Anaconda, lalu nama environment-nya
call "C:\Users\stame\anaconda3\Scripts\activate.bat" umar309

echo.
echo -------------------------------------------------
echo [3/4] Mengeksekusi Script Python...
echo -------------------------------------------------

echo =^> Run 1 - Data Collector (HF Radar)
python extractdata.py

echo =^> Run 2 - Spatial Pre-processing
python "Spatial Pre-processing.py"

echo =^> Run 3 - Data Preparation (EOF)
python "Data Prep.py"

echo =^> Run 4 - Auto Validation
python verification.py

echo =^> Run 5 - Operational Prediction
python Predict.py

echo.
echo -------------------------------------------------
echo [4/4] Auto-Commit ^& Push Hasil Prediksi...
echo -------------------------------------------------
echo =^> Menambahkan file csv ke Git...
git add -f Arsip_Prediksi/*.csv
git add -f Hasil_Validasi/*.csv
:: Menambahkan error handling sederhana jika folder tidak ada
git add -f Dashboard_Arus/Arsip_Prediksi/*.csv 2>nul

echo =^> Melakukan commit...
git commit -m "Auto-update: Hasil prediksi dan validasi terbaru (Local Run) [skip ci]"

echo =^> Mendorong ke GitHub...
git push origin main

echo.
echo =================================================
echo 🎉 SEMUA PROSES SELESAI!
echo =================================================
:: Pause sangat penting agar jendela tidak langsung menutup jika ada error!
:: pause