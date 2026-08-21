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

echo =^> Extract Data Arus dan Pasut
python "Spatial Pre-processing.py"

echo =^> Data Preparation
python "Data Prep.py"

echo =^> Prediction
python "Predict.py"

echo =^> verification
python "verification.py"

echo =^> Prediction
python "push github.py"


echo.
echo =================================================
echo 🎉 SEMUA PROSES SELESAI!
echo =================================================
:: Pause sangat penting agar jendela tidak langsung menutup jika ada error!
pause