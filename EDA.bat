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

REM echo --- Analisis Statistik
REM python "02a_eda_statistics.py"

REM echo --- Analisis Korelasi
REM python "02b_eda_correlation.py"

REM echo --- Analisis Korelasi
REM python "02c_confidence_interval.py"

echo --- Analisis Time Series
python "03_time_series_analysis.py"

echo.
echo =================================================
echo 🎉 SEMUA PROSES SELESAI!
echo =================================================
pause