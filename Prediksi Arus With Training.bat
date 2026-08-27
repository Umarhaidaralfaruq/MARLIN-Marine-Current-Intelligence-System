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

REM echo =^> Extract Data Arus dan Pasut
REM python "extractdata.py"

REM echo =^> Spatial Pre-processing
REM python "Spatial Pre-processing.py"

REM echo =^> MEOF Analysis
REM python "MEOF Analysis.py"

REM echo =^> Training
REM python "Training.py"

REM echo =^> Reconstruction
REM python "reconstruction.py"

echo =^> Mengunggah Model Baru ke GitHub...
git add "01 Artefak/Aset/*.joblib"
git add "01 Artefak/Aset/*.h5"
git add *.joblib
git add *.h5
git commit -m "Update model LSTM dan PCA terbaru hasil eksperimen lokal"
git push origin main
echo =^> Selesai! Model siap digunakan oleh Robot GitHub besok pagi!

echo.
echo =================================================
echo 🎉 SEMUA PROSES SELESAI!
echo =================================================
pause