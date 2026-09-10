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

REM echo =^> cek_nc
REM python "cek_nc.py"

REM echo =^> Mengunggah Model Baru ke GitHub...
REM git add "01 Artefak/Aset/*.joblib"
REM git add "01 Artefak/Aset/*.h5"
REM git add *.joblib
REM git add *.h5
REM git commit -m "Update model LSTM dan PCA terbaru hasil eksperimen lokal"
REM git push origin main
REM echo =^> Selesai! Model siap digunakan oleh Robot GitHub besok pagi!

echo.
echo =================================================
echo 🎉 SEMUA PROSES SELESAI!
echo =================================================
pause