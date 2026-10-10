@echo off
title Pipeline Prediksi Arus Selat Bali

echo -------------------------------------------------
echo [1/4] Pindah ke direktori kerja...
echo -------------------------------------------------
cd /d "D:\Haidar\Upgrade Arus"

echo.
echo -------------------------------------------------
echo [2/4] Mengecek status Git...
echo -------------------------------------------------

REM git status --short

echo.
echo -------------------------------------------------
echo [3/4] Mengaktifkan environment Conda (umar309)...
echo -------------------------------------------------

call "C:\Users\stame\anaconda3\Scripts\activate.bat" umar309

echo.
echo -------------------------------------------------
echo [4/4] Mengeksekusi Pipeline...
echo -------------------------------------------------

echo.
echo --- Extract Data Arus dan Pasut
python "extractdata.py"

echo.
echo --- Spatial Pre-processing
python "Spatial Pre-processing.py"

echo.
echo --- MEOF Analysis
python "MEOF Analysis.py"

echo.
echo --- Training
python "Training.py"

echo.
echo --- Reconstruction
python "reconstruction.py"

echo.
echo --- Reconstruction
python "reconstruction.py"

REM echo.
REM echo -------------------------------------------------
REM echo Mengunggah Model dan Artefak...
REM echo -------------------------------------------------

REM git add "00 Data/*.nc"
REM git add "01 Artefak/Aset/*.joblib"
REM git add "01 Artefak/Aset/*.h5"
REM git add *.joblib
REM git add *.h5

REM git commit -m "Auto-update: Model LSTM, artefak PCA, dan hasil validasi terbaru"

REM echo.
REM echo --- Mengirim hasil terbaru ke GitHub...
REM git push origin main

REM echo.
REM echo =================================================
REM echo SUCCESS: Pipeline dan upload berhasil.
REM echo =================================================
REM goto END


REM :ERROR
REM echo.
REM echo =================================================
REM echo ERROR: Pipeline Python gagal.
REM echo GitHub TIDAK di-update.
REM echo =================================================
REM goto END


REM :GIT_ERROR
REM echo.
REM echo =================================================
REM echo ERROR: Push ke GitHub gagal.
REM echo Periksa status Git sebelum menjalankan ulang.
REM echo =================================================


:END
echo.
pause