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

git status --short

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
if errorlevel 1 goto ERROR

echo.
echo --- Spatial Pre-processing
python "Spatial Pre-processing.py"
if errorlevel 1 goto ERROR

echo.
echo --- MEOF Analysis
python "MEOF Analysis.py"
if errorlevel 1 goto ERROR

echo.
echo --- Training
python "Training.py"
if errorlevel 1 goto ERROR

echo.
echo --- Reconstruction
python "reconstruction.py"
if errorlevel 1 goto ERROR

echo.
echo -------------------------------------------------
echo Mengunggah Model dan Artefak...
echo -------------------------------------------------

git add "00 Data/*.nc"
git add "01 Artefak/Aset/*.joblib"
git add "01 Artefak/Aset/*.h5"
git add *.joblib
git add *.h5

git commit -m "Auto-update: Model LSTM, artefak PCA, dan hasil validasi terbaru"

if errorlevel 1 (
    echo.
    echo --- Tidak ada perubahan output untuk di-commit.
)

echo.
echo --- Mengirim hasil terbaru ke GitHub...
git push origin main

if errorlevel 1 goto GIT_ERROR

echo.
echo =================================================
echo SUCCESS: Pipeline dan upload berhasil.
echo =================================================
goto END


:ERROR
echo.
echo =================================================
echo ERROR: Pipeline Python gagal.
echo GitHub TIDAK di-update.
echo =================================================
goto END


:GIT_ERROR
echo.
echo =================================================
echo ERROR: Push ke GitHub gagal.
echo Periksa status Git sebelum menjalankan ulang.
echo =================================================


:END
echo.
pause