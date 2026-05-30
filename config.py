import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# 1. PENGATURAN DIREKTORI (FOLDERS) UTAMA
# =============================================================================
# Gunakan Pathlib agar aman digunakan di Windows maupun Mac/Linux
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "00 Data"
ARTEFAK_DIR = BASE_DIR / "01 Artefak"
PREDIKSI_DIR = Path(r"D:\Haidar\Dashboard_Arus")
VALIDASI_DIR = BASE_DIR / "02 Hasil Validasi"
HF_RADAR_DIR = Path(r"D:\10 HF Radar NetCDF4")
LOGBOOK_DIR = Path(r"D:\ProgramKantor\Penyeberangan_Ketapang-Gilimanuk\2_Logbook_HFRadar_Ketapang-Gilimanuk")

# Buat folder jika belum ada (Auto-create saat config di-load)
for folder in [DATA_DIR, ARTEFAK_DIR, PREDIKSI_DIR, VALIDASI_DIR, PREDIKSI_DIR / "Arsip_Prediksi", PREDIKSI_DIR / "Plot Spasial"]:
    folder.mkdir(parents=True, exist_ok=True) # <-- Indentasi sudah diperbaiki di sini

# =============================================================================
# 2. PENGATURAN PATH FILE SPESIFIK
# =============================================================================
PATHS = {
    # Data Mentah
    "hf_radar_pattern": str(HF_RADAR_DIR / "*_Bali_Strait.nc"),
    "nc_file_arus": DATA_DIR / "1_Hasil_HFRadar_SelatBali_Ketapang.nc",
    "aws_database": DATA_DIR / "5_AWS_Database.xlsx",
    
    # Data Hasil Preprocessing
    "spatial_mean_u": DATA_DIR / "2_spatial_mean_u.nc",
    "spatial_mean_v": DATA_DIR / "3_spatial_mean_v.nc",
    "dataset_arus": DATA_DIR / "4_dataset_arus.nc",
    "data_gabung": DATA_DIR / "5_data_gabung.csv",   
    
    # Data Hasil MEOF Analysis
    "pc_train": DATA_DIR / "pc_train.csv",
    "pc_test": DATA_DIR / "pc_test.csv",
    "time_train": DATA_DIR / "time_train.npy",
    "time_test": DATA_DIR / "time_test.npy",
    "meof_pca_model": DATA_DIR / "meof_pca_model.joblib",
    "reconstruction_info": DATA_DIR / "reconstruction_info.pkl",
    
    # Artefak Model LSTM
    "scaler_path": ARTEFAK_DIR / "Aset" / "scaler_ketapang.joblib",
    "model_path": ARTEFAK_DIR / "Aset" / "lstm_model_ketapang.h5",
}

# =============================================================================
# 3. KREDENSIAL & PENGATURAN WEB SCRAPING
# =============================================================================
HFRADAR_URL = os.getenv("HFRADAR_URL")
AWS_STATION_ID = "3000000007"
AWS_LOGIN_URL = os.getenv("AWS_LOGIN_URL")
AWS_DATA_URL = os.getenv("AWS_DATA_URL")
AWS_USERNAME = os.getenv("AWS_USERNAME")          
AWS_PASSWORD = os.getenv("AWS_PASSWORD")          
AWS_START_DEFAULT = datetime(2025, 7, 7)

# =============================================================================
# 5. HYPERPARAMETER MODEL
# =============================================================================
MODEL_CONFIG = {
    # Parameter Dimensi
    'train_split_ratio': 0.8,
    'n_modes' : 2,
    'input_window': 48,   # Membaca data histori 48 jam (2 hari)
    'output_window': 24,  # Memprediksi 24 jam ke depan
    'min_seq_len': 96,    # Syarat panjang segmen minimal sebelum dipotong window
    'min_segmen_len': 48, # Syarat panjang minimal pembuatan segmen
    'max_save_nan': 3,     # Syarat panjang maksimal untuk jam kosong
    'ocean_threshold': 0.5,
    
    # Parameter Arsitektur Neural Network
    'lstm_units': 256,
    'dropout_rate': 0.2,
    'dense_units': 256,
    
    # [BARU] Parameter Training (Kompilasi & Fit)
    'learning_rate': 0.001,
    'loss_function': 'mse',          # Typo 'mse' kita perbaiki di sini!
    'epochs': 200,
    'batch_size': 32,
    
    # [BARU] Parameter Callbacks
    'early_stop_patience': 15,
    'reduce_lr_patience': 4,
    'reduce_lr_factor': 0.5,
    'reduce_lr_min': 1e-5
}

# =============================================================================
# 6. PANEL KONTROL EKSPERIMEN FITUR (PREPROCESSING)
# =============================================================================
FEATURE_TOGGLES = {
    'ENABLE_TIME_FEATURES': False,   # Waktu astronomis (Sin/Cos Jam)
    'ENABLE_TARGET_FEATURES': True,  # Menggunakan masa lalu PC1-PC5 sebagai input
    'ENABLE_WATER_LEVEL': False,      # Data elevasi pasut
    'ENABLE_WIND': False,            # Data angin komponen U & V
}

# =============================================================================
# 7. KONFIGURASI VISUALISASI & EVALUASI SPASIAL
# =============================================================================
VISUAL_CONFIG = {
    # Jam ke-berapa di masa depan yang mau di-plot di quiver plot (0 = jam ke-1)
    # Kita buat default 5 (artinya jam ke-6 masa depan)
    'plot_target_hour_idx': 5, 
    
    # Batas Maksimum Skala Warna di Gambar (Colorbar Maks)
    'max_val_mae_speed': 1.0,      # m/s
    'max_val_rmse_speed': 1.0,     # m/s
    'max_val_rmse_dir': 90,        # derajat
    
    # Garis Batas Toleransi (Dashed Blue Line di Peta)
    'threshold_mae_speed': 0.3,    # m/s
    'threshold_rmse_speed': 0.4,   # m/s
    'threshold_rmse_dir': 45,      # derajat
}