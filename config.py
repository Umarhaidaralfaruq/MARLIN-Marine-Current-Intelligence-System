import os
from pathlib import Path
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# 1. PENGATURAN DIREKTORI (FOLDERS) UTAMA
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "00 Data"
ARTEFAK_DIR = BASE_DIR / "01 Artefak"
DASHBOARD_DIR = BASE_DIR / "Dashboard_Arus"
PREDIKSI_DIR = BASE_DIR / "Arsip_Prediksi"
VALIDASI_DIR = BASE_DIR / "02 Hasil Validasi"
HF_RADAR_DIR = Path(r"D:\10 HF Radar NetCDF4")
LOGBOOK_DIR = Path(r"D:\ProgramKantor\Penyeberangan_Ketapang-Gilimanuk\2_Logbook_HFRadar_Ketapang-Gilimanuk")

for folder in [DATA_DIR, ARTEFAK_DIR, PREDIKSI_DIR, VALIDASI_DIR, PREDIKSI_DIR / "Arsip_Prediksi", PREDIKSI_DIR / "Plot Spasial"]:
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# =============================================================================
# 2. PENGATURAN PATH FILE SPESIFIK
# =============================================================================
PATHS = {
    "Dashboard": DASHBOARD_DIR,
    
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
    
    # Hasil Prediksi
    "OUTPUT_PRED": BASE_DIR / "Arsip_Prediksi",
    
    # Verifikasi
    "Verif_12jam" : BASE_DIR / "Hasil_Validasi" / "Validasi_Real_Max_12H.csv",
    "Verif_Spasial": "Hasil_Validasi/Validasi_Spasial_1H.csv",
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
# 4. KOORDINAT AREA (DASHBOARD)
# =============================================================================
# Area Utama
AREA_LAT_MIN, AREA_LAT_MAX = -8.209161, -8.160084
AREA_LON_MIN, AREA_LON_MAX = 114.387089, 114.435215

# Pelabuhan Ketapang
KET_LAT_MIN, KET_LAT_MAX = -8.209161, -8.151693
KET_LON_MIN, KET_LON_MAX = 114.38673, 114.414975

# Pelabuhan Gilimanuk
GIL_LAT_MIN, GIL_LAT_MAX = -8.21003, -8.16198
GIL_LON_MIN, GIL_LON_MAX = 114.41354, 114.43416

# =============================================================================
# 5. HYPERPARAMETER MODEL
# =============================================================================
MODEL_CONFIG = {
    'train_split_ratio': 0.8,
    'n_modes' : 2,
    'input_window': 24,
    'output_window': 24, 
    'min_seq_len': 48,    
    'min_segmen_len': 24, 
    'max_save_nan': 5,     
    'ocean_threshold': 0.2,
    'interp_method': 'polynomial', # Opsi: 'polynomial', 'linear'
    'interp_order': 3,
    
    'lstm_units': 256,
    'dropout_rate': 0.2,
    'dense_units': 256,
    
    'learning_rate': 0.001,
    'loss_function': 'mse',          
    'epochs': 200,
    'batch_size': 32,
    
    'early_stop_patience': 15,
    'reduce_lr_patience': 4,
    'reduce_lr_factor': 0.5,
    'reduce_lr_min': 1e-5,
    'spatial_mode': 'MEOF', # Opsi: 'EOF', 'MEOF'
}

# =============================================================================
# 6. PANEL KONTROL EKSPERIMEN FITUR (PREPROCESSING)
# =============================================================================
FEATURE_TOGGLES = {
    'ENABLE_TIME_FEATURES': False,   
    'ENABLE_TARGET_FEATURES': True,  
    'ENABLE_WATER_LEVEL': False,     
    'ENABLE_WIND': False,            
}

# =============================================================================
# 7. PANEL KONTROL SPATIAL PREPROCESSING
# =============================================================================
SPATIAL_CONFIG = {
    'RESAMPLE_METHOD': 'max_amplitude',  # Opsi: 'max_amplitude', 'mean'
    'RESAMPLE_FREQ': '1H',
    'min_spatial_percent_from_avg': 0.5,     # Batas minimal 50% dari rata-rata titik valid
    'min_temporal_records': 3,
    'max_physical_speed': 2.59,
}

# =============================================================================
# 8. PARAMETER KALKULASI & SISTEM (DASHBOARD)
# =============================================================================
FAKTOR_KNOT = 1.94384
METODE_STATISTIK = "P95"  # Opsi: "MAX" atau "P95"
REFRESH_INTERVAL_MS = 1800000

# =============================================================================
# 9. KONFIGURASI VISUALISASI & EVALUASI SPASIAL
# =============================================================================
VISUAL_CONFIG = {
    'plot_target_hour_idx': 5, 
    
    'max_val_mae_speed': 1.0,      
    'max_val_rmse_speed': 1.0,     
    'max_val_rmse_dir': 90,        
    
    'threshold_mae_speed': 0.3,    
    'threshold_rmse_speed': 0.4,   
    'threshold_rmse_dir': 45,      
}

# =============================================================================
# 10. KONFIGURASI DASHBOARD & PETA (STATUS THRESHOLD)
# =============================================================================
GRID_NX, GRID_NY = 60, 60
BATAS_RESOLUSI = 0.011
JARAK_LABEL_PETA = 0.015

KONFIG_STATUS = {
    "AWAS":    {"batas_ms": 2.5, "icon": "🔴", "bg": "#dc3545", "text_color": "#ffffff"},
    "WASPADA": {"batas_ms": 2.0, "icon": "🟠", "bg": "#fd7e14", "text_color": "#ffffff"},
    "SIAGA":   {"batas_ms": 1.5, "icon": "🟡", "bg": "#ffc107", "text_color": "#000000"},
    "AMAN":    {"batas_ms": 0.0, "icon": "🟢", "bg": "#28a745", "text_color": "#ffffff"} 
}

# =============================================================================
# 11. KONFIGURASI GRAFIK & UI (DASHBOARD)
# =============================================================================
CHART_GROUPING_OPTIONS = [2, 3, 4, 6]  # Pilihan rentang jam di dropdown
CHART_GROUPING_DEFAULT = 3             # Nilai default awal (harus ada di dalam list di atas)

# =============================================================================
# 12. KONFIGURASI VALIDASI OTOMATIS
# =============================================================================
VALIDATION_CONFIG = {
    'window_hours': 3,          # Berapa jam per kelompok timeframe
    'threshold_percent': 0.5    # Persentase minimal kelulusan (0.5 = 50%)
}

# =========================================================
# 13. PENGATURAN GITHUB REPOSITORY UNTUK BACKEND 
# =========================================================
GITHUB_OWNER = "Umarhaidaralfaruq"
GITHUB_REPO = "MARLIN-Marine-Current-Intelligence-System"   # Contoh: "marlin-data"
GITHUB_BRANCH = "main"                                      # Cabang utama (main / master)
GITHUB_DATA_DIR = "Arsip_Prediksi"                  # Folder tempat CSV (isi "" jika di luar/root, atau "data" jika di dalam folder data)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
