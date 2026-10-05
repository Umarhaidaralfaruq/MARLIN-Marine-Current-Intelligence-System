# =============================================================================
# FILE: 03_feature_engineering.py
# DESKRIPSI: Modul Rakitan Fitur Dinamis (Dipanggil saat Training & Predict)
# =============================================================================

import pandas as pd
import numpy as np

from config import MODEL_CONFIG, FEATURE_TOGGLES

# Sesuaikan dengan nama kolom target dan angin
TARGET_COLS = [f'PC{i+1}' for i in range(MODEL_CONFIG['n_modes'])]
WIND_COLS = ['wind_u', 'wind_v'] 

# =============================================================================
# FUNGSI PEMBUAT FITUR
# =============================================================================

def _add_time_features(df):
    """Menambahkan fitur siklus waktu (Harmonik Jam dan Pasut)"""
    features = []
    
    # Pastikan datetime jadi acuan (kalau dia masih kolom biasa)
    dt_series = df.index if isinstance(df.index, pd.DatetimeIndex) else df['datetime'].dt
    
    df['hour_sin'] = np.sin(2 * np.pi * dt_series.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt_series.hour / 24)
    
    # Siklus Harmonik Pasut Semi-Diurnal (12.42 jam)
    df['tide_sin'] = np.sin(2 * np.pi * dt_series.hour / 12.42)
    df['tide_cos'] = np.cos(2 * np.pi * dt_series.hour / 12.42)
    
    features += ['hour_sin', 'hour_cos', 'tide_sin', 'tide_cos']
    return df, features

def _add_target_features(df, target_columns):
    """
    Percobaan 2: Feature Engineering Optimal untuk LSTM
    Tanpa Lag Manual, Tanpa Shift, Tanpa Z-Score.
    """
    features = []
    
    for col_name in target_columns:
        if col_name not in df.columns: 
            continue
            
        # 1. ROLLING MEAN 24 JAM (Tanpa .shift!)
        # LSTM berhak melihat rata-rata historis hingga tepat pada jam ini (t)
        # karena indeks pemotongan Lookback di Training.py sudah mengunci masa depan.
        win = 24
        col_mean = f'rolling_mean_{col_name}_{win}'
        df[col_mean] = df.groupby('Segment_ID')[col_name].transform(
            lambda x: x.rolling(window=win, min_periods=1).mean()
        )
        features.append(col_mean)

        # 2. JARAK DARI LATAR (Sinyal Anomali Utama)
        # Ini adalah fitur dengan korelasi +0.9 yang kita pertahankan
        col_dist = f'jarak_dari_latar_{col_name}_{win}'
        df[col_dist] = df[col_name] - df[col_mean]
        features.append(col_dist)

    return df, features

def _add_water_level_features(df):
    """Menambahkan Lags dan Rolling Stats khusus untuk elevasi Pasut"""
    if 'waterlevel' not in df.columns: 
        return df, []
        
    features = []
    
    for lag in [12, 13]:
        col = f'lag_pasut_{lag}'
        df[col] = df.groupby('Segment_ID')['waterlevel'].shift(lag)
        features.append(col)

    for win in [14]:
        col = f'rolling_mean_waterlevel_{win}'
        df[col] = df.groupby('Segment_ID')['waterlevel'].transform(
            lambda x: x.shift(1).rolling(window=win, min_periods=1).mean()
        )
        features.append(col)

    for win in [6, 12, 24]:
        col = f'rolling_std_waterlevel_{win}'
        df[col] = df.groupby('Segment_ID')['waterlevel'].transform(
            lambda x: x.shift(1).rolling(window=win, min_periods=1).std().fillna(0)
        )
        features.append(col)
        
    return df, features

def _add_wind_features(df):
    """Menambahkan fitur angin jika diaktifkan di panel"""
    features = []
    for w_col in WIND_COLS:
        if w_col in df.columns:
            features.append(w_col)
            # Bisa ditambahkan lag/rolling angin di sini jika perlu
    return df, features

# =============================================================================
# FUNGSI UTAMA (MAIN WRAPPER)
# =============================================================================

def build_features(df):
    """
    Mengeksekusi pembuatan fitur berdasarkan Panel Kendali (FEATURE_TOGGLES).
    Mengembalikan DataFrame yang HANYA berisi fitur-fitur valid, ditambah
    kolom wajib seperti datetime, Segment_ID, dan Target (PC).
    """
    print("⚙️ Mengeksekusi Feature Engineering...")
    df_processed = df.copy()
    all_feats = []

    # Eksekusi berdasarkan Toggle
    if FEATURE_TOGGLES['ENABLE_TIME_FEATURES']:
        df_processed, new_feats = _add_time_features(df_processed)
        all_feats.extend(new_feats)
        print(f"   [+] Fitur Waktu ditambahkan ({len(new_feats)} fitur)")

    if FEATURE_TOGGLES['ENABLE_TARGET_FEATURES']:
        df_processed, new_feats = _add_target_features(df_processed, TARGET_COLS)
        all_feats.extend(new_feats)
        print(f"   [+] Fitur PC Target ditambahkan ({len(new_feats)} fitur)")

    if FEATURE_TOGGLES['ENABLE_WATER_LEVEL']:
        df_processed, new_feats = _add_water_level_features(df_processed)
        all_feats.extend(new_feats)
        print(f"   [+] Fitur Water Level ditambahkan ({len(new_feats)} fitur)")
        
    if FEATURE_TOGGLES['ENABLE_WIND']:
        df_processed, new_feats = _add_wind_features(df_processed)
        all_feats.extend(new_feats)
        print(f"   [+] Fitur Angin ditambahkan ({len(new_feats)} fitur)")

    # Daftar kolom wajib yang harus selalu ada agar pipeline tidak patah
    mandatory_cols = ['datetime', 'Segment_ID'] + TARGET_COLS
    
    # Gabungkan fitur baru dengan kolom wajib
    final_list = sorted(list(set(all_feats)))
    for col in mandatory_cols:
        if col not in final_list:
            final_list.insert(0, col) # Taruh mandatory di depan biar rapi

    # Filter hanya kolom yang benar-benar ada di dataframe
    final_list = [col for col in final_list if col in df_processed.columns]
    
    # Karena kita pakai shift(lag), pasti ada baris awal di tiap segmen yang jadi NaN
    # Kita buang baris NaN tersebut agar tidak meracuni LSTM
    df_final = df_processed[final_list].dropna().reset_index(drop=True)
    
    print(f"✅ Feature Engineering selesai. Total Kolom: {df_final.shape[1]}")
    return df_final