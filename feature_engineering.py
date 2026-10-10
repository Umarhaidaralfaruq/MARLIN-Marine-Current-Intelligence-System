# =============================================================================
# FILE: 03_feature_engineering.py
# DESKRIPSI: Modul Rakitan Fitur Dinamis (Dipanggil saat Training & Predict)
# =============================================================================

import pandas as pd
import numpy as np
from config import MODEL_CONFIG, FEATURE_TOGGLES

WIND_COLS = ['wind_u', 'wind_v'] 

def _add_time_features(df):
    features = []
    dt_series = df.index if isinstance(df.index, pd.DatetimeIndex) else df['datetime'].dt
    df['hour_sin'] = np.sin(2 * np.pi * dt_series.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt_series.hour / 24)
    df['tide_sin'] = np.sin(2 * np.pi * dt_series.hour / 12.42)
    df['tide_cos'] = np.cos(2 * np.pi * dt_series.hour / 12.42)
    features += ['hour_sin', 'hour_cos', 'tide_sin', 'tide_cos']
    return df, features

def _add_target_features(df, target_columns):
    features = []
    for col_name in target_columns:
        if col_name not in df.columns: 
            continue
        win = 24
        col_mean = f'rolling_mean_{col_name}_{win}'
        df[col_mean] = df.groupby('Segment_ID')[col_name].transform(
            lambda x: x.rolling(window=win, min_periods=1).mean()
        )
        features.append(col_mean)

        col_dist = f'jarak_dari_latar_{col_name}_{win}'
        df[col_dist] = df[col_name] - df[col_mean]
        features.append(col_dist)
    return df, features

def _add_water_level_features(df):
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
    features = []
    for w_col in WIND_COLS:
        if w_col in df.columns:
            features.append(w_col)
    return df, features

def build_features(df):
    print("⚙️ Mengeksekusi Feature Engineering...")
    df_processed = df.copy()
    all_feats = []

    # 🚀 DETEKSI DINAMIS KOLOM TARGET (Mendukung EOF & MEOF)
    target_cols = [c for c in df_processed.columns if 'PC' in c and not c.startswith('rolling_') and not c.startswith('jarak_')]

    if FEATURE_TOGGLES['ENABLE_TIME_FEATURES']:
        df_processed, new_feats = _add_time_features(df_processed)
        all_feats.extend(new_feats)
        print(f"   [+] Fitur Waktu ditambahkan ({len(new_feats)} fitur)")

    if FEATURE_TOGGLES['ENABLE_TARGET_FEATURES']:
        df_processed, new_feats = _add_target_features(df_processed, target_cols)
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

    mandatory_cols = ['datetime', 'Segment_ID'] + target_cols
    
    final_list = sorted(list(set(all_feats)))
    final_list = mandatory_cols + [col for col in final_list if col not in mandatory_cols] 

    final_list = [col for col in final_list if col in df_processed.columns]
    
    df_final = df_processed[final_list].dropna().reset_index(drop=True)
    
    print(f"✅ Feature Engineering selesai. Total Kolom: {df_final.shape[1]}")
    return df_final