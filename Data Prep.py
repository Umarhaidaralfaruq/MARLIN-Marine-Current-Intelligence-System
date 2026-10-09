# =============================================================================
# FILE: 01_data_preparation.py (MODULAR EOF / MEOF)
# DESKRIPSI: Membersihkan Data, Ekstrak EOF/MEOF (Semua Data), dan Segmentasi
# =============================================================================

import xarray as xr
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.decomposition import PCA
from config import PATHS, MODEL_CONFIG, FEATURE_TOGGLES

N_MODES = MODEL_CONFIG['n_modes']
MAX_SAFE_NAN = MODEL_CONFIG['max_save_nan']
SPATIAL_MODE = MODEL_CONFIG.get('spatial_mode', 'EOF').upper()

def apply_segmentation_datetime(df):
    df = df.sort_values('datetime').reset_index(drop=True)
    df['dt_hour'] = df['datetime'].diff().dt.total_seconds() / 3600
    df['break_seg'] = df['dt_hour'] > MAX_SAFE_NAN
    df['Segment_ID'] = df['break_seg'].cumsum() + 1
    
    seg_len = df.groupby('Segment_ID').size()
    valid_seg = seg_len[seg_len >= MODEL_CONFIG['min_seq_len']].index
    
    df_final = df[df['Segment_ID'].isin(valid_seg)].copy()
    df_final['Segment_ID'] = df_final.groupby('Segment_ID').ngroup() + 1
    df_final = df_final.drop(columns=['dt_hour', 'break_seg'])
    return df_final

def prepare_operational_data():
    print("\n" + "="*60)
    print(f"🧹 FASE 1: PENYIAPAN MASTER DATA ({SPATIAL_MODE})")
    print("="*60)

    file_input = PATHS['dataset_arus']
    if not os.path.exists(file_input): 
        print(f"❌ File {file_input} tidak ditemukan!")
        return

    df_pasut = None
    if FEATURE_TOGGLES.get('ENABLE_WATER_LEVEL', False):
        try:
            print("🌊 Membaca dan resampling data Elevasi Pasut...")
            df_pasut = pd.read_excel(PATHS['aws_database'])
            df_pasut.columns = df_pasut.columns.str.strip()
            df_pasut['datetime'] = pd.to_datetime(df_pasut['datetime'], format='%d/%m/%Y %H:%M:%S')
            df_pasut = df_pasut.set_index('datetime').resample('1H').mean()
            df_pasut = df_pasut.interpolate(method='time').reset_index()
            df_pasut = df_pasut.drop_duplicates(subset=['datetime'])
        except Exception as e:
            print(f"❌ Error memuat data pasut: {e}")

    print("🥞 Memuat data NetCDF dan meratakan matriks spasial (Stacking)...")
    ds = xr.open_dataset(file_input)
    time_arr = ds.time.values
    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    valid_grids_mask = ds_mean_u['u'].notnull().values.flatten()
    stacked = ds.stack(grid=['lat', 'lon'])
    ocean_only = stacked.isel(grid=valid_grids_mask)

    df_master = None

    if SPATIAL_MODE == 'EOF':
        components_to_run = ['u', 'v']
        for comp in components_to_run:
            print(f"\n🌪️ EKSTRAKSI EOF UNTUK: {comp.upper()}")
            comp_matrix = ocean_only[comp].values
            
            is_radar_off = (comp_matrix == 0.0).all(axis=1)
            X_clean, time_clean = comp_matrix[~is_radar_off], time_arr[~is_radar_off]
            print(f"   -> Membuang {is_radar_off.sum()} jam data saat radar mati total.")
            
            pca = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', f'_{comp}_model.joblib'))
            
            expected_features = pca.n_features_in_
            if X_clean.shape[1] > expected_features:
                X_clean = X_clean[:, :expected_features]
            elif X_clean.shape[1] < expected_features:
                padding = np.zeros((X_clean.shape[0], expected_features - X_clean.shape[1]))
                X_clean = np.hstack((X_clean, padding))
            
            pc_all = pca.transform(X_clean)
            
            col_names = [f"{comp.upper()}_PC{i+1}" for i in range(N_MODES)]
            df_comp = pd.DataFrame(pc_all, columns=col_names)
            df_comp.insert(0, 'datetime', pd.to_datetime(time_clean))
            
            if df_master is None:
                df_master = df_comp
            else:
                df_master = pd.merge(df_master, df_comp, on='datetime', how='inner')

    elif SPATIAL_MODE == 'MEOF':
        print(f"\n🌪️ EKSTRAKSI MEOF (GABUNGAN U & V)")
        matrix_u = ocean_only['u'].values
        matrix_v = ocean_only['v'].values
        matrix_gabungan = np.hstack((matrix_u, matrix_v))
        
        is_radar_off = (matrix_gabungan == 0.0).all(axis=1)
        X_clean, time_clean = matrix_gabungan[~is_radar_off], time_arr[~is_radar_off]
        print(f"   -> Membuang {is_radar_off.sum()} jam data saat radar mati total.")
        
        pca = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_meof_model.joblib'))
        
        expected_features = pca.n_features_in_
        if X_clean.shape[1] > expected_features:
            X_clean = X_clean[:, :expected_features]
        elif X_clean.shape[1] < expected_features:
            padding = np.zeros((X_clean.shape[0], expected_features - X_clean.shape[1]))
            X_clean = np.hstack((X_clean, padding))
            
        pc_all = pca.transform(X_clean)
        
        col_names = [f"MEOF_PC{i+1}" for i in range(N_MODES)]
        df_master = pd.DataFrame(pc_all, columns=col_names)
        df_master.insert(0, 'datetime', pd.to_datetime(time_clean))

    if df_pasut is not None:
        df_master = pd.merge(df_master, df_pasut, on='datetime', how='inner')

    print("🏷️ Melabeli Segmen menggunakan logika Datetime Gap...")
    df_final = apply_segmentation_datetime(df_master)

    output_path = PATHS['data_gabung']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    ds.close()
    print(f"\n✅ FASE 1 SELESAI! File Master tersimpan di: {output_path}")

if __name__ == "__main__":
    prepare_operational_data()