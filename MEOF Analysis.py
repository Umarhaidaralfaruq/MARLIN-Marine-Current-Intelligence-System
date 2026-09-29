# =============================================================================
# FILE: 02_eof_extract.py (REVISI SEGMENTASI DATETIME)
# DESKRIPSI: Separasi EOF, Filter Radar Off sebelum PCA, dan Segmentasi
# =============================================================================

import xarray as xr
import numpy as np
import pandas as pd
import pickle
import joblib
import os
from sklearn.decomposition import PCA

from config import PATHS, MODEL_CONFIG, FEATURE_TOGGLES

TRAIN_RATIO = MODEL_CONFIG['train_split_ratio']
N_MODES = MODEL_CONFIG['n_modes']
MAX_SAFE_NAN = MODEL_CONFIG['max_save_nan']

def apply_segmentation_datetime(df):
    """
    Logika segmentasi menggunakan lompatan datetime (diff > 3 jam)
    """
    # 1. Pastikan urut waktu
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 2. Hitung selisih waktu dengan baris sebelumnya (dalam jam)
    df['dt_hour'] = df['datetime'].diff().dt.total_seconds() / 3600
    
    # 3. Deteksi lompatan > MAX_SAFE_NAN
    # dt_hour > 3.0 menghasilkan True (1). Di-cumsum agar ID Segmen naik.
    df['break_seg'] = df['dt_hour'] > MAX_SAFE_NAN
    df['Segment_ID'] = df['break_seg'].cumsum() + 1
    
    # 4. FILTER LAPIS 1: Buang Segmen yang panjangnya < 48 Jam
    seg_len = df.groupby('Segment_ID').size()
    valid_seg = seg_len[seg_len >= MODEL_CONFIG['input_window']].index
    
    df_final = df[df['Segment_ID'].isin(valid_seg)].copy()
    
    # 5. Urutkan ulang ID Segmen agar rapi (1, 2, 3...)
    df_final['Segment_ID'] = df_final.groupby('Segment_ID').ngroup() + 1
    
    # Bersihkan kolom bantuan
    df_final = df_final.drop(columns=['dt_hour', 'break_seg'])
    
    return df_final

def run_eof_extraction():
    print("\n" + "="*60)
    print("🧠 FASE 2: SEPARASI EOF & SEGMENTASI DATETIME")
    print("="*60)

    file_input = PATHS['dataset_arus']
    print(f"🔍 Mencari dataset di: {file_input}")

    if not os.path.exists(file_input):
        print(f"❌ ERROR: File dataset tidak ditemukan di '{file_input}'!")
        print(f"📂 Isi folder '00 Data' saat ini: {os.listdir(PATHS['dataset_arus'].parent) if os.path.exists(PATHS['dataset_arus'].parent) else 'Folder tidak ada'}")
        raise FileNotFoundError(f"File {file_input} wajib ada sebelum EOF Extraction dijalankan.")

    # Load Data Pasut (Dengan Interpolasi) JIKA DIAKTIFKAN
    if FEATURE_TOGGLES.get('ENABLE_WATER_LEVEL', False):
        try:
            print("🌊 Membaca dan resampling data Elevasi Pasut...")
            df_pasut = pd.read_excel(PATHS['aws_database'])
            df_pasut.columns = df_pasut.columns.str.strip()
            df_pasut['datetime'] = pd.to_datetime(df_pasut['datetime'], format='%d/%m/%Y %H:%M:%S')
            
            # Resample 1 Jam lalu Interpolasi agar tidak ada yg bolong
            df_pasut = df_pasut.set_index('datetime').resample('1H').mean()
            df_pasut = df_pasut.interpolate(method='time').reset_index()
            df_pasut = df_pasut.drop_duplicates(subset=['datetime'])
        except Exception as e:
            print(f"⚠️ Peringatan: Gagal memuat data pasut ({e}). Melanjutkan tanpa data pasut.")
            df_pasut = None
    else:
        print("⏭️ Fitur Elevasi Pasut dimatikan di config. Melewati pemrosesan pasut.")
        df_pasut = None

    # 1. LOAD DATA NETCDF
    print("🥞 Memuat data dan meratakan matriks spasial (Stacking)...")
    ds = xr.open_dataset(file_input)
    time_arr = ds.time.values

    # Load file mean u dari Fase 1 hanya untuk mendapatkan Masking Laut yang murni
    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    valid_grids_mask = ds_mean_u['u'].notnull().values.flatten()
    
    stacked = ds.stack(grid=['lat', 'lon'])
    ocean_only = stacked.isel(grid=valid_grids_mask)

    for comp in ['u', 'v']:
        print(f"\n" + "-"*40)
        print(f"🌪️ MENJALANKAN EKSTRAKSI UNTUK KOMPONEN: {comp.upper()}")
        print("-"*40)
        
        comp_matrix = ocean_only[comp].values
        
        # 🚀 MANUVER JENIUS: BUANG RADAR MATI SEBELUM PCA
        # Jika semua titik grid = 0.0 (karena imputasi di Fase 1), berarti radar off
        is_radar_off = (comp_matrix == 0.0).all(axis=1)
        
        comp_matrix_clean = comp_matrix[~is_radar_off]
        time_clean = time_arr[~is_radar_off]
        
        print(f"   -> Membuang {is_radar_off.sum()} jam data saat radar mati total.")
        
        # 2. CHRONOLOGICAL SPLIT (Menggunakan data bersih)
        split_idx = int(len(comp_matrix_clean) * TRAIN_RATIO)
        X_train = comp_matrix_clean[:split_idx]
        X_test  = comp_matrix_clean[split_idx:]
        
        time_train = time_clean[:split_idx]
        time_test  = time_clean[split_idx:]
        
        # 3. EOF ANALYSIS (PCA)
        pca = PCA(n_components=N_MODES, random_state=42)
        pc_train = pca.fit_transform(X_train)
        pc_test = pca.transform(X_test)
        
        explained_var = pca.explained_variance_ratio_ * 100
        print(f"✅ Total Variansi Terpilih ({N_MODES} Mode): {np.sum(explained_var):.2f}%")
        for i, var in enumerate(explained_var):
            print(f"   -> Mode {i+1}: {var:.2f}%")

        # 4. PEMBUATAN DATAFRAME & JOIN PASUT
        print("🔗 Menggabungkan Datetime dan PC...")
        col_names = [f"PC{i+1}" for i in range(N_MODES)]
        
        df_train = pd.DataFrame(pc_train, columns=col_names)
        df_train.insert(0, 'datetime', pd.to_datetime(time_train))
        
        df_test = pd.DataFrame(pc_test, columns=col_names)
        df_test.insert(0, 'datetime', pd.to_datetime(time_test))

        # Gabungkan data pasut HANYA JIKA objek df_pasut berhasil dibuat (tidak None)
        if df_pasut is not None:
            print("🔗 Menyisipkan Data Elevasi Pasut ke dalam Dataset...")
            df_train = pd.merge(df_train, df_pasut, on='datetime', how='left')
            df_test = pd.merge(df_test, df_pasut, on='datetime', how='left')

        # 5. SEGMENTASI DATETIME
        print("🏷️ Melabeli Segmen menggunakan logika Datetime Gap...")
        df_train_final = apply_segmentation_datetime(df_train)
        df_test_final = apply_segmentation_datetime(df_test)

        print(f"   -> Sisa Baris Train: {len(df_train_final)} | Jumlah Segmen: {df_train_final['Segment_ID'].nunique()}")
        print(f"   -> Sisa Baris Test : {len(df_test_final)} | Jumlah Segmen: {df_test_final['Segment_ID'].nunique()}")

        # 6. SIMPAN ARTIFAK
        csv_train_path = str(PATHS['pc_train']).replace('.csv', f'_{comp}.csv')
        csv_test_path = str(PATHS['pc_test']).replace('.csv', f'_{comp}.csv')
        df_train_final.to_csv(csv_train_path, index=False)
        df_test_final.to_csv(csv_test_path, index=False)
        
        model_path = str(PATHS['meof_pca_model']).replace('_model.joblib', f'_{comp}_model.joblib')
        joblib.dump(pca, model_path)

    ds.close()
    print("\n✅ FASE 2 SELESAI!")

if __name__ == "__main__":
    run_eof_extraction()
