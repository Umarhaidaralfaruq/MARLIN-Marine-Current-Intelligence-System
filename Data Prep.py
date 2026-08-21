# =============================================================================
# FILE: 01_data_preparation.py
# DESKRIPSI: Membersihkan Data, Ekstrak EOF (Semua Data), dan Segmentasi
#            Output akhirnya adalah data_gabung.csv untuk Operasional.
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

def apply_segmentation_datetime(df):
    """
    Logika segmentasi menggunakan lompatan datetime (diff > 3 jam)
    """
    df = df.sort_values('datetime').reset_index(drop=True)
    df['dt_hour'] = df['datetime'].diff().dt.total_seconds() / 3600
    
    df['break_seg'] = df['dt_hour'] > MAX_SAFE_NAN
    df['Segment_ID'] = df['break_seg'].cumsum() + 1
    
    # FILTER LAPIS 1: Buang Segmen yang panjangnya < 48 Jam
    seg_len = df.groupby('Segment_ID').size()
    valid_seg = seg_len[seg_len >= MODEL_CONFIG['output_window']].index
    
    df_final = df[df['Segment_ID'].isin(valid_seg)].copy()
    
    # Urutkan ulang ID Segmen agar rapi (1, 2, 3...)
    df_final['Segment_ID'] = df_final.groupby('Segment_ID').ngroup() + 1
    df_final = df_final.drop(columns=['dt_hour', 'break_seg'])
    
    return df_final

def prepare_operational_data():
    print("\n" + "="*60)
    print("🧹 FASE 1: PENYIAPAN MASTER DATA (DATA_GABUNG.CSV)")
    print("="*60)

    file_input = PATHS['dataset_arus']
    if not os.path.exists(file_input): 
        print(f"❌ File {file_input} tidak ditemukan!")
        return

    # Load Data Pasut (Dengan Interpolasi) JIKA DIAKTIFKAN
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
            df_pasut = None
    else:
        print("⏭️ Fitur Elevasi Pasut dimatikan di config. Melewati pemrosesan pasut.")
        df_pasut = None

    # 1. LOAD DATA NETCDF
    print("🥞 Memuat data NetCDF dan meratakan matriks spasial (Stacking)...")
    ds = xr.open_dataset(file_input)
    time_arr = ds.time.values

    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    valid_grids_mask = ds_mean_u['u'].notnull().values.flatten()

    stacked = ds.stack(grid=['lat', 'lon'])
    ocean_only = stacked.isel(grid=valid_grids_mask)

    # DataFrame Master untuk menampung gabungan U dan V
    df_master = None

    for comp in ['u', 'v']:
        print(f"\n" + "-"*40)
        print(f"🌪️ MENJALANKAN EKSTRAKSI EOF UNTUK: {comp.upper()}")
        print("-"*40)
        
        comp_matrix = ocean_only[comp].values
        
        # 🚀 BUANG RADAR MATI (TIDAK ADA SPLIT TRAIN/TEST)
        is_radar_off = (comp_matrix == 0.0).all(axis=1)
        
        X_clean = comp_matrix[~is_radar_off]
        time_clean = time_arr[~is_radar_off]
        
        print(f"   -> Membuang {is_radar_off.sum()} jam data saat radar mati total.")
        
        # 3. EOF ANALYSIS (Menggunakan SEMUA Data Bersih)
        model_path = str(PATHS['meof_pca_model']).replace('_model.joblib', f'_{comp}_model.joblib')
        pca = joblib.load(model_path)
        pc_all = pca.transform(X_clean)
        
        explained_var = pca.explained_variance_ratio_ * 100
        print(f"✅ Total Variansi Terpilih ({N_MODES} Mode): {np.sum(explained_var):.2f}%")
        
        # 4. PEMBUATAN DATAFRAME SEMENTARA UNTUK KOMPONEN INI
        col_names = [f"{comp.upper()}_PC{i+1}" for i in range(N_MODES)]
        df_comp = pd.DataFrame(pc_all, columns=col_names)
        df_comp.insert(0, 'datetime', pd.to_datetime(time_clean))
        
        # Gabungkan ke Master DataFrame (Merge U dan V berdasarkan Datetime)
        if df_master is None:
            df_master = df_comp
        else:
            df_master = pd.merge(df_master, df_comp, on='datetime', how='inner')

    # 5. PENGGABUNGAN DENGAN DATA PASUT & SEGMENTASI
    if df_pasut is not None:
        print("\n🔗 Menggabungkan Datetime, Seluruh PC (U & V), dan Data Pasut...")
        df_master = pd.merge(df_master, df_pasut, on='datetime', how='inner')
    else:
        print("\n🔗 Menggabungkan Datetime dan Seluruh PC (U & V) saja...")

    print("🏷️ Melabeli Segmen menggunakan logika Datetime Gap...")
    df_final = apply_segmentation_datetime(df_master)

    # 6. SIMPAN KE DATA_GABUNG.CSV
    output_path = PATHS['data_gabung']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    ds.close()
    print(f"\n✅ FASE 1 SELESAI! File Master tersimpan di: {output_path}")

if __name__ == "__main__":
    prepare_operational_data()