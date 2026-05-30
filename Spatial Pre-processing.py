# =============================================================================
# FILE: Spatial Pre-processing.py
# DESKRIPSI: Akuisisi data BMKG, Cleaning, Masking, dan Anomaly Calculation
# =============================================================================

import xarray as xr
import numpy as np
import glob
import os
import requests
import warnings

from config import PATHS, MODEL_CONFIG
warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

# Definisikan nama file output spesifik untuk modul ini
NC_LATEST_FILE = PATHS["nc_file_arus"]
TRAIN_RATIO = MODEL_CONFIG['train_split_ratio']

# =============================================================================
# FUNGSI QC (FASE 1)
# =============================================================================
def filter_temporal_iqr(da):
    """Menghapus outlier menggunakan Temporal IQR absolut (Vectorized)"""
    Q1 = da.quantile(0.25, dim='time')
    Q3 = da.quantile(0.75, dim='time')
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Grid yang nilainya di luar batas akan otomatis diubah menjadi NaN
    is_normal = (da >= lower_bound) & (da <= upper_bound)
    return da.where(is_normal)

def run_data_preparation():
    print("\n" + "="*50)
    print("🌊 FASE 1: AKUISISI & SPATIAL DATA PREPARATION")
    print("="*50)

    all_files = []

    # ---------------------------------------------------------------------------
    # 1. MENGUMPULKAN FILE LOKAL & TAMBAHAN
    # ---------------------------------------------------------------------------
    print("📁 2. Mengumpulkan file lokal...")
    folder_path = PATHS["hf_radar_pattern"]
    file_lokal = sorted(glob.glob(folder_path))
    all_files.extend(file_lokal)

    # Tambahan File Extra
    file_tambahan = PATHS["nc_file_arus"] 
    if os.path.exists(file_tambahan):
        all_files.append(file_tambahan)
    else:
        print(f"⚠️ Peringatan: File tambahan tidak ditemukan di {file_tambahan}")

    print(f"\n🔍 Total file yang akan diproses: {len(all_files)}")
    if not all_files:
        print("❌ Tidak ada file NetCDF ditemukan.")
        return None

    # ---------------------------------------------------------------------------
    # 2. BUKA FILE & GABUNGKAN (XARRAY)
    # ---------------------------------------------------------------------------
    datasets_list = []
    print(f"🔄 3. Membuka dan mengekstrak data spasial...")
    for f in all_files:
        if not os.path.exists(f): continue
        try:
            # Load ke RAM biar kalkulasi cepet & gak kena error chunk Dask
            ds_temp = xr.open_dataset(f, engine='netcdf4')[['u', 'v']].load()
            datasets_list.append(ds_temp)
        except Exception:
            print(f"⚠️ SKIP File rusak/korup: {os.path.basename(f)}")
            
    if len(datasets_list) == 0:
        print("❌ Tidak ada file NetCDF yang bisa dibuka.")
        return None

    try:
        ds = xr.concat(datasets_list, dim='time')
        # Ini yang bikin urutan file gak masalah, xarray yang ngurutin otomatis!
        ds = ds.sortby('time') 
        ds = ds.sel(time=~ds.get_index('time').duplicated())
        ds = ds.resample(time='1H').mean()
        print(f"✅ Dataset Tergabung: {ds.dims}")
    except Exception as e:
        print(f"❌ Gagal menggabungkan: {e}")
        return None

    # ---------------------------------------------------------------------------
    # 3. PROSES SPASIAL & CLEANING (FIXED)
    # ---------------------------------------------------------------------------
    print("🏝️ 4. Menerapkan Ocean Masking...")
    ocean_mask = (ds['u'].notnull().sum(dim='time') / len(ds.time)) > MODEL_CONFIG['ocean_threshold']
    ds = ds.where(ocean_mask)

    print("🧹 5. Menghapus Outlier (Temporal IQR -> NaN)...")
    ds['u'] = filter_temporal_iqr(ds['u'])
    ds['v'] = filter_temporal_iqr(ds['v'])

    print(f"🩹 6. Menambal data bolong (Interpolation maks {MODEL_CONFIG['max_save_nan']} jam)...")
    # HANYA interpolasi waktu (maks 3 jam). Dilarang keras ffill spasial (lon/lat)!
    ds_filled = ds.interpolate_na(dim='time', method='linear', limit=MODEL_CONFIG['max_save_nan'])

    print("📉 7. Menghitung Anomali Spasial (Mean-Centering Bebas Leakage)...")
    
    # A. Cari tahu batas indeks untuk pemotongan (Splitting)
    total_time_steps = len(ds_filled.time)
    split_idx = int(total_time_steps * TRAIN_RATIO)
    
    # B. Isolasi data HANYA untuk porsi Training
    # .isel() digunakan untuk memotong berdasarkan indeks
    ds_train_only = ds_filled.isel(time=slice(0, split_idx))
    
    # C. Hitung rata-rata murni HANYA dari masa lalu (Data Train)
    u_mean = ds_train_only['u'].mean(dim='time')
    v_mean = ds_train_only['v'].mean(dim='time')
    
    # D. Kurangi seluruh rentang waktu (Train & Test) dengan rata-rata Train
    ds_anom = ds_filled.copy()
    ds_anom['u'] = ds_filled['u'] - u_mean
    ds_anom['v'] = ds_filled['v'] - v_mean

    # Pake mode='w' biar selalu nimpa file rata-rata yang lama pakai update terbaru
    u_mean.to_netcdf(PATHS['spatial_mean_u'], mode='w')
    v_mean.to_netcdf(PATHS['spatial_mean_v'], mode='w')
    
    print("🧼 8. Imputasi Akhir (Sisa NaN > 3 Jam menjadi 0.0 murni)...")
    # Merubah sisa celah waktu yang besar menjadi netral (0.0) untuk PCA
    ds_final = ds_anom.fillna(0.0)

    ds_final.to_netcdf(PATHS['dataset_arus'])
    
    # Cleanup memory
    for d in datasets_list: d.close()
    
    print(f"✅ Fase 1 Selesai! Data siap diproses ke Fase 2 (EOF).")
    print(f"📅 Rentang Waktu: {ds_final.time.min().values} s/d {ds_final.time.max().values}")
    
    return ds_final

if __name__ == "__main__":
    run_data_preparation()