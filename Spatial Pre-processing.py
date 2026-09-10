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

from config import PATHS, MODEL_CONFIG, SPATIAL_CONFIG
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

def resample_velocity(ds, method='max_amplitude', freq='1H', u_var='u', v_var='v'):
    
    # ==========================================
    # 1. HITUNG JUMLAH REKAMAN ASLI PER JAM
    # ==========================================
    # Buat array 1D berisi angka 1 untuk setiap timestep mentah
    dummy_counts = xr.DataArray(np.ones(len(ds.time)), coords=[ds.time], dims=['time'])
    
    # Hitung total kemunculan data mentah per blok waktu resample (misal per 1 Jam)
    time_records_count = dummy_counts.resample(time=freq).sum()
    
    # Tetapkan batas minimal (Misal minimal butuh 3 atau 4 data dari total 6 data 10-menitan)
    # Bisa juga dimasukkan ke SPATIAL_CONFIG['min_temporal_records']
    MIN_TEMPORAL_RECORDS = SPATIAL_CONFIG.get('min_temporal_records', 3)
    # ==========================================

    # 2. LAKUKAN PROSES RESAMPLE SEPERTI BIASA
    if method == 'mean':
        print(f"🔄 Resample ({freq}) menggunakan metode: MEAN")
        ds_resampled = ds.resample(time=freq).mean()
        
    elif method == 'max_amplitude':
        print(f"🔄 Resample ({freq}) menggunakan metode: MAX AMPLITUDE (Vector-Preserving)")
        
        def _get_vector_at_max_speed(subset_ds):
            if len(subset_ds.time) == 0:
                return subset_ds
                
            speed = np.sqrt(subset_ds[u_var]**2 + subset_ds[v_var]**2)
            idx = speed.fillna(-1.0).argmax(dim='time')
            return subset_ds.isel(time=idx)
            
        ds_resampled = ds.resample(time=freq).map(_get_vector_at_max_speed)
        
    elif method == 'p95':
        print(f"🔄 Resample ({freq}) menggunakan metode: P95 (95th Percentile Extreme)")
        ds_p95 = ds.resample(time=freq).quantile(0.95).drop_vars('quantile', errors='ignore')
        ds_p05 = ds.resample(time=freq).quantile(0.05).drop_vars('quantile', errors='ignore')
        ds_resampled = xr.where(abs(ds_p05) > abs(ds_p95), ds_p05, ds_p95)
        
    else:
        raise ValueError(f"Metode '{method}' tidak didukung.")

    # ==========================================
    # 3. TERAPKAN MASKING BERDASARKAN SYARAT TEMPORAL
    # ==========================================
    # Grid yang jumlah rekaman aslinya kurang dari batas akan otomatis diubah jadi NaN
    ds_resampled = ds_resampled.where(time_records_count >= MIN_TEMPORAL_RECORDS)
    
    dropped_hours = int((time_records_count < MIN_TEMPORAL_RECORDS).sum())
    if dropped_hours > 0:
        print(f"⚖️ Menganulir {dropped_hours} jam hasil resample karena rekaman data mentah < {MIN_TEMPORAL_RECORDS}.")

    return ds_resampled

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
            # Coba buka dengan engine h5netcdf terlebih dahulu
            # Cek ukuran file
            ukuran_kb = os.path.getsize(f) / 1024
            print(f"📊 Ukuran file {os.path.basename(f)}: {ukuran_kb:.2f} KB")
            
            # Intip 5 baris pertama jika ukurannya sangat kecil (< 50 KB)
            if ukuran_kb < 50:
                with open(f, 'r', encoding='utf-8', errors='ignore') as text_file:
                    print("👀 Isi file sebenarnya:\n", text_file.read(200))
                    
            ds_temp = xr.open_dataset(f, engine='netcdf4')[['u', 'v']].load()
            datasets_list.append(ds_temp)
        except ValueError:
            try:
                # Fallback jika format NetCDF3 klasik
                ds_temp = xr.open_dataset(f, engine='scipy')[['u', 'v']].load()
                datasets_list.append(ds_temp)
            except Exception as e:
                print(f"⚠️ SKIP File gagal di-fallback: {os.path.basename(f)} | Penyebab: {e}")
        except Exception as e:
            # Print error asli agar kita tahu file benar-benar rusak atau masalah dependensi
            print(f"⚠️ SKIP File bermasalah: {os.path.basename(f)} | Penyebab: {e}")
            
    if len(datasets_list) == 0:
        print("❌ Tidak ada file NetCDF yang bisa dibuka. Coba periksa ukuran file tersebut.")
        return None

    try:
        ds = xr.concat(datasets_list, dim='time')
        # Ini yang bikin urutan file gak masalah, xarray yang ngurutin otomatis!
        ds = ds.sortby('time') 
        ds = ds.sel(time=~ds.get_index('time').duplicated())
        
        # ==========================================
        # TREATMENT: POTONG JAM TERAKHIR JIKA BELUM PENUH
        # ==========================================
        import pandas as pd
        
        # Ambil waktu paling ujung/terakhir dari dataset
        waktu_terakhir = pd.to_datetime(ds.time.values[-1])
        
        # Cek apakah menit terakhir belum menyentuh 50
        if waktu_terakhir.minute < 50:
            # Mundurkan batas waktu ke jam sebelumnya (Contoh: 11:30 jadi 10:59:59)
            batas_waktu = waktu_terakhir.floor('h') - pd.Timedelta(seconds=1)
            
            # Potong dataset hanya sampai batas waktu tersebut
            ds = ds.sel(time=slice(None, batas_waktu))
            print(f"✂️ CUTOFF: Data jam {waktu_terakhir.hour}:00 dibuang karena tidak penuh (hanya sampai menit {waktu_terakhir.minute}).")
        # ==========================================

        # ==========================================
        # TREATMENT: SPATIAL THRESHOLD PER TIMESTEP
        # ==========================================
        # Asumsi dimensi spasial bernama 'lat' dan 'lon' (sesuaikan jika 'x'/'y')
        spatial_dims = [dim for dim in ds.dims if dim != 'time']
        
        # Hitung total grid yang berisi angka (bukan NaN) di setiap timestep
        valid_points_per_time = ds['u'].notnull().sum(dim=spatial_dims)
        
        # Tetapkan batas minimal dari config (misal minimal 200 titik grid per timestep)
        MIN_VALID_POINTS = SPATIAL_CONFIG.get('min_spatial_points_per_timestep', 200)
        
        # Filter timestep yang memenuhi syarat
        valid_time_mask = valid_points_per_time >= MIN_VALID_POINTS
        
        dropped_timesteps = int((~valid_time_mask).sum())
        if dropped_timesteps > 0:
            print(f"✂️ Membuang {dropped_timesteps} timestep karena titik spasial valid < {MIN_VALID_POINTS}.")
        
        # Update dataset hanya dengan waktu yang valid
        ds = ds.sel(time=valid_time_mask)

        # Setelah bersih, baru dilakukan resample
        ds = resample_velocity(
            ds, 
            method=SPATIAL_CONFIG['RESAMPLE_METHOD'], 
            freq=SPATIAL_CONFIG['RESAMPLE_FREQ']
        )
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
    
    # Tambahkan parameter max_gap agar gap yang lebih besar dari limit tidak disentuh sama sekali
    ds_filled = ds.interpolate_na(
        dim='time', 
        method='polynomial', 
        limit=MODEL_CONFIG['max_save_nan'],
        max_gap=f"{MODEL_CONFIG['max_save_nan']}h",
        order=3
    )

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
    
    print(f"🧼 8. Imputasi Akhir (Sisa NaN > {MODEL_CONFIG['max_save_nan']} Jam menjadi 0.0 murni)...")
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
