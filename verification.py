# =============================================================================
# FILE: 06_auto_validation.py (UPDATED)
# DESKRIPSI: Ekstraksi Otomatis Nilai Maksimum & Arah HF Radar (Resolusi 1 Jam)
#            Sinkronisasi metodologi dengan 2_Memproses_HFRadar_Ketapang-Gilimanuk
# =============================================================================

import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import circmean # Tambahan pustaka untuk menghitung rata-rata arah/derajat

# Tambahkan VALIDATION_CONFIG di import
from config import PATHS, VALIDASI_DIR, VALIDATION_CONFIG 

def ekstrak_max_1_jam():
    print("\n" + "="*70)
    print("🔍 MEMULAI VALIDASI OTOMATIS: EKSTRAKSI REAL DATA (1 JAM)")
    print("="*70)
    
    output_csv = PATHS["Verif_12jam"] 
    file_terbaru = PATHS["nc_file_arus"]
    all_files = []
    
    # =========================================================================
    # 1. LOGIKA INCREMENTAL LOADING
    # =========================================================================
    if not output_csv.exists():
        print("🚀 [MODE AWAL] Database CSV belum ada. Menarik SELURUH file historis...")
        folder_path = PATHS["hf_radar_pattern"]
        file_lokal = sorted(glob.glob(str(folder_path))) 
        all_files.extend(file_lokal)
        
        if file_terbaru.exists():
            all_files.append(str(file_terbaru))
    else:
        print("⚡ [MODE UPDATE] Database CSV ditemukan. Hanya memproses 1 file terbaru...")
        if file_terbaru.exists():
            all_files.append(str(file_terbaru))
        else:
            print(f"❌ File terbaru tidak ditemukan di {file_terbaru}. Proses dihentikan.")
            return None

    if not all_files:
        print("❌ Tidak ada file NetCDF yang bisa diproses.")
        return None
        
    print(f"🔍 Total file NetCDF yang akan diekstrak: {len(all_files)}")

    # =========================================================================
    # 2. EKSTRAKSI DATA 10 MENITAN
    # =========================================================================
    lat_min_kec, lat_max_kec = -8.187078, -8.137236
    lon_min_kec, lon_max_kec = 114.399884, 114.438983
    
    lat_min_arah, lat_max_arah = -8.209161, -8.160084
    lon_min_arah, lon_max_arah = 114.387089, 114.435215
    
    list_df_10m = []
    
    print("⏱️ Mengekstrak nilai P95 (Kecepatan) & Circular Mean (Arah) per 10 menit...")
    for f in all_files:
        if not os.path.exists(f): continue
        try:
            ds = xr.open_dataset(f)
            
            area_ds_kec = ds.sel(lat=slice(lat_min_arah, lat_max_arah), lon=slice(lon_min_arah, lon_max_arah))
            area_ds_arah = ds.sel(lat=slice(lat_min_arah, lat_max_arah), lon=slice(lon_min_arah, lon_max_arah))
            
            uvvel_vals = area_ds_kec['uvvel'].compute().values
            uvvel_flat = uvvel_vals.reshape(uvvel_vals.shape[0], -1)
            
            p95_10m_list = []
            waktu_list = area_ds_kec['time'].values
            
            for t_idx in range(uvvel_flat.shape[0]):
                uv_values = uvvel_flat[t_idx][~np.isnan(uvvel_flat[t_idx])]
                time_str = str(waktu_list[t_idx])[:19]
                
                if len(uv_values) > 0:
                    uv_max_baru = float(np.nanpercentile(uv_values, 95))
                else:
                    uv_max_baru = np.nan
                    
                p95_10m_list.append(uv_max_baru)
            
            df_temp = pd.DataFrame({'time': waktu_list, 'Real_Max_Speed_ms': p95_10m_list})
            df_temp = df_temp.set_index('time')
            
            uvdir_vals = area_ds_arah['uvdir'].compute().values
            uvdir_flat = uvdir_vals.reshape(uvdir_vals.shape[0], -1)
            
            modes_10m = []
            for t_idx in range(uvdir_flat.shape[0]):
                arr_clean = uvdir_flat[t_idx][~np.isnan(uvdir_flat[t_idx])]
                if len(arr_clean) > 0:
                    # 1. Hitung rata-rata sirkular derajat asli
                    val_asli = float(circmean(arr_clean, high=360, low=0))
                    # 2. Balik arahnya 180 derajat (Flowing From -> Flowing To)
                    val_dibalik = (val_asli + 180) % 360
                    val = val_dibalik
                else:
                    val = np.nan
                modes_10m.append(val)
                
            df_temp['Arah_Derajat'] = modes_10m
            list_df_10m.append(df_temp)
            
            ds.close()
        except Exception as e:
            print(f"   ⚠️ SKIP File rusak/korup: {os.path.basename(f)} | Error: {e}")

    if not list_df_10m:
        print("❌ Gagal mengekstrak data dari semua file.")
        return None

    # =========================================================================
    # 3. GABUNGKAN & RESAMPLE KE RESOLUSI 1 JAM
    # =========================================================================
    print("🔗 Menggabungkan dan meresample data ke resolusi 1 Jam...")
    df_all = pd.concat(list_df_10m)
    df_all = df_all[~df_all.index.duplicated(keep='last')].sort_index()
    
    # PERUBAHAN: Fungsi agregasi dari mode ke circular mean
    # Fungsi agregasi ke 1 Jam (Tidak perlu ditambah 180 lagi di sini)
    def get_circular_mean(x):
        x = x.dropna()
        if len(x) > 0:
            return float(circmean(x, high=360, low=0))
        return np.nan

    df_1jam = df_all.resample('1H').agg(
        Real_Max_Speed_ms=('Real_Max_Speed_ms', 'max'),
        Arah_Derajat=('Arah_Derajat', get_circular_mean), # Panggil fungsi baru
        Jumlah_Data=('Real_Max_Speed_ms', 'count')
    ).reset_index()
    
    df_1jam = df_1jam[df_1jam['Jumlah_Data'] > 0]

    # =========================================================================
    # 4. FILTERING KELOMPOK TIMEFRAME (MENGGUNAKAN config.py)
    # =========================================================================
    window_hours = VALIDATION_CONFIG['window_hours']
    threshold_percent = VALIDATION_CONFIG['threshold_percent']
    
    max_data_per_window = window_hours * 6 # 1 jam = 6 data (10 menitan)
    min_data_required = int(max_data_per_window * threshold_percent)
    
    df_1jam['Group_Time'] = df_1jam['time'].dt.floor(f'{window_hours}H')
    group_counts = df_1jam.groupby('Group_Time')['Jumlah_Data'].sum().reset_index()
    group_counts.rename(columns={'Jumlah_Data': 'Total_Data_Kelompok'}, inplace=True)
    
    df_1jam = df_1jam.merge(group_counts, on='Group_Time', how='left')
    
    def tandai_kualitas_kelompok(total):
        if total >= min_data_required:
            return "Lulus"
        else:
            return "Tidak Lulus"
            
    df_1jam['Kualitas_Data'] = df_1jam['Total_Data_Kelompok'].apply(tandai_kualitas_kelompok)

    df_1jam['Waktu_UTC'] = df_1jam['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df_baru = df_1jam[['Waktu_UTC', 'Real_Max_Speed_ms', 'Arah_Derajat', 'Kualitas_Data', 'Jumlah_Data']].copy()
    df_baru['Real_Max_Speed_ms'] = df_baru['Real_Max_Speed_ms'].round(2)
    df_baru['Arah_Derajat'] = df_baru['Arah_Derajat'].round(1)

    # =========================================================================
    # 5. MERGE KE CSV LAMA
    # =========================================================================
    if output_csv.exists():
        print("📂 Menggabungkan dengan database CSV historis...")
        df_old = pd.read_csv(output_csv)
        
        # --- TAMBAHAN UNTUK MENYERAGAMKAN DATA LAMA ---
        mapping_kualitas = {
            ">75% (Sangat Baik)": "Lulus",
            "50% - 75% (Cukup)": "Lulus",
            "<50% (Kurang)": "Tidak Lulus",
            "Tidak Valid": "Tidak Lulus"
        }
        df_old['Kualitas_Data'] = df_old['Kualitas_Data'].replace(mapping_kualitas)
        # ----------------------------------------------

        df_final = pd.concat([df_old, df_baru], ignore_index=True)
        
        # Hapus duplikat waktu dengan menimpa data terlama
        df_final = df_final.drop_duplicates(subset=['Waktu_UTC'], keep='last')
        df_final = df_final.sort_values(by=['Waktu_UTC']).reset_index(drop=True)
    else:
        df_final = df_baru

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    
    print("\n📊 PREVIEW HASIL DATABASE 1-JAMAN (10 Terakhir):")
    print("-" * 65)
    print(df_final.tail(10).to_string(index=False))
    print("-" * 65)
    
    print(f"✅ Database validasi berhasil diperbarui! Total data sekarang: {len(df_final)} baris.")
    print(f"💾 Disimpan di: {output_csv}")
    
    return df_final

def ekstrak_spasial_1_jam():
    print("\n" + "="*70)
    print("🗺️ MEMULAI VALIDASI SPASIAL: EKSTRAKSI U & V (1 JAM)")
    print("="*70)
    
    output_csv = PATHS.get("Verif_Spasial", "Hasil_Validasi/Validasi_Spasial_1H.csv")
    file_terbaru = PATHS["nc_file_arus"]
    all_files = []
    
    if not os.path.exists(output_csv):
        print("🚀 [MODE AWAL] Tarik semua file untuk Spasial...")
        all_files.extend(sorted(glob.glob(str(PATHS["hf_radar_pattern"]))))
        if file_terbaru.exists(): all_files.append(str(file_terbaru))
    else:
        print("⚡ [MODE UPDATE] Update data Spasial dari file terbaru...")
        if file_terbaru.exists(): all_files.append(str(file_terbaru))
        else: return None

    if not all_files: return None

    list_df_spasial = []
    
    print("⏱️ Mengekstrak komponen vektor U dan V per jam...")
    for f in all_files:
        if not os.path.exists(f): continue
        try:
            ds = xr.open_dataset(f)
            area_ds = ds
            
            # KITA BUANG SEMUA HITUNGAN SIN/COS. 
            # Asumsi nama variabel di NetCDF lu adalah 'u' dan 'v' (sesuaikan kalau beda)
            # Langsung resample komponen U dan V bawaan radarnya:
            ds_1h = area_ds[['u', 'v']].resample(time='1H').mean(skipna=True)
            
            # Konversi ke Pandas
            df_temp = ds_1h.to_dataframe().reset_index().dropna(subset=['u', 'v'])
            list_df_spasial.append(df_temp)
            ds.close()
        except Exception as e:
            print(f"   ⚠️ SKIP File rusak: {os.path.basename(f)} | Error: {e}")

    if not list_df_spasial: return None

    df_all = pd.concat(list_df_spasial)
    
    # Rapikan nama kolomnya
    df_all = df_all.rename(columns={
        'u': 'U_obs',
        'v': 'V_obs'
    })
    
    df_all['Waktu_UTC'] = df_all['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df_baru = df_all[['Waktu_UTC', 'lat', 'lon', 'U_obs', 'V_obs']].copy()
    df_baru['U_obs'] = df_baru['U_obs'].round(4)
    df_baru['V_obs'] = df_baru['V_obs'].round(4)

    if os.path.exists(output_csv):
        print("📂 Menggabungkan dengan database Spasial historis...")
        df_old = pd.read_csv(output_csv)
        df_final = pd.concat([df_old, df_baru], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=['Waktu_UTC', 'lat', 'lon'], keep='last')
    else:
        df_final = df_baru

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    
    print(f"✅ Database Spasial berhasil diperbarui! ({len(df_final)} titik data)")
    return df_final

if __name__ == "__main__":
    # 1. Jalankan ekstraksi untuk grafik garis (Time-series)
    ekstrak_max_1_jam()
    
    # 2. Jalankan ekstraksi untuk peta (Spasial)
    ekstrak_spasial_1_jam()
