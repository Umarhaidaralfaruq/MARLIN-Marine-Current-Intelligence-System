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
from config import PATHS, VALIDASI_DIR

def ekstrak_max_1_jam():
    print("\n" + "="*70)
    print("🔍 MEMULAI VALIDASI OTOMATIS: EKSTRAKSI REAL DATA (1 JAM)")
    print("="*70)
    
    # Path output mengikuti config (pastikan nama config sesuai kebutuhan)
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
    # 2. EKSTRAKSI DATA 10 MENITAN SESUAI METODOLOGI SCRIPT 2
    # =========================================================================
    # Koordinat Area Kecepatan
    lat_min_kec, lat_max_kec = -8.187078, -8.137236
    lon_min_kec, lon_max_kec = 114.399884, 114.438983
    
    # Koordinat Area Arah Arus
    lat_min_arah, lat_max_arah = -8.209161, -8.160084
    lon_min_arah, lon_max_arah = 114.387089, 114.435215
    
    list_df_10m = []
    
    print("⏱️ Mengekstrak nilai P95 (Kecepatan) & Modus (Arah) per 10 menit...")
    for f in all_files:
        if not os.path.exists(f): continue
        try:
            ds = xr.open_dataset(f)
            
            # Potong Area
            area_ds_kec = ds.sel(lat=slice(lat_min_arah, lat_max_arah), lon=slice(lon_min_arah, lon_max_arah))
            area_ds_arah = ds.sel(lat=slice(lat_min_arah, lat_max_arah), lon=slice(lon_min_arah, lon_max_arah))
            
            # A. Ekstrak P95 Kecepatan
            uvvel_vals = area_ds_kec['uvvel'].compute().values # Dimensi: (time, lat, lon)
            uvvel_flat = uvvel_vals.reshape(uvvel_vals.shape[0], -1) # Ratakan lat & lon
            
            p95_10m_list = []
            waktu_list = area_ds_kec['time'].values # Ambil informasi waktu untuk log & dataframe
            
            for t_idx in range(uvvel_flat.shape[0]):
                # Buang nilai NaN
                uv_values = uvvel_flat[t_idx][~np.isnan(uvvel_flat[t_idx])]
                time_str = str(waktu_list[t_idx])[:19] # Ambil string waktu (YYYY-MM-DD HH:MM:SS)
                
                if len(uv_values) > 0:
                    uv_max_baru = float(np.nanpercentile(uv_values, 95))
                else:
                    print(f"   ⚠️ Warning: Data kecepatan kosong di {time_str} ({os.path.basename(f)})")
                    uv_max_baru = np.nan # Pakai np.nan (jangan None) agar Pandas tetap membacanya sebagai float numerik
                    
                p95_10m_list.append(uv_max_baru)
            
            # Masukkan hasil ke DataFrame sementara dan jadikan 'time' sebagai index
            df_temp = pd.DataFrame({'time': waktu_list, 'Real_Max_Speed_ms': p95_10m_list})
            df_temp = df_temp.set_index('time')
            
            # B. Ekstrak Modus Arah Arus
            uvdir_vals = area_ds_arah['uvdir'].compute().values # Dimensi: (time, lat, lon)
            uvdir_flat = uvdir_vals.reshape(uvdir_vals.shape[0], -1) # Ratakan lat & lon
            
            modes_10m = []
            for t_idx in range(uvdir_flat.shape[0]):
                arr_clean = uvdir_flat[t_idx][~np.isnan(uvdir_flat[t_idx])]
                if len(arr_clean) > 0:
                    # Ambil derajat yang paling sering muncul
                    val = float(stats.mode(arr_clean, keepdims=True)[0][0])
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
    
    # Hapus duplikasi waktu (jika ada irisan antar file NetCDF)
    df_all = df_all[~df_all.index.duplicated(keep='last')].sort_index()
    
    # Fungsi custom mencari modus per 1 Jam
    def get_mode(x):
        x = x.dropna()
        if len(x) > 0:
            return float(stats.mode(x, keepdims=True)[0][0])
        return np.nan

    # Proses Resample!
    # Kecepatan -> Ambil nilai Max dalam 1 Jam
    # Arah -> Ambil nilai Modus dalam 1 Jam
    # Jumlah Data -> Hitung ada berapa titik waktu 10 menitan (Max = 6 titik)
    df_1jam = df_all.resample('1H').agg(
        Real_Max_Speed_ms=('Real_Max_Speed_ms', 'max'),
        Arah_Derajat=('Arah_Derajat', get_mode),
        Jumlah_Data=('Real_Max_Speed_ms', 'count')
    )
    
    # Filter hanya data yang ada isinya
    df_1jam = df_1jam[df_1jam['Jumlah_Data'] > 0].reset_index()

    # =========================================================================
    # 4. FORMATTING & KALKULASI KUALITAS DATA
    # =========================================================================
    # Ketersediaan Data: Max 6 point/Jam.
    def tandai_kualitas(jml):
        if jml >= 5:
            return ">75% (Sangat Baik)"
        elif jml >= 3:
            return "50% - 75% (Cukup)"
        else:
            return "<50% (Kurang)" # Ganti keterangan jadi Kurang, bukan Tidak Valid
            
    df_1jam['Kualitas_Data'] = df_1jam['Jumlah_Data'].apply(tandai_kualitas)
    
    # HAPUS ATAU JADIKAN KOMENTAR BARIS DI BAWAH INI:
    # df_1jam = df_1jam[df_1jam['Kualitas_Data'] != "Tidak Valid"]
    
    # Konversi waktu ke standard UTC dan bulatkan angka
    df_1jam['Waktu_UTC'] = df_1jam['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df_baru = df_1jam[['Waktu_UTC', 'Real_Max_Speed_ms', 'Arah_Derajat', 'Kualitas_Data']].copy()
    df_baru['Real_Max_Speed_ms'] = df_baru['Real_Max_Speed_ms'].round(2)
    df_baru['Arah_Derajat'] = df_baru['Arah_Derajat'].round(1)

    # =========================================================================
    # 5. MERGE KE CSV LAMA
    # =========================================================================
    if output_csv.exists():
        print("📂 Menggabungkan dengan database CSV historis...")
        df_old = pd.read_csv(output_csv)
        df_final = pd.concat([df_old, df_baru], ignore_index=True)
        
        # Hapus duplikat waktu dengan menimpa data terlama
        df_final = df_final.drop_duplicates(subset=['Waktu_UTC'], keep='last')
        df_final = df_final.sort_values(by=['Waktu_UTC']).reset_index(drop=True)
    else:
        df_final = df_baru

    # Simpan
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    
    print("\n📊 PREVIEW HASIL DATABASE 1-JAMAN (10 Terakhir):")
    print("-" * 65)
    print(df_final.tail(10).to_string(index=False))
    print("-" * 65)
    
    print(f"✅ Database validasi berhasil diperbarui! Total data sekarang: {len(df_final)} baris.")
    print(f"💾 Disimpan di: {output_csv}")
    
    return df_final

if __name__ == "__main__":
    ekstrak_max_1_jam()