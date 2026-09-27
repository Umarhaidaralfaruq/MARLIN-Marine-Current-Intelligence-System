# =============================================================================
# FILE: 02a_eda_comparative_statistics.py
# DESKRIPSI: Modul 2 - Komparasi Distribusi Arus (RAW vs PRE-PROCESSED Cleaned)
#            Menggabungkan fitur komparasi dengan filtering imputasi 0.0
# =============================================================================

import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PATHS

def generate_stats_and_plot(u_valid, v_valid, title_prefix, save_filename):
    """Fungsi helper untuk menghitung deskripsi statistik dan menggambar histogram"""
    speed = np.sqrt(u_valid**2 + v_valid**2)
    
    df_stats = pd.DataFrame({
        'U_Velocity': u_valid,
        'V_Velocity': v_valid,
        'Current_Speed': speed
    })
    
    print(f"\n{'-'*65}")
    print(f"📈 RINGKASAN STATISTIK: {title_prefix.upper()}")
    print(f"{'-'*65}")
    summary = df_stats.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.999]).T
    print(summary[['mean', 'std', 'min', '50%', '95%', '99%', '99.9%', 'max']].round(3))
    print(f"{'-'*65}")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Distribusi Statistik - {title_prefix}", fontsize=16, fontweight='bold')

    # Plot U Velocity
    sns.histplot(df_stats['U_Velocity'], bins=100, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribusi Komponen U (Timur-Barat)', fontsize=12)
    axes[0].set_xlabel('Kecepatan (m/s)')
    axes[0].set_ylabel('Frekuensi Kejadian')
    axes[0].axvline(df_stats['U_Velocity'].mean(), color='red', linestyle='--', label='Mean')

    # Plot V Velocity
    sns.histplot(df_stats['V_Velocity'], bins=100, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Distribusi Komponen V (Utara-Selatan)', fontsize=12)
    axes[1].set_xlabel('Kecepatan (m/s)')
    axes[1].axvline(df_stats['V_Velocity'].mean(), color='red', linestyle='--', label='Mean')

    # Plot Speed (Magnitude)
    sns.histplot(df_stats['Current_Speed'], bins=100, kde=True, ax=axes[2], color='purple')
    axes[2].set_title('Distribusi Total Kecepatan (Speed)', fontsize=12)
    axes[2].set_xlabel('Kecepatan (m/s)')
    axes[2].axvline(df_stats['Current_Speed'].mean(), color='red', linestyle='--', label='Mean')
    
    p99_speed = df_stats['Current_Speed'].quantile(0.99)
    p999_speed = df_stats['Current_Speed'].quantile(0.999)
    axes[2].axvline(p99_speed, color='orange', linestyle='-', linewidth=2, label=f'P99 ({p99_speed:.2f} m/s)')
    axes[2].axvline(p999_speed, color='red', linestyle='--', linewidth=2, label=f'P99.9 ({p999_speed:.2f} m/s)')
    
    for ax in axes:
        ax.legend()

    plt.tight_layout()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Grafik tersimpan di: {save_path}")
    plt.close()

def run_comparative_analysis():
    print("\n" + "="*60)
    print("🌊 MODUL 2: KOMPARASI RAW DATA VS PRE-PROCESSED (CLEANED)")
    print("="*60)

    # =========================================================================
    # 1. ESTRAKSI RAW DATA (BELUM DI-TREATMENT)
    # =========================================================================
    pattern = PATHS["hf_radar_pattern"]
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"❌ Tidak ada file RAW ditemukan dengan pattern: {pattern}")
        return

    datasets_list = []
    print(f"\n🔄 [1/2] Mengekstrak {len(all_files)} file NetCDF RAW...")
    for f in all_files:
        if not os.path.exists(f): continue
        try:
            ds_temp = xr.open_dataset(f, engine='netcdf4')[['u', 'v']].load()
            datasets_list.append(ds_temp)
        except Exception:
            try:
                ds_temp = xr.open_dataset(f, engine='scipy')[['u', 'v']].load()
                datasets_list.append(ds_temp)
            except Exception: pass

    if not datasets_list:
        print("❌ Gagal membaca dataset RAW.")
        return

    print("🧩 Menggabungkan dataset RAW...")
    ds_raw = xr.concat(datasets_list, dim='time').sortby('time')
    ds_raw = ds_raw.sel(time=~ds_raw.get_index('time').duplicated())
    
    u_raw = ds_raw['u'].values.flatten()
    v_raw = ds_raw['v'].values.flatten()
    valid_raw_idx = ~np.isnan(u_raw) & ~np.isnan(v_raw)
    
    generate_stats_and_plot(
        u_raw[valid_raw_idx], 
        v_raw[valid_raw_idx], 
        "RAW DATA (Sebelum Treatment)", 
        "01_distribusi_raw.png"
    )

    # =========================================================================
    # 2. EKSTRAKSI PRE-PROCESSED DATA (DENGAN FILTERING IMPUTASI 0.0)
    # =========================================================================
    preprocessed_path = PATHS['dataset_arus']
    mean_u_path = PATHS['spatial_mean_u']
    mean_v_path = PATHS['spatial_mean_v']

    print(f"\n🔄 [2/2] Membaca data PRE-PROCESSED dari {preprocessed_path} ...")
    if not (os.path.exists(preprocessed_path) and os.path.exists(mean_u_path) and os.path.exists(mean_v_path)):
        print("❌ File hasil Fase 1 (dataset_arus / spatial_mean) tidak lengkap.")
        return

    ds_anom = xr.open_dataset(preprocessed_path)
    ds_mean_u = xr.open_dataset(mean_u_path)
    ds_mean_v = xr.open_dataset(mean_v_path)

    print("🔧 Merekonstruksi kecepatan asli & memfilter padding imputasi (0.0)...")
    
    # Deteksi data yang di-fillna(0.0) paksa saat radar mati/gap panjang
    is_imputed = (ds_anom['u'] == 0.0) & (ds_anom['v'] == 0.0)
    
    # Rekonstruksi ke nilai asli (Anomali + Mean), ubah kembali padding 0.0 menjadi NaN
    u_real = xr.where(is_imputed, np.nan, ds_anom['u'] + ds_mean_u['u'])
    v_real = xr.where(is_imputed, np.nan, ds_anom['v'] + ds_mean_v['v'])
    
    # Flatten dan buang NaN (Daratan + Gap Imputasi) untuk analisis murni
    u_prep_flat = u_real.values.flatten()
    v_prep_flat = v_real.values.flatten()
    
    valid_prep_idx = ~np.isnan(u_prep_flat) & ~np.isnan(v_prep_flat)
    
    generate_stats_and_plot(
        u_prep_flat[valid_prep_idx], 
        v_prep_flat[valid_prep_idx], 
        "PRE-PROCESSED DATA (Cleaned)", 
        "02_distribusi_preprocessed_cleaned.png"
    )

if __name__ == "__main__":
    run_comparative_analysis()