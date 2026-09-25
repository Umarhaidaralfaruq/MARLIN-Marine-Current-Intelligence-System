# =============================================================================
# FILE: 02a_eda_statistics.py
# DESKRIPSI: Modul 2 - Statistik Dasar & Distribusi Fisik Arus Laut (U, V, Speed)
# =============================================================================

import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PATHS

def run_physical_statistics():
    print("\n" + "="*60)
    print("📊 MODUL 2: ANALISIS STATISTIK FISIK ARUS LAUT")
    print("="*60)

    # 1. LOAD DATA FASE 1 (Anomali & Mean)
    print("📂 Memuat dataset anomali dan rata-rata spasial...")
    ds_anom = xr.open_dataset(PATHS['dataset_arus'])
    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    ds_mean_v = xr.open_dataset(PATHS['spatial_mean_v'])

    # 2. REKONSTRUKSI NILAI FISIK & FILTERING IMPUTASI 0.0
    print("🔧 Merekonstruksi kecepatan asli dan membuang nilai imputasi 0.0...")
    
    # Deteksi area laut yang valid (Bukan daratan)
    valid_mask_2d = ds_mean_u['u'].notnull()
    
    # Deteksi data yang sebelumnya di-fillna(0.0) paksa saat Fase 1
    is_imputed = (ds_anom['u'] == 0.0) & (ds_anom['v'] == 0.0)
    
    # Kembalikan ke nilai asli (Anomali + Mean), jadikan NaN jika itu hasil imputasi
    u_real = xr.where(is_imputed, np.nan, ds_anom['u'] + ds_mean_u['u'])
    v_real = xr.where(is_imputed, np.nan, ds_anom['v'] + ds_mean_v['v'])
    
    # Ratakan (flatten) dan buang semua NaN/Daratan untuk analisis statistik murni
    u_flat = u_real.values.flatten()
    v_flat = v_real.values.flatten()
    
    valid_idx = ~np.isnan(u_flat) & ~np.isnan(v_flat)
    u_valid = u_flat[valid_idx]
    v_valid = v_flat[valid_idx]
    
    # 3. KALKULASI FISIKA (SPEED & DIRECTION)
    print("🧮 Menghitung Magnitude (Speed) dan Arah (Direction)...")
    speed = np.sqrt(u_valid**2 + v_valid**2)
    
    # 4. TABEL RINGKASAN STATISTIK (DESCRIPTIVE STATISTICS)
    df_stats = pd.DataFrame({
        'U_Velocity (Timur-Barat)': u_valid,
        'V_Velocity (Utara-Selatan)': v_valid,
        'Current_Speed (Magnitude)': speed
    })
    
    print("\n" + "-"*60)
    print("📈 RINGKASAN STATISTIK DATA MENTAH (Satuan: m/s)")
    print("-"*60)
    summary = df_stats.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).T
    print(summary[['mean', 'std', 'min', '50%', '95%', '99%', 'max']].round(3))
    print("-"*60)

    # =========================================================================
    # 5. VISUALISASI DISTRIBUSI (HISTOGRAM & KDE)
    # =========================================================================
    print("\n🎨 Menggambar grafik distribusi data...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Distribusi Statistik Kecepatan Arus Laut Selat Bali", fontsize=16, fontweight='bold')

    # Plot U Velocity
    sns.histplot(df_stats['U_Velocity (Timur-Barat)'], bins=100, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribusi Komponen U', fontsize=12)
    axes[0].set_xlabel('Kecepatan (m/s)')
    axes[0].set_ylabel('Frekuensi Kejadian')
    axes[0].axvline(df_stats['U_Velocity (Timur-Barat)'].mean(), color='red', linestyle='--', label='Mean')

    # Plot V Velocity
    sns.histplot(df_stats['V_Velocity (Utara-Selatan)'], bins=100, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Distribusi Komponen V', fontsize=12)
    axes[1].set_xlabel('Kecepatan (m/s)')
    axes[1].axvline(df_stats['V_Velocity (Utara-Selatan)'].mean(), color='red', linestyle='--', label='Mean')

    # Plot Speed (Magnitude)
    sns.histplot(df_stats['Current_Speed (Magnitude)'], bins=100, kde=True, ax=axes[2], color='purple')
    axes[2].set_title('Distribusi Total Kecepatan (Speed)', fontsize=12)
    axes[2].set_xlabel('Kecepatan (m/s)')
    axes[2].axvline(df_stats['Current_Speed (Magnitude)'].mean(), color='red', linestyle='--', label='Mean')
    
    # Garis P99 (Batas Anomali Ekstrem)
    p99_speed = df_stats['Current_Speed (Magnitude)'].quantile(0.99)
    axes[2].axvline(p99_speed, color='orange', linestyle='-', linewidth=2, label=f'P99 ({p99_speed:.2f})')
    
    for ax in axes:
        ax.legend()

    plt.tight_layout()
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "distribusi_statistik_arus.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Grafik distribusi tersimpan di: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_physical_statistics()
