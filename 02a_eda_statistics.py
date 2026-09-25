# =============================================================================
# FILE: 02a_eda_raw_statistics.py
# DESKRIPSI: Modul 2 - Statistik Distribusi Arus Mentah (Raw Data)
#            Menggunakan metode ekstraksi XR Concat & config PATHS
# =============================================================================

import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import PATHS dari config lu
from config import PATHS

def run_raw_physical_statistics():
    print("\n" + "="*60)
    print("🌊 MODUL 2: ANALISIS STATISTIK ARUS MENTAH (RAW DATA)")
    print("="*60)

    # ---------------------------------------------------------------------------
    # 1. BUKA FILE & GABUNGKAN (METODE XARRAY CONCAT MANUAL)
    # ---------------------------------------------------------------------------
    # Gunakan pattern HF Radar dari config.py
    pattern = PATHS["hf_radar_pattern"]
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"❌ Tidak ada file .nc ditemukan dengan pattern: {pattern}")
        return

    datasets_list = []
    print(f"🔄 Membuka dan mengekstrak {len(all_files)} file NetCDF mentah dari {pattern} ...")
    
    for f in all_files:
        if not os.path.exists(f): continue
        try:
            # Coba buka dengan engine netcdf4 terlebih dahulu
            ukuran_kb = os.path.getsize(f) / 1024
            
            # Intip 5 baris pertama jika ukurannya sangat kecil (< 50 KB)
            if ukuran_kb < 50:
                with open(f, 'r', encoding='utf-8', errors='ignore') as text_file:
                    pass # Bypass print biar terminal nggak terlalu penuh saat EDA
                    
            ds_temp = xr.open_dataset(f, engine='netcdf4')[['u', 'v']].load()
            datasets_list.append(ds_temp)
        except ValueError:
            try:
                # Fallback jika format NetCDF3 klasik
                ds_temp = xr.open_dataset(f, engine='scipy')[['u', 'v']].load()
                datasets_list.append(ds_temp)
            except Exception as e:
                pass
        except Exception as e:
            pass
            
    if len(datasets_list) == 0:
        print("❌ Tidak ada file NetCDF yang bisa dibuka.")
        return

    print("🧩 Menggabungkan dan mengurutkan dimensi waktu...")
    try:
        ds_raw = xr.concat(datasets_list, dim='time')
        ds_raw = ds_raw.sortby('time') 
        ds_raw = ds_raw.sel(time=~ds_raw.get_index('time').duplicated())
    except Exception as e:
        print(f"❌ Gagal menggabungkan dataset: {e}")
        return

    # ---------------------------------------------------------------------------
    # 2. EKSTRAKSI KOMPONEN MURNI & HAPUS NAN (DARATAN/SENSOR MATI)
    # ---------------------------------------------------------------------------
    print("🔧 Mengekstrak matriks U dan V murni...")
    
    # Ratakan array 3D (Time x Lat x Lon) menjadi 1D
    u_flat = ds_raw['u'].values.flatten()
    v_flat = ds_raw['v'].values.flatten()
    
    # Filter hanya data yang benar-benar ada angkanya (bukan daratan / bukan radar mati)
    valid_idx = ~np.isnan(u_flat) & ~np.isnan(v_flat)
    u_valid = u_flat[valid_idx]
    v_valid = v_flat[valid_idx]
    
    # ---------------------------------------------------------------------------
    # 3. KALKULASI FISIKA (SPEED) & TABEL STATISTIK
    # ---------------------------------------------------------------------------
    print("🧮 Menghitung Magnitude (Speed)...")
    speed = np.sqrt(u_valid**2 + v_valid**2)
    
    df_stats = pd.DataFrame({
        'U_Velocity (Timur-Barat)': u_valid,
        'V_Velocity (Utara-Selatan)': v_valid,
        'Current_Speed (Magnitude)': speed
    })
    
    print("\n" + "-"*65)
    print("📈 RINGKASAN STATISTIK RAW DATA (Belum di-Treatment)")
    print("-" * 65)
    summary = df_stats.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.999]).T
    print(summary[['mean', 'std', 'min', '50%', '95%', '99%', '99.9%', 'max']].round(3))
    print("-" * 65)

    # ---------------------------------------------------------------------------
    # 4. VISUALISASI DISTRIBUSI (HISTOGRAM & KDE)
    # ---------------------------------------------------------------------------
    print("\n🎨 Menggambar grafik distribusi data mentah...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Distribusi Statistik Kecepatan Arus Mentah (Sebelum Outlier Treatment)", fontsize=16, fontweight='bold')

    # Plot U Velocity
    sns.histplot(df_stats['U_Velocity (Timur-Barat)'], bins=150, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribusi RAW Komponen U', fontsize=12)
    axes[0].set_xlabel('Kecepatan (m/s)')
    axes[0].set_ylabel('Frekuensi Kejadian')

    # Plot V Velocity
    sns.histplot(df_stats['V_Velocity (Utara-Selatan)'], bins=150, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Distribusi RAW Komponen V', fontsize=12)
    axes[1].set_xlabel('Kecepatan (m/s)')

    # Plot Speed (Magnitude)
    sns.histplot(df_stats['Current_Speed (Magnitude)'], bins=150, kde=True, ax=axes[2], color='purple')
    axes[2].set_title('Distribusi RAW Total Kecepatan (Speed)', fontsize=12)
    axes[2].set_xlabel('Kecepatan (m/s)')
    
    p99_speed = df_stats['Current_Speed (Magnitude)'].quantile(0.99)
    p999_speed = df_stats['Current_Speed (Magnitude)'].quantile(0.999)
    max_speed = df_stats['Current_Speed (Magnitude)'].max()
    
    axes[2].axvline(p99_speed, color='orange', linestyle='-', linewidth=2, label=f'P99 ({p99_speed:.2f} m/s)')
    axes[2].axvline(p999_speed, color='red', linestyle='--', linewidth=2, label=f'P99.9 ({p999_speed:.2f} m/s)')
    
    for ax in axes:
        ax.legend()
        # Jika nilai max raw data sangat ekstrem, batasi sumbu X agar lonceng utama tetap terlihat
        if max_speed > 5:
            ax.set_xlim(df_stats.iloc[:, list(axes).index(ax)].quantile(0.0005), 
                        df_stats.iloc[:, list(axes).index(ax)].quantile(0.9995))

    plt.tight_layout()
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "distribusi_raw_arus.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Grafik distribusi RAW tersimpan di: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_raw_physical_statistics()