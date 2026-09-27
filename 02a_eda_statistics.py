# =============================================================================
# FILE: 02a_eda_comparative_statistics.py
# DESKRIPSI: Modul 2 - Komparasi Distribusi Arus (RAW vs PRE-PROCESSED)
# =============================================================================

import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PATHS

# SESUAIKAN NAMA FILE OUTPUT FASE 1 LU DI SINI
PREPROCESSED_FILE = "01_Output/dataset_arus.nc" 

def generate_stats_and_plot(u_valid, v_valid, title_prefix, save_filename):
    """Fungsi helper untuk menghitung stats dan menggambar plot"""
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

    sns.histplot(df_stats['U_Velocity'], bins=150, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribusi Komponen U', fontsize=12)
    axes[0].set_xlabel('Kecepatan (m/s)')

    sns.histplot(df_stats['V_Velocity'], bins=150, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Distribusi Komponen V', fontsize=12)
    axes[1].set_xlabel('Kecepatan (m/s)')

    sns.histplot(df_stats['Current_Speed'], bins=150, kde=True, ax=axes[2], color='purple')
    axes[2].set_title('Distribusi Total Kecepatan (Speed)', fontsize=12)
    axes[2].set_xlabel('Kecepatan (m/s)')
    
    p999_speed = df_stats['Current_Speed'].quantile(0.999)
    max_speed = df_stats['Current_Speed'].max()
    
    axes[2].axvline(p999_speed, color='red', linestyle='--', linewidth=2, label=f'P99.9 ({p999_speed:.2f} m/s)')
    axes[2].legend()
    
    if max_speed > 5:
        for ax in axes:
            ax.set_xlim(df_stats.iloc[:, list(axes).index(ax)].quantile(0.0005), 
                        df_stats.iloc[:, list(axes).index(ax)].quantile(0.9995))

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    save_path = os.path.join("outputs", save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Grafik tersimpan di: {save_path}")
    plt.close()

def run_comparative_analysis():
    print("\n" + "="*60)
    print("🌊 MODUL 2: KOMPARASI RAW DATA VS PRE-PROCESSED")
    print("="*60)

    # =========================================================================
    # 1. ANALISIS RAW DATA
    # =========================================================================
    pattern = PATHS["hf_radar_pattern"]
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"❌ Tidak ada file RAW ditemukan dengan pattern: {pattern}")
        return

    datasets_list = []
    print(f"\n🔄 [1/2] Mengekstrak {len(all_files)} file RAW dari {pattern} ...")
    for f in all_files:
        if not os.path.exists(f): continue
        try:
            if os.path.getsize(f) / 1024 < 50:
                pass
            ds_temp = xr.open_dataset(f, engine='netcdf4')[['u', 'v']].load()
            datasets_list.append(ds_temp)
        except ValueError:
            try:
                ds_temp = xr.open_dataset(f, engine='scipy')[['u', 'v']].load()
                datasets_list.append(ds_temp)
            except: pass
        except: pass

    print("🧩 Menggabungkan dataset RAW...")
    ds_raw = xr.concat(datasets_list, dim='time').sortby('time')
    ds_raw = ds_raw.sel(time=~ds_raw.get_index('time').duplicated())
    
    u_raw = ds_raw['u'].values.flatten()
    v_raw = ds_raw['v'].values.flatten()
    valid_raw_idx = ~np.isnan(u_raw) & ~np.isnan(v_raw)
    
    generate_stats_and_plot(u_raw[valid_raw_idx], v_raw[valid_raw_idx], 
                            "RAW DATA (Sebelum Treatment)", "01_distribusi_raw.png")

    # =========================================================================
    # 2. ANALISIS PRE-PROCESSED DATA
    # =========================================================================
    print(f"\n🔄 [2/2] Membaca data PRE-PROCESSED dari {PREPROCESSED_FILE} ...")
    if not os.path.exists(PREPROCESSED_FILE):
        print(f"❌ File pre-processed tidak ditemukan. Sesuaikan path PREPROCESSED_FILE.")
        return
        
    ds_prep = xr.open_dataset(PREPROCESSED_FILE)
    
    # Deteksi nama variabel (terkadang 'u'/'v' atau 'u_anomali'/'v_anomali')
    u_col = 'u' if 'u' in ds_prep.data_vars else 'u_anomali'
    v_col = 'v' if 'v' in ds_prep.data_vars else 'v_anomali'
    
    u_prep = ds_prep[u_col].values.flatten()
    v_prep = ds_prep[v_col].values.flatten()
    valid_prep_idx = ~np.isnan(u_prep) & ~np.isnan(v_prep)
    
    generate_stats_and_plot(u_prep[valid_prep_idx], v_prep[valid_prep_idx], 
                            "PRE-PROCESSED DATA (Hasil Fase 1)", "02_distribusi_preprocessed.png")

if __name__ == "__main__":
    run_comparative_analysis()