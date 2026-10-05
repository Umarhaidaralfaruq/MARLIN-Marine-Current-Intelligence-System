# =============================================================================
# FILE: 03_time_series_analysis.py
# DESKRIPSI: Modul 3 - Analisis Stasioneritas, ACF/PACF, & Penentuan Lookback Window
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import PATHS

def run_time_series_analysis():
    print("\n" + "="*60)
    print("📈 MODUL 3: ANALISIS TIME SERIES MARLIN")
    print("="*60)

    # 1. BUKA FILE PC TRAIN KOMPONEN U (atau V, cukup salah satu karena mewakili struktur waktu yang sama)
    train_path = str(PATHS['pc_train']).replace('.csv', '_u.csv')
    if not os.path.exists(train_path):
         print(f"❌ ERROR: File tidak ditemukan di {train_path}")
         return

    print(f"📂 Membaca file: {os.path.basename(train_path)}")
    df = pd.read_csv(train_path)
    
    # Ambil segmen pertama yang cukup panjang untuk dianalisis
    # Kita tidak menganalisis seluruh data agar grafik ACF tidak berantakan akibat gap segmen
    longest_seg_id = df.groupby('Segment_ID').size().idxmax()
    df_seg = df[df['Segment_ID'] == longest_seg_id].reset_index(drop=True)
    
    print(f"🔍 Menggunakan Segmen ID: {longest_seg_id} (Panjang: {len(df_seg)} jam) untuk analisis.")
    
    pc_target = df_seg['PC1']
    
    # =========================================================================
    # 2. UJI STASIONERITAS (AUGMENTED DICKEY-FULLER TEST)
    # =========================================================================
    print("\n🧪 [1/2] Melakukan Uji Stasioneritas (ADF Test)...")
    adf_result = adfuller(pc_target, autolag='AIC')
    p_value = adf_result[1]
    
    print(f"   -> ADF Statistic: {adf_result[0]:.4f}")
    print(f"   -> P-Value      : {p_value:.6f}")
    print("   -> Kesimpulan   : ", end="")
    if p_value < 0.05:
        print("Data SUDAH Stasioner (P < 0.05). Lolos Uji. ✅")
    else:
         print("Data TIDAK Stasioner (P >= 0.05). Berisiko Spurious Regression! ❌")

    # =========================================================================
    # 3. ANALISIS ACF & PACF
    # =========================================================================
    print("\n📊 [2/2] Menggambar Grafik ACF & PACF...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Analisis Autokorelasi Time Series - PC1 (Segmen {longest_seg_id})', fontsize=16, fontweight='bold')
    
    # Kita buat lag hingga 48 jam untuk melihat siklus diurnal (24h) dan memori panjang
    max_lag = 48
    
    # Plot ACF (Autocorrelation Function) -> Menunjukkan memori historis total dan seasonality
    plot_acf(pc_target, lags=max_lag, ax=axes[0], color='blue', alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF) - Menunjukkan Memori Historis & Seasonality')
    axes[0].set_xlabel('Lag (Jam)')
    axes[0].set_ylabel('Korelasi')
    
    # Plot PACF (Partial Autocorrelation Function) -> Menunjukkan memori langsung (direct effect)
    plot_pacf(pc_target, lags=max_lag, ax=axes[1], color='red', alpha=0.05, method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF) - Menunjukkan Direct Memori Tanpa Perantara')
    axes[1].set_xlabel('Lag (Jam)')
    axes[1].set_ylabel('Korelasi Parsial')
    
    plt.tight_layout()
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "03_acf_pacf_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"💾 Grafik tersimpan di: {save_path}")
    print("\n✅ Modul 3 Selesai. Silakan baca grafik untuk menentukan nilai Lookback Window (Input Sequence) model LSTM lu.")
    plt.show()

if __name__ == "__main__":
    run_time_series_analysis()