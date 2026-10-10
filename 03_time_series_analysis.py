# =============================================================================
# FILE: 03_time_series_analysis.py
# DESKRIPSI: Modul 3 - Analisis Stasioneritas, ACF/PACF, & Penentuan Lookback Window
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import PATHS

def run_time_series_analysis():
    print("\n" + "="*60)
    print("📈 MODUL 3: ANALISIS TIME SERIES MARLIN (MEOF)")
    print("="*60)

    # KOREKSI: Baca file _meof.csv
    train_path = str(PATHS['pc_train']).replace('.csv', '_meof.csv')
    if not os.path.exists(train_path):
         print(f"❌ ERROR: File tidak ditemukan di {train_path}")
         return

    print(f"📂 Membaca file: {os.path.basename(train_path)}")
    df = pd.read_csv(train_path, parse_dates=['datetime'])
    
    longest_seg_id = df.groupby('Segment_ID').size().idxmax()
    df_seg = df[df['Segment_ID'] == longest_seg_id].reset_index(drop=True)
    
    print(f"🔍 Menggunakan Segmen ID: {longest_seg_id} (Panjang: {len(df_seg)} jam) untuk analisis.")
    
    # KOREKSI: Targetkan MEOF_PC1
    pc_target = df_seg['MEOF_PC1']
    
    # =========================================================================
    # 2. UJI STASIONERITAS (AUGMENTED DICKEY-FULLER TEST) SEMUA SEGMEN
    # =========================================================================
    print("\n🧪 [1/2] Melakukan Uji Stasioneritas (ADF Test) pada SEMUA Segmen...")
    
    unique_segments = df['Segment_ID'].unique()
    stasioner_count = 0
    non_stasioner_count = 0
    min_length = 100  # Abaikan segmen yang terlalu pendek (misal di bawah 100 jam)
    
    for seg_id in unique_segments:
        df_seg_loop = df[df['Segment_ID'] == seg_id].reset_index(drop=True)
        
        # Skip segmen yang terlalu pendek agar uji statistik valid
        if len(df_seg_loop) < min_length:
            continue
            
        pc_target_loop = df_seg_loop['MEOF_PC1']
        
        try:
            adf_result = adfuller(pc_target_loop, autolag='AIC')
            p_value = adf_result[1]
            
            if p_value < 0.05:
                stasioner_count += 1
            else:
                non_stasioner_count += 1
                
                # --- TAMBAHAN UNTUK MENGECEK TANGGAL ---
                # Pastikan formatnya terbaca sebagai datetime
                start_date = pd.to_datetime(df_seg_loop['datetime'].iloc[0]).strftime('%d %B %Y (%H:%M)')
                end_date = pd.to_datetime(df_seg_loop['datetime'].iloc[-1]).strftime('%d %B %Y (%H:%M)')
                
                print(f"   -> Segmen {seg_id:<3} (Panjang: {len(df_seg_loop):<4} jam) | P-Value: {p_value:.6f} ❌")
                print(f"      Waktu Kejadian: {start_date} s/d {end_date}")
                
        except Exception as e:
            print(f"   -> Segmen {seg_id} error saat uji ADF: {e}")

    print(f"\n📊 Ringkasan Uji ADF (Minimal panjang {min_length} jam):")
    print(f"   -> Stasioner       : {stasioner_count} segmen ✅")
    print(f"   -> Tidak Stasioner : {non_stasioner_count} segmen ❌")
    

    print("\n📊 [2/2] Menggambar Grafik ACF & PACF...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Analisis Autokorelasi Time Series - MEOF_PC1 (Segmen {longest_seg_id})', fontsize=16, fontweight='bold')
    
    max_lag = 48
    
    plot_acf(pc_target, lags=max_lag, ax=axes[0], color='blue', alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF) - Menunjukkan Memori Historis & Seasonality')
    axes[0].set_xlabel('Lag (Jam)')
    axes[0].set_ylabel('Korelasi')
    
    plot_pacf(pc_target, lags=max_lag, ax=axes[1], color='red', alpha=0.05, method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF) - Menunjukkan Direct Memori Tanpa Perantara')
    axes[1].set_xlabel('Lag (Jam)')
    axes[1].set_ylabel('Korelasi Parsial')
    
    plt.tight_layout()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "03_acf_pacf_analysis_meof.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"💾 Grafik tersimpan di: {save_path}")
    
    # =========================================================================
    # MENCARI SEGMEN STASIONER TERPANJANG UNTUK VISUALISASI
    # =========================================================================
    print("\n🔍 Mencari segmen stasioner terpanjang untuk visualisasi...")
    
    stationary_segments = []
    unique_segments = df['Segment_ID'].unique()
    
    for seg_id in unique_segments:
        df_seg_loop = df[df['Segment_ID'] == seg_id].reset_index(drop=True)
        if len(df_seg_loop) < 100: # Abaikan segmen pendek
            continue
            
        pc_target_loop = df_seg_loop['MEOF_PC1']
        try:
            p_value = adfuller(pc_target_loop, autolag='AIC')[1]
            if p_value < 0.05:
                stationary_segments.append((seg_id, len(df_seg_loop)))
        except:
            continue
            
    if not stationary_segments:
        print("❌ Tidak ada segmen stasioner yang memenuhi syarat!")
    else:
        # Urutkan berdasarkan panjang durasi (descending) dan ambil yang pertama
        longest_stat_seg_id = sorted(stationary_segments, key=lambda x: x[1], reverse=True)[0][0]
        df_stat_seg = df[df['Segment_ID'] == longest_stat_seg_id].reset_index(drop=True)
        
        print(f"✅ Segmen stasioner terpanjang ditemukan: Segmen {longest_stat_seg_id} (Panjang: {len(df_stat_seg)} jam)")

        # =========================================================================
        # VISUALISASI ROLLING STATISTICS (Untuk Cek Stasioneritas Visual)
        # =========================================================================
        print(f"📈 Menggambar Visualisasi Rolling Statistics untuk Segmen {longest_stat_seg_id}")
        
        df_plot = df_stat_seg.set_index('datetime')
        ts_target = df_plot['MEOF_PC1']
        
        window_size = 24
        rolmean = ts_target.rolling(window=window_size).mean()
        rolstd = ts_target.rolling(window=window_size).std()

        plt.figure(figsize=(15, 6))
        plt.plot(ts_target.index, ts_target.values, color='cornflowerblue', label='Data Asli (MEOF_PC1)', alpha=0.5)
        plt.plot(rolmean.index, rolmean.values, color='red', label=f'Rolling Mean ({window_size} Jam)', linewidth=2)
        plt.plot(rolstd.index, rolstd.values, color='black', label=f'Rolling Std ({window_size} Jam)', linewidth=2)

        plt.title(f'Uji Stasioneritas Visual: Rolling Mean & Standard Deviation\nSegmen ID: {longest_stat_seg_id} (Stasioner)', fontsize=14, fontweight='bold')
        plt.xlabel('Tanggal', fontsize=11)
        plt.ylabel('Nilai MEOF_PC1', fontsize=11)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        plt.xticks(rotation=45)
        
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        save_path_rolling = os.path.join(output_dir, f"03_rolling_stats_seg{longest_stat_seg_id}_stasioner.png")
        plt.savefig(save_path_rolling, dpi=300, bbox_inches='tight')
        print(f"💾 Grafik Rolling Stats tersimpan di: {save_path_rolling}")
    
    
    print("\n✅ Modul 3 Selesai. Silakan baca grafik untuk menentukan nilai Lookback Window (Input Sequence) model LSTM lu.")
    plt.show()

if __name__ == "__main__":
    run_time_series_analysis()