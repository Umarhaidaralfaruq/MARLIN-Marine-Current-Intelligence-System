# =============================================================================
# FILE: 05_reconstruction.py
# DESKRIPSI: Fase 5 - Inferensi, Inverse PCA, Rekonstruksi Peta, & Visualisasi
# =============================================================================

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf

from config import PATHS, MODEL_CONFIG, VISUAL_CONFIG

# Import modul FE yang udah kita buat
# Sesuaikan nama file importnya jika nama modul lu beda (misal: from 03_feature_engineering import ...)
from feature_engineering import build_features, TARGET_COLS

LOOKBACK = MODEL_CONFIG['input_window']
HORIZON = MODEL_CONFIG['output_window']
N_PC_FEATURES = MODEL_CONFIG['n_modes']

def load_artifacts(comp):
    """Fungsi untuk memuat semua senjata kita: Model, Scaler, dan PCA"""
    print(f"📦 Memuat artifak untuk komponen {comp.upper()}...")
    
    # Load PCA
    pca_path = str(PATHS['meof_pca_model']).replace('_model.joblib', f'_{comp}_model.joblib')
    pca = joblib.load(pca_path)
    
    # Load Scaler
    scaler_path = str(PATHS['scaler_path']).replace('.joblib', f'_{comp}.joblib')
    scaler = joblib.load(scaler_path)
    
    # Load LSTM
    model_path = str(PATHS['model_path']).replace('.h5', f'_{comp}.h5')
    model = tf.keras.models.load_model(model_path, compile=False) # compile=False karena cuma buat predict
    
    return pca, scaler, model

def prepare_test_data(comp):
    """Membangun ulang data Test dengan fitur-fitur FE"""
    test_path = str(PATHS['pc_test']).replace('.npy', f'_{comp}.csv')
    df_test = pd.read_csv(test_path, parse_dates=['datetime'])
    
    # Panggil Feature Engineering Otomatis
    df_test = build_features(df_test)
    
    # Deteksi kolom Input (seperti di skrip Training)
    all_generated_cols = [col for col in df_test.columns if col not in ['datetime', 'Segment_ID']]
    input_cols = [col for col in all_generated_cols if col not in TARGET_COLS]
    scale_cols = TARGET_COLS + input_cols
    
    return df_test, scale_cols, input_cols

def inverse_scale_dynamic(y_data, scaler_obj, total_scale_cols, n_target_cols):
    """Mengembalikan nilai dari -1..1 ke nilai PC asli"""
    n_samples, out_len, n_pc = y_data.shape
    dummy = np.zeros((n_samples * out_len, total_scale_cols))
    dummy[:, :n_target_cols] = y_data.reshape(-1, n_pc)
    unscaled = scaler_obj.inverse_transform(dummy)
    return unscaled[:, :n_target_cols].reshape(n_samples, out_len, n_target_cols)

def run_visual_test():
    print("\n" + "="*60)
    print("🗺️ FASE 5: REKONSTRUKSI SPASIAL & VISUAL TEST")
    print("="*60)
    
    # 1. LOAD RATA-RATA SPASIAL (Panduan Peta Spasial Langsung dari Sumbernya)
    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    ds_mean_v = xr.open_dataset(PATHS['spatial_mean_v'])

    # Ekstrak koordinat dan mask yang paling up-to-date!
    lats = ds_mean_u.lat.values
    lons = ds_mean_u.lon.values
    n_lat, n_lon = len(lats), len(lons)
    
    # Buat valid_mask otomatis (True jika nilai rata-ratanya bukan NaN)
    valid_mask = ds_mean_u['u'].notnull().values.flatten()

    # Dictionary penampung hasil prediksi PC
    pred_pc_dict = {}
    true_pc_dict = {}
    target_dates = []

    # 2. PREDIKSI UNTUK U DAN V
    for comp in ['u', 'v']:
        pca, scaler, model = load_artifacts(comp)
        df_test, scale_cols, input_cols = prepare_test_data(comp)
        
        # Scaling
        df_scaled = df_test.copy()
        df_scaled[scale_cols] = scaler.transform(df_scaled[scale_cols])
        
        # Bikin Array untuk 1 Sequence (Kita ambil sampel Segmen Valid pertama)
        valid_seg_ids = []
        for seg_id, group in df_scaled.groupby('Segment_ID'):
            if len(group) >= (LOOKBACK + HORIZON):
                valid_seg_ids.append(seg_id)
                
        if not valid_seg_ids:
            print(f"❌ FATAL: Pada komponen {comp.upper()}, tidak ada segmen Test yang sisa panjangnya mencapai {LOOKBACK + HORIZON} jam setelah Feature Engineering.")
            print("💡 Tips: Coba turunin LOOKBACK lu di config atau pastiin split rasio Test lu cukup besar.")
            continue # Lanjut ke komponen V kalau U gagal
            
        chosen_seg = valid_seg_ids[0] # Ambil segmen valid pertama
        print(f"   -> Menggunakan Segmen ID {chosen_seg} untuk sampel visualisasi (Sisa baris: {len(df_scaled[df_scaled['Segment_ID'] == chosen_seg])})")
        df_sample = df_scaled[df_scaled['Segment_ID'] == chosen_seg].reset_index(drop=True)
        
        # Kita ambil Jendela Waktu (Window) Pertama saja sebagai sampel visual
        X_sample = df_sample[input_cols].values[:LOOKBACK].reshape(1, LOOKBACK, -1)
        
        # Target PC yang sebenarnya untuk perbandingan
        Y_true_sample = df_sample[TARGET_COLS].values[LOOKBACK : LOOKBACK + HORIZON].reshape(1, HORIZON, N_PC_FEATURES)
        
        # Simpan waktu target untuk plotting
        if comp == 'u':
            target_dates = df_sample['datetime'].values[LOOKBACK : LOOKBACK + HORIZON]
        
        # Predict dengan Model
        print(f"🤖 Menjalankan prediksi LSTM untuk {comp.upper()}...")
        y_pred = model.predict(X_sample, verbose=0)
        
        # Inverse Scaling
        y_pred_unscaled = inverse_scale_dynamic(y_pred, scaler, len(scale_cols), N_PC_FEATURES)
        y_true_unscaled = inverse_scale_dynamic(Y_true_sample, scaler, len(scale_cols), N_PC_FEATURES)
        
        # Ekstrak dari batch dimension (hilangkan angka 1 di depan)
        pred_pc_dict[comp] = y_pred_unscaled[0]  # Shape: (24, N_MODES)
        true_pc_dict[comp] = y_true_unscaled[0]  # Shape: (24, N_MODES)
    
    # =========================================================================
    # 🕵️ ALAT INVESTIGASI: PLOT TIME-SERIES & RMSE (MODE 1 & 2)
    # =========================================================================
    print("\n📈 Menggambar grafik Time-Series perbandingan PC Asli vs Prediksi...")
    
    # Kita fokus nge-plot untuk komponen V (atau ganti 'u' kalau mau lihat U)
    comp_cek = 'u' 
    
    fig_ts, axes_ts = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig_ts.suptitle(f"Investigasi PC Mode 1 & 2 selama 24 Jam (Komponen {comp_cek.upper()})", fontsize=16, fontweight='bold')
    
    waktu_24_jam = range(1, HORIZON + 1) # Sumbu X: Jam 1 sampai 24
    
    for i in range(2): # Loop untuk Mode 1 (i=0) dan Mode 2 (i=1)
        garis_asli = true_pc_dict[comp_cek][:, i]
        garis_pred = pred_pc_dict[comp_cek][:, i]
        
        # Hitung RMSE khusus untuk 24 jam ini
        rmse_val = np.sqrt(np.mean((garis_asli - garis_pred)**2))
        
        axes_ts[i].plot(waktu_24_jam, garis_asli, label='Observasi Asli (True)', marker='o', color='black', linewidth=2)
        axes_ts[i].plot(waktu_24_jam, garis_pred, label='Tebakan AI (Predict)', marker='x', color='red', linewidth=2, linestyle='--')
        
        axes_ts[i].set_title(f"Pergerakan Mode {i+1} (RMSE: {rmse_val:.3f})", fontsize=12)
        axes_ts[i].set_ylabel("Nilai PC (Anomali)")
        axes_ts[i].legend()
        axes_ts[i].grid(True, linestyle=':', alpha=0.7)
        
    axes_ts[1].set_xlabel("Jam ke- (Masa Depan)", fontsize=12)
    plt.tight_layout()
    
    # Auto-Save Gambar Time Series
    output_dir = "outputs"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    waktu_awal_pred = pd.to_datetime(target_dates[0]).strftime('%Y%m%d_%H%M')
    ts_save_path = os.path.join(output_dir, f"timeseries_PC_{waktu_awal_pred}.png")
    
    print(f"💾 Menyimpan grafik Time-Series ke: {ts_save_path}")
    plt.savefig(ts_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # =========================================================================
    # 3. REKONSTRUKSI KE SPASIAL (PILIH JAM KE-BERAPA YANG MAU DIPLOT)
    # Mari kita plot tebakan masa depan jam ke-6 (Index 5)
    target_idx = VISUAL_CONFIG['plot_target_hour_idx'] 
    plot_time = pd.to_datetime(target_dates[target_idx])
    print(f"\n🎨 Merekam Peta Arus untuk target waktu: {plot_time}")
    
    # Ambil nilai rata-rata spasial pada jam tersebut
    # Kita pakai nearest fallback kalau jam pastinya tidak ada (walau harusnya ada)
    mean_u_map = ds_mean_u['u'].values
    mean_v_map = ds_mean_v['v'].values

    reconstructed_maps = {'True': {'u': None, 'v': None}, 'Pred': {'u': None, 'v': None}}

    for comp in ['u', 'v']:
        pca = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', f'_{comp}_model.joblib'))
        
        # Ambil PC pada jam ke-6
        pc_true = true_pc_dict[comp][target_idx].reshape(1, -1)
        pc_pred = pred_pc_dict[comp][target_idx].reshape(1, -1)
        
        # INVERSE PCA (Mengubah PC jadi Anomali Spasial 1D)
        anom_true_1d = pca.inverse_transform(pc_true)[0]
        anom_pred_1d = pca.inverse_transform(pc_pred)[0]
        
        # 🕵️ TAMBAHAN DETEKTIF: Print nilai PC-nya!
        if comp == 'v':  # Kita cek komponen V aja
            print(f"\n🔍 Cek Nilai Asli vs Prediksi (Mode 1 & 2):")
            print(f"   -> PC Asli : Mode1={pc_true[0,0]:.2f}, Mode2={pc_true[0,1]:.2f}")
            print(f"   -> PC Tebakan AI: Mode1={pc_pred[0,0]:.2f}, Mode2={pc_pred[0,1]:.2f}")
        
        # KEMBALIKAN KE PETA 2D (Unstacking)
        def rebuild_2d(anom_1d, mean_2d):
            base_map = np.full(n_lat * n_lon, np.nan)
            base_map[valid_mask] = anom_1d
            map_2d = base_map.reshape(n_lat, n_lon)
            return map_2d + mean_2d # Tambahkan rata-rata historis (Fase 1)
            
        reconstructed_maps['True'][comp] = rebuild_2d(anom_true_1d, mean_u_map if comp=='u' else mean_v_map)
        reconstructed_maps['Pred'][comp] = rebuild_2d(anom_pred_1d, mean_u_map if comp=='u' else mean_v_map)

    # 4. VISUALISASI QUIVER PLOT (TRUE vs PREDICTED)
    Lon, Lat = np.meshgrid(lons, lats)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.suptitle(f"Uji Visual Arus Selat Bali (Target: {plot_time})", fontsize=16, fontweight='bold')
    
    titles = ['Observasi Asli (True)', 'Tebakan AI (Predicted)']
    dict_keys = ['True', 'Pred']
    
    for ax, key, title in zip(axes, dict_keys, titles):
        U = reconstructed_maps[key]['u']
        V = reconstructed_maps[key]['v']
        
        # Hitung Kecepatan (Magnitude) untuk warna
        speed = np.sqrt(U**2 + V**2)
        
        # Plot quiver (skip setiap 2 grid agar panah tidak menumpuk)
        skip = (slice(None, None, 2), slice(None, None, 2))
        
        contour = ax.contourf(Lon, Lat, speed, cmap='jet', alpha=0.7, levels=np.linspace(0, 2, 20))
        ax.quiver(Lon[skip], Lat[skip], U[skip], V[skip], 
                  color='black', scale=15, width=0.003, headwidth=4)
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Longitude')
        if ax == axes[0]: ax.set_ylabel('Latitude')
        
    cbar = fig.colorbar(contour, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Kecepatan Arus (m/s)')
    
    plt.tight_layout()
    
    # 💾 TAMBAHAN: AUTO-SAVE GAMBAR HASIL PREDIKSI (KUALITAS JURNAL)
    # Membuat folder 'outputs' jika belum ada
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Format nama file berdasarkan waktu prediksi agar tidak menimpa file lama
    time_str = plot_time.strftime('%Y%m%d_%H%M')
    file_name = f"rekonstruksi_arus_{time_str}.png"
    save_path = os.path.join(output_dir, file_name)
    
    # Simpan dengan resolusi 300 DPI (Standar Jurnal Internasional Q1/Q2)
    print(f"💾 Menyimpan gambar visualisasi ke: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Tampilkan di layar
    plt.show()
    
    # =========================================================================
    # 🗺️ ALAT EVALUASI: PLOT SPATIAL RMSE (RATA-RATA 24 JAM)
    # =========================================================================
    print("\n🌐 Menghitung dan menggambar Peta Spatial RMSE selama 24 Jam...")
    
    # 1. Load Ulang PCA U dan V (Pastikan tidak tertimpa)
    pca_u = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_u_model.joblib'))
    pca_v = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_v_model.joblib'))
    
    # 2. INVERSE TRANSFORM 24 JAM SEKALIGUS (Cepat & Efisien)
    # Hasil array 1D Anomali dengan shape: (24 jam, jumlah_titik_grid_valid)
    anom_u_true_24 = pca_u.inverse_transform(true_pc_dict['u'])
    anom_u_pred_24 = pca_u.inverse_transform(pred_pc_dict['u'])
    
    anom_v_true_24 = pca_v.inverse_transform(true_pc_dict['v'])
    anom_v_pred_24 = pca_v.inverse_transform(pred_pc_dict['v'])
    
    # 3. HITUNG ERROR KUADRAT (VECTOR ERROR)
    # Catatan Fisika: Selisih (Prediksi - Asli) sama dengan selisih anomalinya, 
    # karena Background Mean (Arus Dasar) saling meniadakan.
    sq_err_u = (anom_u_pred_24 - anom_u_true_24)**2
    sq_err_v = (anom_v_pred_24 - anom_v_true_24)**2
    
    # 4. HITUNG SPATIAL RMSE (Akar dari Rata-rata 24 jam di tiap titik)
    # axis=0 merata-ratakan 24 baris (waktu), menyisakan 1 baris spasial array
    rmse_1d = np.sqrt(np.mean(sq_err_u + sq_err_v, axis=0))
    
    # 5. KEMBALIKAN KE BENTUK PETA 2D
    rmse_map = np.full(n_lat * n_lon, np.nan)
    rmse_map[valid_mask] = rmse_1d
    rmse_map_2d = rmse_map.reshape(n_lat, n_lon)
    
    # 6. PLOTTING PETA RMSE
    fig_rmse, ax_rmse = plt.subplots(figsize=(9, 7))
    
    # Perbaikan format waktu agar lebih rapi (tidak ada T dan 000000000)
    target_time_str = pd.to_datetime(target_dates[0]).strftime('%Y-%m-%d %H:%M')
    fig_rmse.suptitle(f"Evaluasi Spasial: Peta Vector RMSE (24 Jam Prediksi)\nTarget Awal: {target_time_str}", fontsize=14, fontweight='bold')
    
    # Pakai colormap 'hot_r' (Putih/Kuning = Error Kecil (Bagus), Merah/Hitam = Error Besar)
    contour_rmse = ax_rmse.contourf(Lon, Lat, rmse_map_2d, cmap='hot_r', levels=np.linspace(0, np.nanmax(rmse_map_2d), 20))
    
    #  🎯 TAMBAHAN: Garis Penanda Batas Error 0.4 m/s
    batas_04 = ax_rmse.contour(Lon, Lat, rmse_map_2d, levels=[0.4], colors='blue', linewidths=2, linestyles='dashed')
    
    # Simpan hasil clabel ke dalam variabel 'teks_label' (tanpa fontweight)
    teks_label = ax_rmse.clabel(batas_04, inline=True, fontsize=12, fmt='Batas 0.4', colors='blue')
    
    # Menambahkan label teks "0.4" langsung menempel di garisnya
    ax_rmse.clabel(batas_04, inline=True, fontsize=12, fmt='Batas 0.4 m/s', colors='blue')
    
    ax_rmse.set_xlabel('Longitude')
    ax_rmse.set_ylabel('Latitude')
    
    cbar_rmse = fig_rmse.colorbar(contour_rmse, ax=ax_rmse, orientation='vertical', fraction=0.05, pad=0.05)
    cbar_rmse.set_label('Vector RMSE Error (m/s)')
    
    # 7. AUTO-SAVE
    rmse_save_path = os.path.join(output_dir, f"spatial_rmse_{waktu_awal_pred}.png")
    print(f"💾 Menyimpan Peta Spatial RMSE ke: {rmse_save_path}")
    plt.savefig(rmse_save_path, dpi=300, bbox_inches='tight')
    plt.show()

# =========================================================================
# 🚀 FITUR BARU: SIMULASI METRIK KOMPREHENSIF (RMSE, MAE, ARAH)
# =========================================================================
def run_period_simulation(start_date, end_date):
    print("\n" + "="*70)
    print(f"🌪️ MENJALANKAN SIMULASI KOMPREHENSIF: {start_date} s/d {end_date}")
    print("="*70)
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Ekstrak informasi spasial yang paling up-to-date langsung dari NetCDF
    ds_mean_u_info = xr.open_dataset(PATHS['spatial_mean_u'])
    lats = ds_mean_u_info.lat.values
    lons = ds_mean_u_info.lon.values
    n_lat, n_lon = len(lats), len(lons)
    valid_mask = ds_mean_u_info['u'].notnull().values.flatten()

    # Dictionary penampung tebakan anomali absolut
    pred_pc_dict = {}
    true_pc_dict = {}

    for comp in ['u', 'v']:
        pca, scaler, model = load_artifacts(comp)
        df_test, scale_cols, input_cols = prepare_test_data(comp)
        
        df_scaled = df_test.copy()
        df_scaled[scale_cols] = scaler.transform(df_scaled[scale_cols])
        
        X_list, Y_true_list = [], []
        
        # ⚡ OPTIMASI: Vektorisasi pencarian jendela waktu (Jauh lebih cepat dari iterasi baris Pandas)
        for seg_id, group in df_scaled.groupby('Segment_ID'):
            group = group.reset_index(drop=True)
            n_rows = len(group)
            if n_rows < LOOKBACK + HORIZON: continue
            
            # Ekstrak data sebagai NumPy untuk kecepatan penuh
            X_data = group[input_cols].values
            Y_data = group[TARGET_COLS].values
            times = group['datetime'].values
            
            # Ambil semua waktu mulai dari tiap jendela (window start time)
            target_start_times = pd.to_datetime(times[LOOKBACK : n_rows - HORIZON + 1])
            
            # Buat filter array (mask)
            valid_mask_times = (target_start_times >= start_dt) & (target_start_times <= end_dt)
            valid_indices = np.where(valid_mask_times)[0]
            
            for i in valid_indices:
                X_list.append(X_data[i : i+LOOKBACK])
                Y_true_list.append(Y_data[i+LOOKBACK : i+LOOKBACK+HORIZON])
                    
        if not X_list:
            print(f"❌ Tidak ada data Test komponen {comp.upper()} di rentang tanggal tersebut!")
            return
            
        X_batch = np.array(X_list)
        Y_true_batch = np.array(Y_true_list)
        print(f"🤖 Memprediksi {len(X_batch)} skenario (x 24 jam) untuk komponen {comp.upper()}...")
        
        y_pred_batch = model.predict(X_batch, verbose=0)
        
        pred_pc_dict[comp] = inverse_scale_dynamic(y_pred_batch, scaler, len(scale_cols), N_PC_FEATURES)
        true_pc_dict[comp] = inverse_scale_dynamic(Y_true_batch, scaler, len(scale_cols), N_PC_FEATURES)

    print("\n🌐 Merekonsiliasi ke Peta Spasial dan Menghitung Fisika Kecepatan & Arah...")
    
    pca_u = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_u_model.joblib'))
    pca_v = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_v_model.joblib'))
    
    n_samples = pred_pc_dict['u'].shape[0]
    
    # ==========================================================
    # SIMPAN DATA 3D UNTUK ANALISIS PER HORIZON
    # Shape = (sample, horizon, pc)
    # ==========================================================

    pred_u_3d = pred_pc_dict['u']
    true_u_3d = true_pc_dict['u']

    pred_v_3d = pred_pc_dict['v']
    true_v_3d = true_pc_dict['v']
    
    # Flatten jam dan skenario -> Bentuk: (Total_Jam_Prediksi, Jumlah_Mode)
    pc_u_pred_flat = pred_pc_dict['u'].reshape(-1, N_PC_FEATURES)
    pc_u_true_flat = true_pc_dict['u'].reshape(-1, N_PC_FEATURES)
    pc_v_pred_flat = pred_pc_dict['v'].reshape(-1, N_PC_FEATURES)
    pc_v_true_flat = true_pc_dict['v'].reshape(-1, N_PC_FEATURES)
    
    # Transform ke Anomali Spasial 1D
    anom_u_pred = pca_u.inverse_transform(pc_u_pred_flat)
    anom_u_true = pca_u.inverse_transform(pc_u_true_flat)
    anom_v_pred = pca_v.inverse_transform(pc_v_pred_flat)
    anom_v_true = pca_v.inverse_transform(pc_v_true_flat)

    # =====================================================================
    # 🌊 KONVERSI KECEPATAN (SPEED) DAN ARAH (DIRECTION)
    # =====================================================================
    # Karena kita menghitung Metrik absolut (Arah), anomali saja tidak cukup.
    # Kita harus tambahkan Mean Historis ke seluruh grid prediksi dan asli
    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    ds_mean_v = xr.open_dataset(PATHS['spatial_mean_v'])
    
    # Ambil grid valid 1D dari mean absolut
    mean_u_1d = ds_mean_u['u'].values.flatten()[valid_mask]
    mean_v_1d = ds_mean_v['v'].values.flatten()[valid_mask]
    
    # Arus Riil = Anomali + Mean Historis
    u_pred_real = anom_u_pred + mean_u_1d
    v_pred_real = anom_v_pred + mean_v_1d
    u_true_real = anom_u_true + mean_u_1d
    v_true_real = anom_v_true + mean_v_1d

    # Hitung Kecepatan (m/s)
    speed_pred = np.sqrt(u_pred_real**2 + v_pred_real**2)
    speed_true = np.sqrt(u_true_real**2 + v_true_real**2)
    
    # Hitung Arah Arus Oseanografi (Derajat, 0-360)
    dir_pred = np.degrees(np.arctan2(u_pred_real, v_pred_real)) % 360
    dir_true = np.degrees(np.arctan2(u_true_real, v_true_real)) % 360
    
    # ==========================================================
    # ANALISIS ERROR PER HORIZON
    # ==========================================================

    print("\n📊 EVALUASI PER HORIZON (LSTM vs PERSISTENCE)")
    print("-" * 75)

    speed_pred_3d = speed_pred.reshape(n_samples, HORIZON, -1)
    speed_true_3d = speed_true.reshape(n_samples, HORIZON, -1)
    
    # Baseline Persistence: Kondisi jam ke-0 diulang untuk seluruh horizon
    persistence_speed = np.repeat(speed_true_3d[:, 0:1, :], HORIZON, axis=1)

    rmse_lstm_all = []
    rmse_persist_all = []

    # 🧹 REFACTORING: Penggabungan loop evaluasi per horizon
    for h in range(HORIZON):
        # Error LSTM
        diff_lstm = speed_pred_3d[:, h, :] - speed_true_3d[:, h, :]
        rmse_lstm = np.sqrt(np.mean(diff_lstm ** 2))
        mae_lstm = np.mean(np.abs(diff_lstm))

        # Error Persistence
        diff_persist = persistence_speed[:, h, :] - speed_true_3d[:, h, :]
        rmse_persist = np.sqrt(np.mean(diff_persist ** 2))

        # Skill Score
        skill_score = 1 - (rmse_lstm / max(rmse_persist, 1e-8))

        rmse_lstm_all.append(rmse_lstm)
        rmse_persist_all.append(rmse_persist)

        print(
            f"H+{h+1:02d} | "
            f"RMSE={rmse_lstm:.4f} | "
            f"MAE={mae_lstm:.4f} | "
            f"Persist={rmse_persist:.4f} | "
            f"Skill={skill_score:.3f}"
        )

    print("\n🏆 RINGKASAN SKILL SCORE")
    print("-" * 70)

    rmse_lstm_mean = np.mean(rmse_lstm_all)
    rmse_persist_mean = np.mean(rmse_persist_all)
    skill_mean = 1 - (rmse_lstm_mean / rmse_persist_mean)

    print(f"RMSE LSTM Mean       : {rmse_lstm_mean:.4f} m/s")
    print(f"RMSE Persistence Mean: {rmse_persist_mean:.4f} m/s")
    print(f"Skill Score Mean     : {skill_mean:.3f}")
    
    # =====================================================================
    # 📈 FITUR BARU: PLOTTING GRAFIK EVALUASI PER HORIZON
    # =====================================================================
    print("\n📈 Menggambar grafik Evaluasi per Horizon (RMSE & Skill Score)...")
    
    # Siapkan data X (Jam ke-1 sampai 24) dan Y (Skill Score tiap jam)
    hours = np.arange(1, HORIZON + 1)
    # Gunakan trik max() agar tidak terjadi error pembagian dengan nol seperti sebelumnya
    skill_scores_all = 1 - (np.array(rmse_lstm_all) / np.maximum(np.array(rmse_persist_all), 1e-8))
    
    fig_hz, axes_hz = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig_hz.suptitle(f"Performa Model per Horizon Prediksi (1 - {HORIZON} Jam)", fontsize=16, fontweight='bold')
    
    # 📊 Subplot 1: RMSE LSTM vs Persistence
    axes_hz[0].plot(hours, rmse_lstm_all, marker='o', color='blue', label='LSTM (AI)', linewidth=2.5)
    axes_hz[0].plot(hours, rmse_persist_all, marker='x', color='gray', linestyle='--', label='Persistence (Baseline)', linewidth=2)
    axes_hz[0].set_ylabel("RMSE Kecepatan (m/s)", fontsize=12)
    axes_hz[0].set_title("Perbandingan Error (Semakin rendah semakin baik)", fontsize=13)
    axes_hz[0].grid(True, linestyle=':', alpha=0.7)
    axes_hz[0].legend(fontsize=11)
    
    # 📊 Subplot 2: Skill Score
    axes_hz[1].plot(hours, skill_scores_all, marker='s', color='green', label='Skill Score', linewidth=2.5)
    axes_hz[1].axhline(y=0, color='red', linestyle='-', linewidth=1.5, label='Batas 0 (AI = Baseline)') 
    axes_hz[1].set_ylabel("Skill Score", fontsize=12)
    axes_hz[1].set_title("Skill Score Oseanografi (Semakin mendekati 1 semakin baik)", fontsize=13)
    axes_hz[1].set_xlabel("Horizon Prediksi Masa Depan (Jam ke-)", fontsize=12)
    axes_hz[1].set_xticks(hours) # Paksa agar sumbu X menampilkan semua angka jam (1-24)
    axes_hz[1].set_ylim(-0.5, 1.0)
    axes_hz[1].grid(True, linestyle=':', alpha=0.7)
    axes_hz[1].legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save output gambar horizon
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    hz_save_path = os.path.join(output_dir, f"grafik_horizon_{start_dt.strftime('%Y%m%d')}.png")
    
    print(f"💾 Menyimpan Grafik Horizon ke: {hz_save_path}")
    plt.savefig(hz_save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================================
    # 📏 PERHITUNGAN METRIK SPASIAL (Rata-rata di setiap titik laut)
    # =====================================================================
    # 1. MAE Kecepatan (m/s) -> Error mutlak
    mae_speed_1d = np.mean(np.abs(speed_pred - speed_true), axis=0)
    
    # 2. RMSE Kecepatan (m/s) -> Error kuadrat rata-rata
    rmse_speed_1d = np.sqrt(np.mean((speed_pred - speed_true)**2, axis=0))
    
    # 3. RMSE Arah (Derajat)
    # PENTING: Error arah adalah sirkular (359 derajat dan 1 derajat selisihnya cuma 2 derajat, bukan 358)
    def circular_error(pred, true):
        diff = np.abs(pred - true) % 360
        # Jika selisih lebih dari 180, ambil jarak terdekat lewat sisi sebaliknya
        diff = np.where(diff > 180, 360 - diff, diff)
        return diff
        
    dir_error = circular_error(dir_pred, dir_true)
    rmse_dir_1d = np.sqrt(np.mean(dir_error**2, axis=0))
    
    print("\n🌐 RINGKASAN ERROR SPASIAL")
    print("-" * 60)
    print(f"MAE Speed Mean    : {np.nanmean(mae_speed_1d):.4f} m/s")
    print(f"MAE Speed Median  : {np.nanmedian(mae_speed_1d):.4f} m/s")
    print(f"RMSE Speed Mean   : {np.nanmean(rmse_speed_1d):.4f} m/s")
    print(f"RMSE Speed Median : {np.nanmedian(rmse_speed_1d):.4f} m/s")
    print(f"RMSE Direction    : {np.nanmean(rmse_dir_1d):.2f}°")
    print(f"Best Grid RMSE    : {np.nanmin(rmse_speed_1d):.4f} m/s")
    print(f"Worst Grid RMSE   : {np.nanmax(rmse_speed_1d):.4f} m/s")
    print(f"P90 Grid RMSE     : {np.nanpercentile(rmse_speed_1d, 90):.4f} m/s")

    # =====================================================================
    # 🗺️ MENGEMBALIKAN KE BENTUK PETA 2D
    # =====================================================================
    def to_2d_map(data_1d):
        base_map = np.full(n_lat * n_lon, np.nan)
        base_map[valid_mask] = data_1d
        return base_map.reshape(n_lat, n_lon)
        
    mae_speed_map = to_2d_map(mae_speed_1d)
    rmse_speed_map = to_2d_map(rmse_speed_1d)
    rmse_dir_map = to_2d_map(rmse_dir_1d)

    # =====================================================================
    # 🎨 PLOTTING MULTIPLE MAPS
    # =====================================================================
    Lon, Lat = np.meshgrid(lons, lats)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(f"Evaluasi Model Spasial Menyeluruh\nPeriode: {start_date} s/d {end_date} | Total Skenario: {n_samples * 24} Jam", fontsize=16, fontweight='bold', y=1.05)

    max_error_speed = max(np.nanmax(mae_speed_map), np.nanmax(rmse_speed_map))
    
    fixed_max_mae = VISUAL_CONFIG['max_val_mae_speed']  
    fixed_max_rmse = VISUAL_CONFIG['max_val_rmse_speed']
    
    maps_to_plot = [
        {'data': mae_speed_map, 'title': 'MAE Kecepatan (m/s)', 'cmap': 'hot_r', 'max_val': fixed_max_mae},
        {'data': rmse_speed_map, 'title': 'RMSE Kecepatan (m/s)', 'cmap': 'hot_r', 'max_val': fixed_max_rmse},
        {'data': rmse_dir_map, 'title': 'RMSE Arah Arus (Derajat)', 'cmap': 'plasma_r', 'max_val': VISUAL_CONFIG['max_val_rmse_dir']} 
    ]

    for ax, cfg in zip(axes, maps_to_plot):
        contour = ax.contourf(Lon, Lat, cfg['data'], cmap=cfg['cmap'], levels=np.linspace(0, cfg['max_val'], 20))
        
        # 🎯 PENGATURAN THRESHOLD DINAMIS DARI CONFIG
        if 'MAE' in cfg['title']:
            thresh = VISUAL_CONFIG['threshold_mae_speed']
            batas = ax.contour(Lon, Lat, cfg['data'], levels=[thresh], colors='blue', linewidths=1.5, linestyles='dashed')
            ax.clabel(batas, inline=True, fontsize=10, fmt=f'Batas {thresh}', colors='blue')
            
        elif 'RMSE Kecepatan' in cfg['title']:
            thresh = VISUAL_CONFIG['threshold_rmse_speed']
            batas = ax.contour(Lon, Lat, cfg['data'], levels=[thresh], colors='blue', linewidths=1.5, linestyles='dashed')
            ax.clabel(batas, inline=True, fontsize=10, fmt=f'Batas {thresh}', colors='blue')
            
        elif 'RMSE Arah' in cfg['title']:
            thresh = VISUAL_CONFIG['threshold_rmse_dir']
            batas = ax.contour(Lon, Lat, cfg['data'], levels=[thresh], colors='cyan', linewidths=1.5, linestyles='dashed')
            ax.clabel(batas, inline=True, fontsize=10, fmt=f'Batas {thresh}°', colors='cyan')

        ax.set_title(cfg['title'], fontsize=14)
        ax.set_xlabel('Longitude')
        if ax == axes[0]: ax.set_ylabel('Latitude')
        
        # ==========================================================
        # 🎯 PERBAIKAN: COLORBAR YANG LEBIH RAPI & RENGGANG
        # ==========================================================
        cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', fraction=0.046, pad=0.1)
        cbar.set_label(cfg['title'].split(' ')[0] + f" ({cfg['title'].split('(')[-1]}") # Label lebih jelas
        
        # Bikin penggaris angka: Cuma ambil 6 titik dari 0 sampai nilai max
        batas_ticks = np.linspace(0, cfg['max_val'], 6)
        cbar.set_ticks(batas_ticks)
        
        # Bedakan format angka (Kecepatan = 2 desimal, Arah = Bilangan bulat)
        if 'Arah' in cfg['title']:
            cbar.ax.set_xticklabels([f"{int(t)}°" for t in batas_ticks])
        else:
            cbar.ax.set_xticklabels([f"{t:.2f}" for t in batas_ticks])
        # ==========================================================

    plt.tight_layout()
    
    # Save output
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"simulasi_komprehensif_{start_dt.strftime('%Y%m%d')}_sd_{end_dt.strftime('%Y%m%d')}.png")
    
    print(f"💾 Menyimpan Peta Simulasi Komprehensif ke: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# =========================================================================
# CARA MENJALANKANNYA
# =========================================================================
if __name__ == "__main__":
    # 1. Kalau mau cek satu peta spesifik (buka tutup komennya)
    #run_visual_test() 
    
    # 2. AUTO-DETECT TANGGAL TEST UNTUK SIMULASI MUSIMAN
    test_path = str(PATHS['pc_test']).replace('.npy', '_u.csv')
    df_cek = pd.read_csv(test_path)
    
    # Ambil batas tanggal riil yang ada di file test
    tgl_awal = df_cek['datetime'].min()
    tgl_akhir = df_cek['datetime'].max()
    
    print(f"\n📅 Info: Data Test lu tersedia dari {tgl_awal} sampai {tgl_akhir}")
    
    # Jalankan simulasi secara otomatis pakai batas tanggal tersebut
    # (Kalau mau dipersingkat, bisa diganti manual misal start_date="2026-02-05", dst)
    run_period_simulation(start_date=tgl_awal, end_date=tgl_akhir)

