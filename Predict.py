# =============================================================================
# FILE: operational_prediction.py
# DESKRIPSI: Fase Production - Prediksi Otomatis Harian Arus Selat Bali
#            Output 1: File CSV (Data Tabel)
#            Output 2: Peta Visual Spasial (4-Panel Snapshot Harian)
# =============================================================================

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt

from config import PATHS, MODEL_CONFIG
from feature_engineering import build_features, TARGET_COLS

LOOKBACK = MODEL_CONFIG['input_window']
HORIZON = MODEL_CONFIG['output_window']
N_PC_FEATURES = MODEL_CONFIG['n_modes']

def load_production_artifacts(comp):
    pca_path = str(PATHS['meof_pca_model']).replace('_model.joblib', f'_{comp}_model.joblib')
    pca = joblib.load(pca_path)
    scaler_path = str(PATHS['scaler_path']).replace('.joblib', f'_{comp}.joblib')
    scaler = joblib.load(scaler_path)
    model_path = str(PATHS['model_path']).replace('.h5', f'_{comp}.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    return pca, scaler, model

def get_latest_data_for_comp(comp):
    print(f"📂 Membaca data historis terbaru untuk komponen {comp.upper()}...")
    df_full = pd.read_csv(PATHS['data_gabung']).dropna(subset=['datetime'])
    df_full['datetime'] = pd.to_datetime(df_full['datetime'])
    df_full = df_full.sort_values('datetime').reset_index(drop=True)
    
    buffer_rows = 24 * 7 
    df_buffer = df_full.iloc[-buffer_rows:].copy().reset_index(drop=True) if len(df_full) >= buffer_rows else df_full.copy()
    
    pc_cols_asal = [f"{comp.upper()}_PC{i+1}" for i in range(N_PC_FEATURES)]
    pc_cols_tujuan = [f"PC{i+1}" for i in range(N_PC_FEATURES)]
    
    cols_to_keep = ['datetime', 'Segment_ID', 'waterlevel'] + pc_cols_asal
    df_comp = df_buffer[cols_to_keep].copy()
    
    rename_dict = dict(zip(pc_cols_asal, pc_cols_tujuan))
    df_comp = df_comp.rename(columns=rename_dict)
    return df_comp

def inverse_scale_operational(y_pred_batch, scaler_obj, total_cols, n_pcs):
    out_len = y_pred_batch.shape[1]
    dummy = np.zeros((out_len, total_cols))
    dummy[:, :n_pcs] = y_pred_batch[0] 
    unscaled = scaler_obj.inverse_transform(dummy)
    return unscaled[:, :n_pcs] 

def main_operational_forecast():
    print("\n" + "="*60)
    print("🚀 RUNNING AUTOMATED OPERATIONAL FORECAST (SELAT BALI)")
    print("="*60)
    
    # 1. LOAD METADATA SPASIAL
    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    ds_mean_v = xr.open_dataset(PATHS['spatial_mean_v'])
    
    lats = ds_mean_u.lat.values
    lons = ds_mean_u.lon.values
    n_lat, n_lon = len(lats), len(lons)
    
    # Valid mask yang 100% akurat sesuai Fase 1
    valid_mask = ds_mean_u['u'].notnull().values.flatten()
    
    mean_u_map = ds_mean_u['u'].values.flatten()[valid_mask]
    mean_v_map = ds_mean_v['v'].values.flatten()[valid_mask]
    
    Lon, Lat = np.meshgrid(lons, lats)
    lon_valid = Lon.flatten()[valid_mask]
    lat_valid = Lat.flatten()[valid_mask]
    n_valid_points = len(lon_valid)

    pred_pc_outputs = {}
    latest_timestamp = None

    # 2. RUN PREDIKSI UNTUK MASING-MASING KOMPONEN
    for comp in ['u', 'v']:
        pca, scaler, model = load_production_artifacts(comp)
        df_buffer = get_latest_data_for_comp(comp)
        latest_timestamp = df_buffer['datetime'].max()
        
        df_proc = build_features(df_buffer)
        all_gen_cols = [c for c in df_proc.columns if c not in ['datetime', 'Segment_ID']]
        input_cols = [c for c in all_gen_cols if c not in TARGET_COLS]
        scale_cols = TARGET_COLS + input_cols
        
        df_input_window = df_proc.iloc[-LOOKBACK:].copy().reset_index(drop=True)
        if df_input_window[scale_cols].isna().any().any():
            df_input_window[scale_cols] = df_input_window[scale_cols].interpolate(method='linear', limit_direction='both')
            
        df_input_window[scale_cols] = scaler.transform(df_input_window[scale_cols])
        X_prod = df_input_window[input_cols].values.reshape(1, LOOKBACK, -1)
        
        print(f"🤖 LSTM memprediksi arus masa depan untuk {comp.upper()}...")
        y_pred = model.predict(X_prod, verbose=0)
        pred_pc_outputs[comp] = inverse_scale_operational(y_pred, scaler, len(scale_cols), N_PC_FEATURES)

    # 3. REKONSTRUKSI SPASIAL & KONVERSI FISIKA
    print(f"\n🌐 Data terakhir tercatat: {latest_timestamp}")
    print("🌐 Merekonsiliasi PC menjadi Peta 2D dan menghitung Kecepatan & Arah...")
    
    future_timestamps = pd.date_range(start=latest_timestamp + pd.Timedelta(hours=1), periods=HORIZON, freq='1H')
    all_forecast_rows = []

    pca_u = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_u_model.joblib'))
    pca_v = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_v_model.joblib'))

    # 🎨 TEMPAT PENAMPUNGAN UNTUK PLOTTING (Jam ke- 6, 12, 18, 24)
    maps_for_plotting = []

    for h_idx in range(HORIZON):
        current_pred_time = future_timestamps[h_idx]
        
        pc_u_hour = pred_pc_outputs['u'][h_idx].reshape(1, -1)
        pc_v_hour = pred_pc_outputs['v'][h_idx].reshape(1, -1)
        
        anom_u_1d = pca_u.inverse_transform(pc_u_hour)[0]
        anom_v_1d = pca_v.inverse_transform(pc_v_hour)[0]
        
        u_absolute = anom_u_1d + mean_u_map
        v_absolute = anom_v_1d + mean_v_map
        
        speed = np.sqrt(u_absolute**2 + v_absolute**2)
        direction = np.degrees(np.arctan2(u_absolute, v_absolute)) % 360
        
        # Simpan ke memori visualisasi jika jam ini masuk radar plot
        if h_idx in range(HORIZON):
            def to_2d(data_1d):
                        base = np.full(n_lat * n_lon, np.nan)
                        base[valid_mask] = data_1d
                        return base.reshape(n_lat, n_lon)
            
            maps_for_plotting.append({
                'time': current_pred_time,
                'u_2d': to_2d(u_absolute),
                'v_2d': to_2d(v_absolute),
                'speed_2d': to_2d(speed)
            })
        
        for i in range(n_valid_points):
            all_forecast_rows.append({
                'datetime_prediction': current_pred_time,
                'forecast_horizon_hour': h_idx + 1,
                'latitude': lat_valid[i],
                'longitude': lon_valid[i],
                'u_velocity_ms': round(u_absolute[i], 4),
                'v_velocity_ms': round(v_absolute[i], 4),
                'current_speed_ms': round(speed[i], 4),
                'current_direction_deg': round(direction[i], 1)
            })

    # 4. EXPORT HASIL KE CSV
    df_final_forecast = pd.DataFrame(all_forecast_rows)
    output_dir = "outputs/predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    file_timestamp = future_timestamps[0].strftime('%Y%m%d_%H%M')
    csv_save_path = os.path.join(output_dir, f"prediksi_arus_{file_timestamp}.csv")
    df_final_forecast.to_csv(csv_save_path, index=False)
    print(f"\n💾 CSV Ramalan Laut tersimpan di:\n   -> {csv_save_path}")

    # =====================================================================
    # 5. AUTO-PLOTTING SPASIAL (24 PANEL & UTC -> WIB)
    # =====================================================================
    print("🎨 Menggambar Peta Prakiraan Cuaca Harian (24-Panel)...")
    
    # Ubah Grid menjadi 4 Baris x 6 Kolom, perbesar ukuran kanvas (24x16 inch)
    fig, axes = plt.subplots(4, 6, figsize=(26, 18), sharex=True, sharey=True)
    
    # Konversi waktu basis (latest_timestamp) dari UTC ke WIB untuk Judul
    waktu_awal_wib = latest_timestamp + pd.Timedelta(hours=7)
    fig.suptitle(f"Prakiraan Arus Permukaan Selat Bali (24 Jam ke Depan)\nBasis Waktu Awal: {waktu_awal_wib.strftime('%Y-%m-%d %H:%M WIB')}", fontsize=22, fontweight='bold', y=0.95)

    axes = axes.flatten()
    
    # Skala skip panah agar rapi (Mungkin perlu dilebarkan karena panelnya mengecil)
    skip = (slice(None, None, 1), slice(None, None, 1))

    for ax, map_data in zip(axes, maps_for_plotting):
        U = map_data['u_2d']
        V = map_data['v_2d']
        Speed = map_data['speed_2d']
        
        # ⌚ KONVERSI UTC KE WIB UNTUK LABEL TIAP PANEL (+ 7 Jam)
        waktu_wib_panel = map_data['time'] + pd.Timedelta(hours=7)
        plot_time_str = waktu_wib_panel.strftime('%H:%M WIB\n(%d %b)')
        
        # Plot Heatmap Kecepatan
        contour = ax.contourf(Lon, Lat, Speed, cmap='jet', alpha=0.7, levels=np.linspace(0, 3.0, 20))
        
        # Plot Quiver (Arah Panah Arus)
        ax.quiver(Lon[skip], Lat[skip], U[skip], V[skip], 
                  color='black', scale=20, width=0.003, headwidth=2)
        
        ax.set_title(f"{plot_time_str}", fontsize=12, fontweight='bold')
        
        # Hilangkan label X dan Y di semua panel kecuali yang paling pinggir agar tidak sumpek
        if ax in axes[-6:]: ax.set_xlabel('Longitude', fontsize=10)
        if ax in axes[::6]: ax.set_ylabel('Latitude', fontsize=10)
        
    # Tambah satu Colorbar raksasa di bagian bawah
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Kecepatan Arus (m/s)', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.08, 1, 0.93]) 
    # Simpan Gambar Peta
    map_save_path = os.path.join(output_dir, f"peta_prakiraan_{file_timestamp}_WIB.png")
    plt.savefig(map_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Peta Spasial berhasil dibuat!\n   -> {map_save_path}")

if __name__ == "__main__":
    try:
        main_operational_forecast()
    except Exception as e:
        print(f"\n❌ PROSES OPERASIONAL GAGAL: {str(e)}")