# =============================================================================
# FILE: operational_prediction.py (MODULAR EOF / MEOF)
# DESKRIPSI: Prediksi Harian Arus Selat Bali (Export CSV & Peta 24-Panel)
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt

from config import PATHS, MODEL_CONFIG, FEATURE_TOGGLES, SPATIAL_CONFIG
from feature_engineering import build_features

LOOKBACK = MODEL_CONFIG['input_window']
HORIZON = MODEL_CONFIG['output_window']
N_PC_FEATURES = MODEL_CONFIG['n_modes']
SPATIAL_MODE = MODEL_CONFIG.get('spatial_mode', 'EOF').upper()

def load_production_artifacts(comp):
    pca_path = str(PATHS['meof_pca_model']).replace('_model.joblib', f'_{comp}_model.joblib')
    pca = joblib.load(pca_path)
    scaler_path = str(PATHS['scaler_path']).replace('.joblib', f'_{comp}.joblib')
    scaler = joblib.load(scaler_path)
    model_path = str(PATHS['model_path']).replace('.h5', f'_{comp}.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    return pca, scaler, model


def get_latest_data_for_comp(comp, target_time=None):
    df_full = pd.read_csv(PATHS['data_gabung']).dropna(subset=['datetime'])
    df_full['datetime'] = pd.to_datetime(df_full['datetime'])
    df_full = df_full.sort_values('datetime').reset_index(drop=True)

    # Potong dataframe jika ada target waktu (custom limit)
    if target_time is not None:
        target_dt = pd.to_datetime(target_time)
        df_full = df_full[df_full['datetime'] <= target_dt].reset_index(drop=True)

    buffer_rows = 24 * 7
    df_buffer = (
        df_full.iloc[-buffer_rows:].copy().reset_index(drop=True)
        if len(df_full) >= buffer_rows
        else df_full.copy()
    )
    
    # Pertahankan nama PC ASLI seperti saat training.
    if SPATIAL_MODE == 'EOF':
        pc_cols = [
            f"{comp.upper()}_PC{i+1}"
            for i in range(N_PC_FEATURES)
        ]
    else:
        pc_cols = [
            f"MEOF_PC{i+1}"
            for i in range(N_PC_FEATURES)
        ]

    cols_to_keep = ['datetime', 'Segment_ID']

    if FEATURE_TOGGLES.get('ENABLE_WATER_LEVEL', False):
        cols_to_keep.append('waterlevel')

    cols_to_keep += pc_cols

    df_comp = df_buffer[cols_to_keep].copy()

    # Tidak ada rename MEOF_PC1 -> PC1.
    return df_comp

def inverse_scale_operational(y_pred_batch, scaler_obj, total_cols, n_pcs):
    out_len = y_pred_batch.shape[1]
    dummy = np.zeros((out_len, total_cols))
    dummy[:, :n_pcs] = y_pred_batch[0] 
    unscaled = scaler_obj.inverse_transform(dummy)
    return unscaled[:, :n_pcs] 

def main_operational_forecast():
    print("\n" + "="*60)
    print(f"🚀 RUNNING OPERATIONAL FORECAST (MODE: {SPATIAL_MODE})")
    print("="*60)
    
    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    ds_mean_v = xr.open_dataset(PATHS['spatial_mean_v'])
    
    lats = ds_mean_u.lat.values
    lons = ds_mean_u.lon.values
    n_lat, n_lon = len(lats), len(lons)
    
    valid_mask = ds_mean_u['u'].notnull().values.flatten()
    valid_grid_count = valid_mask.sum()
    
    mean_u_map = ds_mean_u['u'].values.flatten()[valid_mask]
    mean_v_map = ds_mean_v['v'].values.flatten()[valid_mask]
    
    Lon, Lat = np.meshgrid(lons, lats)
    lon_valid = Lon.flatten()[valid_mask]
    lat_valid = Lat.flatten()[valid_mask]

    pred_pc_outputs = {'u': None, 'v': None}
    latest_timestamp = None
    components_to_run = ['u', 'v'] if SPATIAL_MODE == 'EOF' else ['meof']

    for comp in components_to_run:
        pca, scaler, model = load_production_artifacts(comp)
        target_time='2026-10-26 23:00:00'
        df_buffer = get_latest_data_for_comp(comp, target_time)
        latest_timestamp = df_buffer['datetime'].max()
        
        df_proc = build_features(df_buffer)
        
        target_cols = [c for c in df_proc.columns if 'PC' in c and not c.startswith('rolling_') and not c.startswith('jarak_')]
        input_cols = [c for c in df_proc.columns if c not in ['datetime', 'Segment_ID']]
        scale_cols = target_cols + [c for c in input_cols if c not in target_cols]
        
        df_input_window = df_proc.iloc[-LOOKBACK:].copy().reset_index(drop=True)
        if df_input_window[scale_cols].isna().any().any():
            df_input_window[scale_cols] = df_input_window[scale_cols].interpolate(method='linear', limit_direction='both')
            
        df_input_window[scale_cols] = scaler.transform(df_input_window[scale_cols])
        X_prod = df_input_window[input_cols].values.reshape(1, LOOKBACK, -1)
        
        print(f"🤖 LSTM memprediksi arus masa depan untuk {comp.upper()}...")
        y_pred = model.predict(X_prod, verbose=0)
        
        unscaled_pc = inverse_scale_operational(y_pred, scaler, len(scale_cols), len(target_cols))
        
        if SPATIAL_MODE == 'EOF':
            pred_pc_outputs[comp] = unscaled_pc
        else:
            print("🪓 Membelah Matriks Prediksi MEOF menjadi U dan V...")
            pred_pc_outputs['meof'] = unscaled_pc

    print(f"\n🌐 Data terakhir tercatat: {latest_timestamp}")
    print("🌐 Merekonsiliasi PC menjadi Peta 2D dan menghitung Kecepatan & Arah...")
    
    future_timestamps = pd.date_range(start=latest_timestamp + pd.Timedelta(hours=1), periods=HORIZON, freq='1H')
    all_forecast_rows = []
    maps_for_plotting = []

    if SPATIAL_MODE == 'EOF':
        pca_u = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_u_model.joblib'))
        pca_v = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_v_model.joblib'))
    else:
        pca_meof = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_meof_model.joblib'))

    for h_idx in range(HORIZON):
        current_pred_time = future_timestamps[h_idx]
        
        if SPATIAL_MODE == 'EOF':
            pc_u_hour = pred_pc_outputs['u'][h_idx].reshape(1, -1)
            pc_v_hour = pred_pc_outputs['v'][h_idx].reshape(1, -1)
            anom_u_1d = pca_u.inverse_transform(pc_u_hour)[0]
            anom_v_1d = pca_v.inverse_transform(pc_v_hour)[0]
        else:
            pc_meof_hour = pred_pc_outputs['meof'][h_idx].reshape(1, -1)
            anom_meof_1d = pca_meof.inverse_transform(pc_meof_hour)[0]
            anom_u_1d = anom_meof_1d[:valid_grid_count]
            anom_v_1d = anom_meof_1d[valid_grid_count:]
        
        def sinkronisasi_grid(anomali, target_size):
            if len(anomali) > target_size: return anomali[:target_size]
            elif len(anomali) < target_size: return np.concatenate((anomali, np.zeros(target_size - len(anomali))))
            return anomali

        anom_u_1d = sinkronisasi_grid(anom_u_1d, len(mean_u_map))
        anom_v_1d = sinkronisasi_grid(anom_v_1d, len(mean_u_map))
        
        u_absolute = anom_u_1d + mean_u_map
        v_absolute = anom_v_1d + mean_v_map
        
        speed = np.sqrt(u_absolute**2 + v_absolute**2)
        speed = np.clip(speed, a_min=0.0, a_max=SPATIAL_CONFIG['max_physical_speed'])
        direction = (90 - np.degrees(np.arctan2(v_absolute, u_absolute))) % 360
        
        if h_idx in range(HORIZON):
            def to_2d(data_1d):
                base = np.full(n_lat * n_lon, np.nan)
                base[valid_mask] = data_1d
                return base.reshape(n_lat, n_lon)
            
            maps_for_plotting.append({
                'time': current_pred_time, 'u_2d': to_2d(u_absolute),
                'v_2d': to_2d(v_absolute), 'speed_2d': to_2d(speed)
            })
        
        for i in range(len(lon_valid)):
            all_forecast_rows.append({
                'datetime_prediction': current_pred_time,
                'forecast_horizon_hour': h_idx + 1,
                'latitude': lat_valid[i], 'longitude': lon_valid[i],
                'u_velocity_ms': round(u_absolute[i], 4),
                'v_velocity_ms': round(v_absolute[i], 4),
                'current_speed_ms': round(speed[i], 4),
                'current_direction_deg': round(direction[i], 1)
            })

    df_final_forecast = pd.DataFrame(all_forecast_rows)
    output_dir = "outputs/predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    file_timestamp = future_timestamps[0].strftime('%Y%m%d_%H%M')
    csv_save_path = os.path.join(PATHS['OUTPUT_PRED'], f"prediksi_spasial_{file_timestamp}.csv")
    df_final_forecast.to_csv(csv_save_path, index=False)
    print(f"\n💾 CSV Ramalan Laut tersimpan di:\n   -> {csv_save_path}")

    print("🎨 Menggambar Peta Prakiraan Cuaca Harian (24-Panel)...")
    fig, axes = plt.subplots(4, 6, figsize=(26, 18), sharex=True, sharey=True)
    
    waktu_awal_wib = latest_timestamp + pd.Timedelta(hours=7)
    fig.suptitle(f"Prakiraan Arus Permukaan Selat Bali (24 Jam ke Depan)\nBasis Waktu Awal: {waktu_awal_wib.strftime('%Y-%m-%d %H:%M WIB')} | Mode: {SPATIAL_MODE}", fontsize=22, fontweight='bold', y=0.95)

    axes = axes.flatten()
    skip = (slice(None, None, 1), slice(None, None, 1))

    for ax, map_data in zip(axes, maps_for_plotting):
        U, V, Speed = map_data['u_2d'], map_data['v_2d'], map_data['speed_2d']
        waktu_wib_panel = map_data['time'] + pd.Timedelta(hours=7)
        plot_time_str = waktu_wib_panel.strftime('%H:%M WIB\n(%d %b)')
        
        contour = ax.contourf(Lon, Lat, Speed, cmap='jet', alpha=0.7, levels=np.linspace(0, 3.0, 20))
        ax.quiver(Lon[skip], Lat[skip], U[skip], V[skip], color='black', scale=20, width=0.003, headwidth=2)
        ax.set_title(f"{plot_time_str}", fontsize=12, fontweight='bold')
        if ax in axes[-6:]: ax.set_xlabel('Longitude', fontsize=10)
        if ax in axes[::6]: ax.set_ylabel('Latitude', fontsize=10)
        
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Kecepatan Arus (m/s)', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.08, 1, 0.93]) 
    map_save_path = os.path.join(output_dir, f"peta_prakiraan_{file_timestamp}_WIB.png")
    plt.savefig(map_save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Peta Spasial berhasil dibuat!\n   -> {map_save_path}")

if __name__ == "__main__":
    try:
        main_operational_forecast()
    except Exception as e:
        print(f"\n❌ PROSES OPERASIONAL GAGAL: {str(e)}")
