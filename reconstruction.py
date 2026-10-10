# =============================================================================
# FILE: 05_reconstruction.py (MODULAR EOF / MEOF)
# DESKRIPSI: Fase 5 - Inferensi, Inverse PCA Dinamis, & Visualisasi Evaluasi
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from config import PATHS, MODEL_CONFIG, VISUAL_CONFIG

# Import modul FE
from feature_engineering import build_features

LOOKBACK = MODEL_CONFIG['input_window']
HORIZON = MODEL_CONFIG['output_window']
SPATIAL_MODE = MODEL_CONFIG.get('spatial_mode', 'EOF').upper()

def load_artifacts(comp):
    """Memuat PCA, Scaler, dan Model LSTM untuk komponen tertentu"""
    print(f"📦 Memuat artifak untuk komponen {comp.upper()}...")
    pca_path = str(PATHS['meof_pca_model']).replace('_model.joblib', f'_{comp}_model.joblib')
    pca = joblib.load(pca_path)
    
    scaler_path = str(PATHS['scaler_path']).replace('.joblib', f'_{comp}.joblib')
    scaler = joblib.load(scaler_path)
    
    model_path = str(PATHS['model_path']).replace('.h5', f'_{comp}.h5')
    model = tf.keras.models.load_model(model_path, compile=False) 
    
    return pca, scaler, model

def prepare_test_data(comp):
    """Membangun ulang data Test secara dinamis dan mendeteksi kolom target"""
    test_path = str(PATHS['pc_test']).replace('.csv', f'_{comp}.csv')
    df_test = pd.read_csv(test_path, parse_dates=['datetime'])
    
    df_test = build_features(df_test)
    
    # 🎯 Deteksi Target Dinamis
    target_cols = [c for c in df_test.columns if 'PC' in c and not c.startswith('rolling_') and not c.startswith('jarak_')]
    input_cols = [c for c in df_test.columns if c not in ['datetime', 'Segment_ID']]
    scale_cols = target_cols + [c for c in input_cols if c not in target_cols]
    
    return df_test, scale_cols, input_cols, target_cols

def inverse_scale_dynamic(y_data, scaler_obj, total_scale_cols, n_target_cols):
    """Mengembalikan nilai dari -1..1 ke nilai PC asli"""
    n_samples, out_len, n_pc = y_data.shape
    dummy = np.zeros((n_samples * out_len, total_scale_cols))
    dummy[:, :n_target_cols] = y_data.reshape(-1, n_pc)
    unscaled = scaler_obj.inverse_transform(dummy)
    return unscaled[:, :n_target_cols].reshape(n_samples, out_len, n_target_cols)

def plot_pca_spatial_loadings():
    print(f"\n🎨 Menggambar Peta PCA Spatial Loadings (Mode: {SPATIAL_MODE})...")
    
    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    lats, lons = ds_mean_u.lat.values, ds_mean_u.lon.values
    n_lat, n_lon = len(lats), len(lons)
    valid_mask = ds_mean_u['u'].notnull().values.flatten()
    valid_grid_count = valid_mask.sum()
    Lon, Lat = np.meshgrid(lons, lats)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle(f"Visualisasi Area Pengaruh PCA ({SPATIAL_MODE})\nMode 1 & Mode 2", fontsize=16, fontweight='bold')
    
    if SPATIAL_MODE == 'EOF':
        pca_u = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_u_model.joblib'))
        pca_v = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_v_model.joblib'))
    else:
        pca_meof = joblib.load(str(PATHS['meof_pca_model']).replace('_model.joblib', '_meof_model.joblib'))
    
    def to_2d(data_1d):
        base = np.full(n_lat * n_lon, np.nan)
        base[valid_mask] = data_1d
        return base.reshape(n_lat, n_lon)
            
    for i, mode in enumerate([0, 1]): 
        if SPATIAL_MODE == 'EOF':
            bobot_u_1d = pca_u.components_[mode]
            bobot_v_1d = pca_v.components_[mode]
            var_u = pca_u.explained_variance_ratio_[mode]*100
            var_v = pca_v.explained_variance_ratio_[mode]*100
        else: # MEOF Split Logic
            bobot_meof = pca_meof.components_[mode]
            bobot_u_1d = bobot_meof[:valid_grid_count]
            bobot_v_1d = bobot_meof[valid_grid_count:]
            var_u = var_v = pca_meof.explained_variance_ratio_[mode]*100
        
        map_u, map_v = to_2d(bobot_u_1d), to_2d(bobot_v_1d)
        
        for j, (comp_map, title, var) in enumerate(zip([map_u, map_v], ['Komponen U', 'Komponen V'], [var_u, var_v])):
            ax = axes[i, j]
            max_val = np.nanmax(np.abs(comp_map))
            contour = ax.contourf(Lon, Lat, comp_map, cmap='bwr', levels=np.linspace(-max_val, max_val, 20))
            ax.set_title(f"Mode {mode+1} - {title}\n(Variansi: {var:.1f}%)", fontsize=12)
            if i == 1: ax.set_xlabel('Longitude')
            if j == 0: ax.set_ylabel('Latitude')
            fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"pca_spatial_loadings_{SPATIAL_MODE}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Tersimpan di: {save_path}")

# =========================================================================
# 🚀 SIMULASI KOMPREHENSIF (MENDUKUNG MEOF & EOF)
# =========================================================================
def run_period_simulation(start_date, end_date):
    print("\n" + "="*75)
    print(f"🌪️ MENJALANKAN SIMULASI KOMPREHENSIF ({SPATIAL_MODE}): {start_date} s/d {end_date}")
    print("="*75)
    
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    
    ds_mean_u_info = xr.open_dataset(PATHS['spatial_mean_u'])
    lats, lons = ds_mean_u_info.lat.values, ds_mean_u_info.lon.values
    n_lat, n_lon = len(lats), len(lons)
    valid_mask = ds_mean_u_info['u'].notnull().values.flatten()
    valid_grid_count = valid_mask.sum()

    # Dictionary penampung Anomali 1D untuk U dan V (Output Akhir)
    anom_dict = {'Pred': {'u': None, 'v': None}, 'True': {'u': None, 'v': None}, 'Persist': {'u': None, 'v': None}}

    components_to_run = ['u', 'v'] if SPATIAL_MODE == 'EOF' else ['meof']
    
    for comp in components_to_run:
        pca, scaler, model = load_artifacts(comp)
        df_test, scale_cols, input_cols, target_cols = prepare_test_data(comp)
        n_targets = len(target_cols)
        
        df_scaled = df_test.copy()
        df_scaled[scale_cols] = scaler.transform(df_scaled[scale_cols])
        
        X_list, Y_true_list, Y_persist_list = [], [], []
        
        for seg_id, group in df_scaled.groupby('Segment_ID'):
            group = group.reset_index(drop=True)
            n_rows = len(group)
            if n_rows < LOOKBACK + HORIZON: continue
            
            X_data = group[input_cols].values
            Y_data = group[target_cols].values
            times = group['datetime'].values
            
            target_start_times = pd.to_datetime(times[LOOKBACK : n_rows - HORIZON + 1])
            valid_mask_times = (target_start_times >= start_dt) & (target_start_times <= end_dt)
            valid_indices = np.where(valid_mask_times)[0]
            
            for i in valid_indices:
                idx_start_persist = i + LOOKBACK - 25 # Periodic 25 Jam
                if idx_start_persist < 0: continue
                    
                X_list.append(X_data[i : i+LOOKBACK])
                Y_true_list.append(Y_data[i+LOOKBACK : i+LOOKBACK+HORIZON])
                Y_persist_list.append(Y_data[idx_start_persist : idx_start_persist+HORIZON])
                    
        if not X_list:
            print(f"❌ Tidak ada data Test di rentang {start_date} s/d {end_date}!")
            return
            
        X_batch, Y_true_batch, Y_persist_batch = np.array(X_list), np.array(Y_true_list), np.array(Y_persist_list)
        
        print(f"🤖 Memprediksi {len(X_batch)} skenario (x 24 jam) untuk {comp.upper()}...")
        y_pred_batch = model.predict(X_batch, verbose=0)
        
        # Inverse Scale kembali ke nilai PC
        pc_pred = inverse_scale_dynamic(y_pred_batch, scaler, len(scale_cols), n_targets)
        pc_true = inverse_scale_dynamic(Y_true_batch, scaler, len(scale_cols), n_targets)
        pc_persist = inverse_scale_dynamic(Y_persist_batch, scaler, len(scale_cols), n_targets)
        
        n_samples = pc_pred.shape[0]
        
        # Transformasi PC (Anomali Spasial 1D)
        anom_pred = pca.inverse_transform(pc_pred.reshape(-1, n_targets))
        anom_true = pca.inverse_transform(pc_true.reshape(-1, n_targets))
        anom_persist = pca.inverse_transform(pc_persist.reshape(-1, n_targets))

        # Alokasi hasil rekonstruksi ke U dan V
        if SPATIAL_MODE == 'EOF':
            anom_dict['Pred'][comp], anom_dict['True'][comp], anom_dict['Persist'][comp] = anom_pred, anom_true, anom_persist
        else:
            # 🚀 LOGIKA PEMISAHAN MEOF
            print("🪓 Membelah Matriks MEOF menjadi komponen U dan V...")
            for k, arr in zip(['Pred', 'True', 'Persist'], [anom_pred, anom_true, anom_persist]):
                anom_dict[k]['u'] = arr[:, :valid_grid_count]
                anom_dict[k]['v'] = arr[:, valid_grid_count:]

    # ==========================================================
    print("\n🌐 Merekonsiliasi ke Peta Spasial dan Menghitung Fisika Kecepatan & Arah...")
    
    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    ds_mean_v = xr.open_dataset(PATHS['spatial_mean_v'])
    mean_u_1d = ds_mean_u['u'].values.flatten()[valid_mask]
    mean_v_1d = ds_mean_v['v'].values.flatten()[valid_mask]
    
    # Arus Riil = Anomali + Mean
    u_pred_real = anom_dict['Pred']['u'] + mean_u_1d
    v_pred_real = anom_dict['Pred']['v'] + mean_v_1d
    u_true_real = anom_dict['True']['u'] + mean_u_1d
    v_true_real = anom_dict['True']['v'] + mean_v_1d
    u_persist_real = anom_dict['Persist']['u'] + mean_u_1d 
    v_persist_real = anom_dict['Persist']['v'] + mean_v_1d 

    # Hitung Kecepatan & Arah
    speed_pred = np.sqrt(u_pred_real**2 + v_pred_real**2)
    speed_true = np.sqrt(u_true_real**2 + v_true_real**2)
    speed_persist = np.sqrt(u_persist_real**2 + v_persist_real**2)
    dir_pred = np.degrees(np.arctan2(u_pred_real, v_pred_real)) % 360
    dir_true = np.degrees(np.arctan2(u_true_real, v_true_real)) % 360

    # ==========================================================
    print("\n📊 EVALUASI PER HORIZON (LSTM vs PERIODIC PERSISTENCE 25H)")
    print("-" * 75)

    speed_pred_3d = speed_pred.reshape(n_samples, HORIZON, -1)
    speed_true_3d = speed_true.reshape(n_samples, HORIZON, -1)
    persistence_speed = speed_persist.reshape(n_samples, HORIZON, -1) 

    rmse_lstm_all, rmse_persist_all = [], []

    for h in range(HORIZON):
        diff_lstm = speed_pred_3d[:, h, :] - speed_true_3d[:, h, :]
        rmse_lstm = np.sqrt(np.mean(diff_lstm ** 2))
        mae_lstm = np.mean(np.abs(diff_lstm))

        diff_persist = persistence_speed[:, h, :] - speed_true_3d[:, h, :]
        rmse_persist = np.sqrt(np.mean(diff_persist ** 2))

        skill_score = 1 - (rmse_lstm / max(rmse_persist, 1e-8))
        rmse_lstm_all.append(rmse_lstm)
        rmse_persist_all.append(rmse_persist)

        print(f"H+{h+1:02d} | RMSE={rmse_lstm:.4f} | MAE={mae_lstm:.4f} | Persist={rmse_persist:.4f} | Skill={skill_score:.3f}")

    print("\n🏆 RINGKASAN SKILL SCORE")
    print("-" * 70)
    rmse_lstm_mean = np.mean(rmse_lstm_all)
    rmse_persist_mean = np.mean(rmse_persist_all)
    skill_mean = 1 - (rmse_lstm_mean / rmse_persist_mean)
    
    print(f"RMSE LSTM Mean       : {rmse_lstm_mean:.4f} m/s")
    print(f"RMSE Periodic Persist: {rmse_persist_mean:.4f} m/s")
    print(f"Skill Score Mean     : {skill_mean:.3f}")
    
    y_true_flat, y_pred_flat, y_persist_flat = speed_true.flatten(), speed_pred.flatten(), speed_persist.flatten()

    r2_lstm = r2_score(y_true_flat, y_pred_flat)
    r2_persist = r2_score(y_true_flat, y_persist_flat)
    corr_lstm, _ = pearsonr(y_true_flat, y_pred_flat)
    corr_persist, _ = pearsonr(y_true_flat, y_persist_flat)

    print("\n📈 RINGKASAN R² & KORELASI")
    print("-" * 70)
    print(f"R² Score LSTM            : {r2_lstm:.4f}")
    print(f"R² Score Persist         : {r2_persist:.4f}")
    print(f"Korelasi (r) LSTM        : {corr_lstm:.4f}")
    print(f"Korelasi (r) Persist     : {corr_persist:.4f}")
    
    # (Sisa Evaluasi Spasial Tetap Sama, Tidak Diubah)
    mae_speed_1d = np.mean(np.abs(speed_pred - speed_true), axis=0)
    rmse_speed_1d = np.sqrt(np.mean((speed_pred - speed_true)**2, axis=0))
    
    def circular_error(pred, true):
        diff = np.abs(pred - true) % 360
        return np.where(diff > 180, 360 - diff, diff)
        
    rmse_dir_1d = np.sqrt(np.mean(circular_error(dir_pred, dir_true)**2, axis=0))
    
    print("\n📈 RINGKASAN R² & KORELASI")
    print("-" * 70)
    print(f"R² Score LSTM            : {r2_lstm:.4f}")
    print(f"R² Score Persist         : {r2_persist:.4f}")
    print(f"Korelasi (r) LSTM        : {corr_lstm:.4f}")
    print(f"Korelasi (r) Persist     : {corr_persist:.4f}")

    # =====================================================================
    # 📈 PLOTTING GRAFIK EVALUASI PER HORIZON (DIKEMBALIKAN)
    # =====================================================================
    print("\n📈 Menggambar grafik Evaluasi per Horizon (RMSE & Skill Score)...")
    
    hours = np.arange(1, HORIZON + 1)
    skill_scores_all = 1 - (np.array(rmse_lstm_all) / np.maximum(np.array(rmse_persist_all), 1e-8))
    
    fig_hz, axes_hz = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig_hz.suptitle(f"Performa Model vs Periodic Persistence (1 - {HORIZON} Jam)\nMode Spasial: {SPATIAL_MODE}", fontsize=16, fontweight='bold')
    
    axes_hz[0].plot(hours, rmse_lstm_all, marker='o', color='blue', label='LSTM (AI)', linewidth=2.5)
    axes_hz[0].plot(hours, rmse_persist_all, marker='x', color='gray', linestyle='--', label='Periodic Persistence', linewidth=2)
    axes_hz[0].set_ylabel("RMSE Kecepatan (m/s)", fontsize=12)
    axes_hz[0].set_title("Perbandingan Error (Semakin rendah semakin baik)", fontsize=13)
    axes_hz[0].grid(True, linestyle=':', alpha=0.7)
    axes_hz[0].legend(fontsize=11)
    
    axes_hz[1].plot(hours, skill_scores_all, marker='s', color='green', label='Skill Score', linewidth=2.5)
    axes_hz[1].axhline(y=0, color='red', linestyle='-', linewidth=1.5, label='Batas 0 (AI = Baseline)') 
    axes_hz[1].set_ylabel("Skill Score", fontsize=12)
    axes_hz[1].set_title("Skill Score Oseanografi (Semakin mendekati 1 semakin baik)", fontsize=13)
    axes_hz[1].set_xlabel("Horizon Prediksi Masa Depan (Jam ke-)", fontsize=12)
    axes_hz[1].set_xticks(hours)
    axes_hz[1].set_ylim(-0.5, 1.0)
    axes_hz[1].grid(True, linestyle=':', alpha=0.7)
    axes_hz[1].legend(fontsize=11)
    
    plt.tight_layout()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    hz_save_path = os.path.join(output_dir, f"grafik_horizon_{SPATIAL_MODE}.png")
    print(f"💾 Menyimpan Grafik Horizon ke: {hz_save_path}")
    plt.savefig(hz_save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================================
    # 📏 PERHITUNGAN METRIK SPASIAL (TERMASUK R2, KORELASI & SKILL SCORE)
    # =====================================================================
    # 1. Metrik Error (RMSE & MAE)
    mae_speed_1d = np.mean(np.abs(speed_pred - speed_true), axis=0)
    rmse_speed_1d = np.sqrt(np.mean((speed_pred - speed_true)**2, axis=0))
    rmse_persist_1d = np.sqrt(np.mean((speed_persist - speed_true)**2, axis=0))
    p90_rmse_speed = np.percentile(rmse_speed_1d, 90)
    
    def circular_error(pred, true):
        diff = np.abs(pred - true) % 360
        return np.where(diff > 180, 360 - diff, diff)
        
    rmse_dir_1d = np.sqrt(np.mean(circular_error(dir_pred, dir_true)**2, axis=0))

    # 2. Metrik Kualitas (R², Korelasi) per titik grid
    ss_res = np.sum((speed_true - speed_pred)**2, axis=0)
    ss_tot = np.sum((speed_true - np.mean(speed_true, axis=0))**2, axis=0)
    r2_spatial_1d = 1 - (ss_res / np.maximum(ss_tot, 1e-8))

    true_mean = np.mean(speed_true, axis=0)
    pred_mean = np.mean(speed_pred, axis=0)
    num = np.sum((speed_true - true_mean) * (speed_pred - pred_mean), axis=0)
    den = np.sqrt(np.sum((speed_true - true_mean)**2, axis=0) * np.sum((speed_pred - pred_mean)**2, axis=0))
    corr_spatial_1d = num / np.maximum(den, 1e-8)
    
    # 3. Metrik Spasial Skill Score
    skill_spatial_1d = 1 - (rmse_speed_1d / np.maximum(rmse_persist_1d, 1e-8))
    

    print("\n🌐 RINGKASAN ERROR SPASIAL")
    print("-" * 60)
    print(f"MAE Speed Mean    : {np.nanmean(mae_speed_1d):.4f} m/s")
    print(f"RMSE Speed Mean   : {np.nanmean(rmse_speed_1d):.4f} m/s")
    print(f"RMSE Direction    : {np.nanmean(rmse_dir_1d):.2f}°")
    print(f"Best Grid RMSE    : {np.nanmin(rmse_speed_1d):.4f} m/s")
    print(f"P90 Grid RMSE     : {p90_rmse_speed:.4f} m/s")
    print(f"Worst Grid RMSE   : {np.nanmax(rmse_speed_1d):.4f} m/s")

    # =====================================================================
    # 🗺️ MENGGAMBAR PETA SPASIAL 2D (6 PANEL KUALITAS JURNAL)
    # =====================================================================
    print("\n🎨 Menggambar Peta Evaluasi Spasial 6-Panel...")
    
    def to_2d_map(data_1d):
        base_map = np.full(n_lat * n_lon, np.nan)
        base_map[valid_mask] = data_1d
        return base_map.reshape(n_lat, n_lon)
        
    mae_speed_map = to_2d_map(mae_speed_1d)
    rmse_speed_map = to_2d_map(rmse_speed_1d)
    rmse_dir_map = to_2d_map(rmse_dir_1d)
    r2_map = to_2d_map(r2_spatial_1d)
    corr_map = to_2d_map(corr_spatial_1d)
    skill_map = to_2d_map(skill_spatial_1d)

    Lon, Lat = np.meshgrid(lons, lats)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharex=True, sharey=True)
    fig.suptitle(f"Evaluasi Spasial Ekstensif (Mode: {SPATIAL_MODE})\nPeriode: {start_date} s/d {end_date} | Total Skenario: {n_samples * 24} Jam", fontsize=18, fontweight='bold', y=1.02)

    maps_to_plot = [
        # Baris 1: Error Metrik (Lebih rendah lebih baik)
        (0, 0, mae_speed_map, 'MAE Kecepatan (m/s)', 'hot_r', 0, VISUAL_CONFIG['max_val_mae_speed']),
        (0, 1, rmse_speed_map, 'RMSE Kecepatan (m/s)', 'hot_r', 0, VISUAL_CONFIG['max_val_rmse_speed']),
        (0, 2, rmse_dir_map, 'RMSE Arah Arus (Derajat)', 'plasma_r', 0, VISUAL_CONFIG['max_val_rmse_dir']),
        # Baris 2: Quality Metrik (Lebih tinggi lebih baik)
        (1, 0, r2_map, 'R² Score (Variansi Terekam)', 'viridis', 0.0, 1.0),
        (1, 1, corr_map, 'Korelasi Pearson (r)', 'viridis', 0.0, 1.0),
        (1, 2, skill_map, 'Skill Score (vs Persistence)', 'RdYlGn', -0.5, 1.0)
    ]

    for (row, col, data, title, cmap, vmin, vmax) in maps_to_plot:
        ax = axes[row, col]
        
        # Extend='both' agar nilai yang melebihi batas vmin/vmax tetap diwarnai dengan warna mentok ujung
        contour = ax.contourf(Lon, Lat, data, cmap=cmap, levels=np.linspace(vmin, vmax, 20), extend='both')
        
        # Tambahkan Garis Batas Toleransi / Threshold
        if 'MAE' in title:
            thresh = VISUAL_CONFIG['threshold_mae_speed']
            batas = ax.contour(Lon, Lat, data, levels=[thresh], colors='blue', linewidths=1.5, linestyles='dashed')
            ax.clabel(batas, inline=True, fontsize=10, fmt=f'Batas {thresh}', colors='blue')
        elif 'RMSE Kecepatan' in title:
            thresh = VISUAL_CONFIG['threshold_rmse_speed']
            batas = ax.contour(Lon, Lat, data, levels=[thresh], colors='blue', linewidths=1.5, linestyles='dashed')
            ax.clabel(batas, inline=True, fontsize=10, fmt=f'Batas {thresh}', colors='blue')
        elif 'RMSE Arah' in title:
            thresh = VISUAL_CONFIG['threshold_rmse_dir']
            batas = ax.contour(Lon, Lat, data, levels=[thresh], colors='cyan', linewidths=1.5, linestyles='dashed')
            ax.clabel(batas, inline=True, fontsize=10, fmt=f'Batas {thresh}°', colors='cyan')
        elif 'Skill Score' in title:
            # Garis hitam tegas di area yang AI-nya mulai kalah (Skill Score = 0)
            batas = ax.contour(Lon, Lat, data, levels=[0.0], colors='black', linewidths=1.5, linestyles='solid')
            ax.clabel(batas, inline=True, fontsize=10, fmt='Batas Seri (0.0)', colors='black')

        ax.set_title(title, fontsize=14)
        if row == 1: ax.set_xlabel('Longitude')
        if col == 0: ax.set_ylabel('Latitude')
        
        cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', fraction=0.046, pad=0.1)
        cbar.set_ticks(np.linspace(vmin, vmax, 6))

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"simulasi_komprehensif_6panel_{SPATIAL_MODE}.png")
    
    print(f"💾 Menyimpan Peta Simulasi Komprehensif (6 Panel) ke: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================================
    # 🎯 INVESTIGASI TITIK TERBAIK & TERBURUK
    # =====================================================================
    print("\n" + "="*70)
    print("🎯 INVESTIGASI METRIK DI TITIK KOORDINAT TERBAIK & TERBURUK")
    print("="*70)

    Lon_grid, Lat_grid = np.meshgrid(lons, lats)
    lon_valid, lat_valid = Lon_grid.flatten()[valid_mask], Lat_grid.flatten()[valid_mask]
    best_idx, worst_idx = np.nanargmin(rmse_speed_1d), np.nanargmax(rmse_speed_1d)

    def print_point_metrics(idx, label):
        lat_pt, lon_pt = lat_valid[idx], lon_valid[idx]
        y_true_pt, y_pred_pt, y_persist_pt = speed_true[:, idx], speed_pred[:, idx], speed_persist[:, idx]
        
        rmse_pt = np.sqrt(np.mean((y_true_pt - y_pred_pt)**2))
        rmse_persist_pt = np.sqrt(np.mean((y_true_pt - y_persist_pt)**2))
        skill_pt = 1 - (rmse_pt / max(rmse_persist_pt, 1e-8))
        r2_pt = r2_score(y_true_pt, y_pred_pt)
        corr_pt, _ = pearsonr(y_true_pt, y_pred_pt)
        
        print(f"\n📍 {label}")
        print(f"   Koordinat   : Lat {lat_pt:.5f}, Lon {lon_pt:.5f}")
        print("-" * 50)
        print(f"   RMSE LSTM      : {rmse_pt:.4f} m/s")
        print(f"   RMSE Persist   : {rmse_persist_pt:.4f} m/s")
        print(f"   Skill Score    : {skill_pt:.3f}  ({(skill_pt)*100:.1f}%)")
        print(f"   R² Score       : {r2_pt:.4f} | Korelasi (r): {corr_pt:.4f}")

    print_point_metrics(best_idx, "TITIK PALING AKURAT (BEST GRID)")
    print_point_metrics(worst_idx, "TITIK PALING TURBULEN (WORST GRID)")

if __name__ == "__main__":
    # AUTO-DETECT TANGGAL TEST
    test_file = str(PATHS['pc_test']).replace('.csv', '_meof.csv' if SPATIAL_MODE == 'MEOF' else '_u.csv')
    df_cek = pd.read_csv(test_file)
    tgl_awal, tgl_akhir = df_cek['datetime'].min(), df_cek['datetime'].max()
    
    run_period_simulation(start_date=tgl_awal, end_date=tgl_akhir)
    #plot_pca_spatial_loadings()