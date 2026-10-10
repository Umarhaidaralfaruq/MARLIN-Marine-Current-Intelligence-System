# =============================================================================
# FILE: 02_eof_extract.py (MODULAR EOF / MEOF + SPATIAL PLOT)
# DESKRIPSI: Separasi EOF/MEOF, Filter Radar Off, Segmentasi, dan Plot Loadings
# =============================================================================

import xarray as xr
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import PATHS, MODEL_CONFIG, FEATURE_TOGGLES

TRAIN_RATIO = MODEL_CONFIG['train_split_ratio']
N_MODES = MODEL_CONFIG['n_modes']
MAX_SAFE_NAN = MODEL_CONFIG['max_save_nan']
SPATIAL_MODE = MODEL_CONFIG.get('spatial_mode', 'EOF').upper() # Baca dari config

def apply_segmentation_datetime(df):
    df = df.sort_values('datetime').reset_index(drop=True)
    df['dt_hour'] = df['datetime'].diff().dt.total_seconds() / 3600
    df['break_seg'] = df['dt_hour'] > MAX_SAFE_NAN
    df['Segment_ID'] = df['break_seg'].cumsum() + 1
    
    seg_len = df.groupby('Segment_ID').size()
    valid_seg = seg_len[seg_len >= MODEL_CONFIG['input_window']].index
    
    df_final = df[df['Segment_ID'].isin(valid_seg)].copy()
    df_final['Segment_ID'] = df_final.groupby('Segment_ID').ngroup() + 1
    df_final = df_final.drop(columns=['dt_hour', 'break_seg'])
    return df_final

def plot_pca_spatial_loadings():
    """Menggambar Peta Area Pengaruh PCA (Spatial Loadings) untuk Mode 1 & 2"""
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
        # Jika lu pakai lebih dari 2 mode, fungsi ini tetap hanya memplot 2 mode pertama (paling dominan)
        if mode >= N_MODES: break 
        
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
            
            # KOREKSI LABEL DISINI
            if SPATIAL_MODE == 'EOF':
                ax.set_title(f"Mode {mode+1} - {title}\n(Variansi: {var:.1f}%)", fontsize=12)
            else:
                # Untuk MEOF, variansi ditaruh di title Mode, bukan per komponen
                if j == 0: 
                    ax.set_title(f"Mode {mode+1} - {title}\n(Total Variansi Mode: {var:.1f}%)", fontsize=12)
                else:
                    ax.set_title(f"Mode {mode+1} - {title}", fontsize=12)

    plt.tight_layout()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"pca_spatial_loadings_{SPATIAL_MODE}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Tersimpan di: {save_path}")
    plt.show()

def run_eof_extraction():
    print("\n" + "="*65)
    print(f"🧠 FASE 2: SEPARASI {SPATIAL_MODE} & SEGMENTASI DATETIME")
    print("="*65)

    file_input = PATHS['dataset_arus']
    if not os.path.exists(file_input):
        raise FileNotFoundError(f"❌ File {file_input} tidak ditemukan!")

    # 1. LOAD DATA PASUT
    df_pasut = None
    if FEATURE_TOGGLES.get('ENABLE_WATER_LEVEL', False):
        try:
            print("🌊 Membaca dan resampling data Elevasi Pasut...")
            df_pasut = pd.read_excel(PATHS['aws_database'])
            df_pasut.columns = df_pasut.columns.str.strip()
            df_pasut['datetime'] = pd.to_datetime(df_pasut['datetime'], format='%d/%m/%Y %H:%M:%S')
            df_pasut = df_pasut.set_index('datetime').resample('1H').mean()
            df_pasut = df_pasut.interpolate(method='time').reset_index()
            df_pasut = df_pasut.drop_duplicates(subset=['datetime'])
        except Exception as e:
            print(f"⚠️ Gagal memuat data pasut: {e}")

    # 2. LOAD DATA NETCDF & MASKING
    print("🥞 Memuat data dan meratakan matriks spasial (Stacking)...")
    ds = xr.open_dataset(file_input)
    time_arr = ds.time.values

    ds_mean_u = xr.open_dataset(PATHS['spatial_mean_u'])
    valid_grids_mask = ds_mean_u['u'].notnull().values.flatten()
    
    stacked = ds.stack(grid=['lat', 'lon'])
    ocean_only = stacked.isel(grid=valid_grids_mask)

    # =========================================================================
    # JALUR 1: INDEPENDENT EOF (TERPISAH U DAN V)
    # =========================================================================
    if SPATIAL_MODE == 'EOF':
        for comp in ['u', 'v']:
            print(f"\n🌪️ MENJALANKAN EKSTRAKSI EOF UNTUK KOMPONEN: {comp.upper()}")
            comp_matrix = ocean_only[comp].values
            
            is_radar_off = (comp_matrix == 0.0).all(axis=1)
            X_clean, time_clean = comp_matrix[~is_radar_off], time_arr[~is_radar_off]
            print(f"   -> Membuang {is_radar_off.sum()} jam data saat radar mati total.")
            
            split_idx = int(len(X_clean) * TRAIN_RATIO)
            X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
            time_train, time_test = time_clean[:split_idx], time_clean[split_idx:]
            
            pca = PCA(n_components=N_MODES, random_state=42)
            pc_train, pc_test = pca.fit_transform(X_train), pca.transform(X_test)
            
            explained_var = pca.explained_variance_ratio_ * 100
            print(f"✅ Variansi Terpilih ({N_MODES} Mode): {np.sum(explained_var):.2f}%")
            
            col_names = [f"{comp.upper()}_PC{i+1}" for i in range(N_MODES)]
            df_train = pd.DataFrame(pc_train, columns=col_names)
            df_train.insert(0, 'datetime', pd.to_datetime(time_train))
            df_test = pd.DataFrame(pc_test, columns=col_names)
            df_test.insert(0, 'datetime', pd.to_datetime(time_test))

            if df_pasut is not None:
                df_train = pd.merge(df_train, df_pasut, on='datetime', how='left')
                df_test = pd.merge(df_test, df_pasut, on='datetime', how='left')

            df_train_final = apply_segmentation_datetime(df_train)
            df_test_final = apply_segmentation_datetime(df_test)

            csv_train_path = str(PATHS['pc_train']).replace('.csv', f'_{comp}.csv')
            csv_test_path = str(PATHS['pc_test']).replace('.csv', f'_{comp}.csv')
            df_train_final.to_csv(csv_train_path, index=False)
            df_test_final.to_csv(csv_test_path, index=False)
            
            model_path = str(PATHS['meof_pca_model']).replace('_model.joblib', f'_{comp}_model.joblib')
            joblib.dump(pca, model_path)

    # =========================================================================
    # JALUR 2: MULTIVARIATE EOF (MEOF - GABUNGAN U DAN V)
    # =========================================================================
    elif SPATIAL_MODE == 'MEOF':
        print(f"\n🌪️ MENJALANKAN EKSTRAKSI MEOF (U & V TERINTEGRASI)")
        
        matrix_u = ocean_only['u'].values
        matrix_v = ocean_only['v'].values
        matrix_gabungan = np.hstack((matrix_u, matrix_v))
        
        is_radar_off = (matrix_gabungan == 0.0).all(axis=1)
        X_clean, time_clean = matrix_gabungan[~is_radar_off], time_arr[~is_radar_off]
        print(f"   -> Membuang {is_radar_off.sum()} jam data saat radar mati total.")
        
        split_idx = int(len(X_clean) * TRAIN_RATIO)
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        time_train, time_test = time_clean[:split_idx], time_clean[split_idx:]
        
        pca = PCA(n_components=N_MODES, random_state=42)
        pc_train, pc_test = pca.fit_transform(X_train), pca.transform(X_test)
        
        explained_var = pca.explained_variance_ratio_ * 100
        print(f"✅ Variansi MEOF Terpilih ({N_MODES} Mode): {np.sum(explained_var):.2f}%")
        
        col_names = [f"MEOF_PC{i+1}" for i in range(N_MODES)]
        df_train = pd.DataFrame(pc_train, columns=col_names)
        df_train.insert(0, 'datetime', pd.to_datetime(time_train))
        df_test = pd.DataFrame(pc_test, columns=col_names)
        df_test.insert(0, 'datetime', pd.to_datetime(time_test))

        if df_pasut is not None:
            df_train = pd.merge(df_train, df_pasut, on='datetime', how='left')
            df_test = pd.merge(df_test, df_pasut, on='datetime', how='left')

        df_train_final = apply_segmentation_datetime(df_train)
        df_test_final = apply_segmentation_datetime(df_test)

        csv_train_path = str(PATHS['pc_train']).replace('.csv', '_meof.csv')
        csv_test_path = str(PATHS['pc_test']).replace('.csv', '_meof.csv')
        df_train_final.to_csv(csv_train_path, index=False)
        df_test_final.to_csv(csv_test_path, index=False)
        
        model_path = str(PATHS['meof_pca_model']).replace('_model.joblib', '_meof_model.joblib')
        joblib.dump(pca, model_path)

    else:
        print(f"❌ ERROR: spatial_mode '{SPATIAL_MODE}' tidak dikenali di config. Gunakan 'EOF' atau 'MEOF'.")

    # PANGGIL FUNGSI PLOTTING SETELAH MODEL TERSIMPAN
    if SPATIAL_MODE in ['EOF', 'MEOF']:
        plot_pca_spatial_loadings()

    ds.close()
    print("\n✅ FASE 2 SELESAI!")

if __name__ == "__main__":
    run_eof_extraction()