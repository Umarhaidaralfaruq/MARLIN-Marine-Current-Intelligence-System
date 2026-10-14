# =============================================================================
# FILE: main.py
# FUNGSI: API Backend MARLIN v2 (FastAPI) - GITHUB CLOUD NATIVE
# =============================================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import glob
import json
import matplotlib.tri as tri
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import requests
import time
import math

# Mengambil semua variabel konfigurasi dari config.py (termasuk GITHUB_OWNER dkk)
from config import *

app = FastAPI(title="MARLIN API", description="API Backend Cloud Native (GitHub Fetch)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# FUNGSI HELPER GITHUB (MENGGANTIKAN BACA FILE LOKAL)
# ---------------------------------------------------------
github_cache = {"files": [], "last_fetch": 0}
CACHE_TTL = 300 # Cache daftar file 5 menit agar tidak kena rate limit GitHub API

def get_available_baseruns():
    """Mengambil daftar file CSV langsung dari GitHub REST API"""
    global github_cache
    now = time.time()
    
    if now - github_cache["last_fetch"] < CACHE_TTL and github_cache["files"]:
        return github_cache["files"]

    path_suffix = f"/{GITHUB_DATA_DIR}" if GITHUB_DATA_DIR else ""
    api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents{path_suffix}?ref={GITHUB_BRANCH}"
    
    try:
        # Menambahkan otorisasi token di sini
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {GITHUB_TOKEN}"
        }
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            files = []
            for item in data:
                if item['name'].startswith('prediksi_spasial') and item['name'].endswith('.csv'):
                    files.append({
                        "filename": item['name'],
                        "path": item['download_url'] 
                    })
            
            files = sorted(files, key=lambda x: x["filename"], reverse=True)
            github_cache["files"] = [f["path"] for f in files]
            github_cache["last_fetch"] = now
            return github_cache["files"]
        else:
            print(f"⚠️ Gagal akses GitHub API (Status: {response.status_code}). Fallback ke file lokal.")
            return get_local_baseruns()
    except Exception as e:
        print(f"⚠️ Error GitHub Fetch: {e}. Fallback ke file lokal.")
        return get_local_baseruns()

def get_local_baseruns():
    """Fallback jika internet mati atau testing lokal"""
    kemungkinan_path = ["prediksi_spasial*.csv", "data/prediksi_spasial*.csv", "output/prediksi_spasial*.csv"]
    for path in kemungkinan_path:
        files = sorted(glob.glob(path), reverse=True)
        if len(files) > 0:
            return files
    return []

def get_verif_csv_url():
    """Dapatkan URL Raw untuk file observasi menggunakan PATHS dari config"""
    try:
        # Ambil path dari config.py (fallback ke default kalau error)
        verif_path = PATHS.get("Verif_12jam", "Hasil_Validasi/Validasi_Real_Max_12H.csv")
        
        # Cek lokal dulu buat testing
        if os.path.exists(verif_path): 
            return verif_path
        
        # Bersihkan slash di awal kalau ada, biar URL gak dobel slash (//)
        clean_path = verif_path.lstrip('/')
        
        # Tembak raw GitHub sesuai path di config
        return f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{clean_path}"
    except Exception as e:
        print(f"⚠️ Error ambil path verif: {e}")
        return f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/Hasil_Validasi/Validasi_Real_Max_12H.csv"

# Cache sederhana untuk memori Pandas Dataframe
data_cache = {}

def load_data(file_path_or_url):
    # Buang token sementara dari URL agar kunci cache permanen
    clean_key = file_path_or_url.split("?")[0]
    
    if clean_key in data_cache:
        return data_cache[clean_key]
        
    storage_options = None
    if file_path_or_url.startswith("http"):
        storage_options = {"Authorization": f"token {GITHUB_TOKEN}"}

    df = pd.read_csv(file_path_or_url, storage_options=storage_options).rename(columns={
        'datetime_prediction': 'datetime', 'latitude': 'lat', 'longitude': 'lon', 
        'current_speed_ms': 'predicted_v_max', 'v_velocity_ms': 'predicted_v', 
        'u_velocity_ms': 'predicted_u', 'current_direction_deg': 'arah_derajat'
    })
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime_wib'] = df['datetime'] + pd.Timedelta(hours=7)
    
    # Simpan pakai clean_key
    data_cache[clean_key] = df
    return df

def get_clean_files():
    raw = get_available_baseruns()
    if not raw: return []
    return [f["path"] if isinstance(f, dict) else str(f) for f in raw]

def ambil_index_target(group):
    if METODE_STATISTIK == "P95":
        p95 = group['predicted_v_max'].quantile(0.95)
        if pd.isna(p95): return group.index[0]
        return (group['predicted_v_max'] - p95).abs().idxmin()
    return group['predicted_v_max'].idxmax()

def derajat_ke_utara_selatan(deg):
    return "" if pd.isna(deg) else ("Utara" if deg >= 270 or deg < 90 else "Selatan")

def format_rentang_waktu(waktu_series):
    if waktu_series.empty: return ""
    dt_start = waktu_series.iloc[0]
    dt_end = waktu_series.iloc[-1]
    bulan = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'Mei', 6:'Jun', 7:'Jul', 8:'Agu', 9:'Sep', 10:'Okt', 11:'Nov', 12:'Des'}
    tgl_start = f"{dt_start.day:02d} {bulan[dt_start.month]} {dt_start.year}"
    tgl_end = f"{dt_end.day:02d} {bulan[dt_end.month]} {dt_end.year}"
    if tgl_start == tgl_end:
         return f"({tgl_start}, {dt_start.strftime('%H:%M')} - {dt_end.strftime('%H:%M')} WIB)"
    return f"({tgl_start} {dt_start.strftime('%H:%M')} - {tgl_end} {dt_end.strftime('%H:%M')} WIB)"


# ---------------------------------------------------------
# ENDPOINTS API 
# ---------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "MARLIN API System Online. Data Source: GitHub Raw."}

@app.get("/api/baseruns")
def get_baseruns_list():
    files = get_clean_files()
    if not files: return {"status": "success", "data": []}
    
    baseruns_data = []
    for f in files:
        # os.path.basename berfungsi baik memotong URL maupun path lokal
        filename = os.path.basename(f).split("?")[0]
        label = filename.replace("prediksi_spasial_", "").replace(".csv", "") + " UTC"
        baseruns_data.append({"filename": filename, "label": label, "path": f})
        
    return {"status": "success", "data": baseruns_data}

@app.get("/api/summary")
def get_summary(pelabuhan: str = Query(..., description="ketapang atau gilimanuk"), time_str: str = Query(None)):
    files = get_clean_files()
    if not files: raise HTTPException(status_code=404, detail="Data tidak tersedia")
        
    df_spatial = load_data(files[0])
    
    if pelabuhan.lower() == "ketapang":
        lat_min, lat_max, lon_min, lon_max = KET_LAT_MIN, KET_LAT_MAX, KET_LON_MIN, KET_LON_MAX
    elif pelabuhan.lower() == "gilimanuk":
        lat_min, lat_max, lon_min, lon_max = GIL_LAT_MIN, GIL_LAT_MAX, GIL_LON_MIN, GIL_LON_MAX
    else:
        raise HTTPException(status_code=400, detail="Pelabuhan tidak valid")

    df_area = df_spatial[(df_spatial['lat'] >= lat_min) & (df_spatial['lat'] <= lat_max) & 
                         (df_spatial['lon'] >= lon_min) & (df_spatial['lon'] <= lon_max)]
                         
    if df_area.empty: raise HTTPException(status_code=404, detail="Data area kosong")

    if time_str: waktu_acuan = pd.to_datetime(time_str)
    else: waktu_acuan = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=7)

    idx_target = df_area.groupby('datetime_wib').apply(ambil_index_target)
    df_trend = df_area.loc[idx_target].reset_index(drop=True)
    
    df_trend['selisih_waktu'] = abs(df_trend['datetime_wib'] - waktu_acuan)
    baris_terpilih = df_trend.loc[df_trend['selisih_waktu'].idxmin()]
    
    waktu_batas = waktu_acuan + pd.Timedelta(hours=12)
    df_12h = df_trend[(df_trend['datetime_wib'] >= waktu_acuan) & (df_trend['datetime_wib'] <= waktu_batas)]
    baris_max = df_12h.loc[df_12h['predicted_v_max'].idxmax()] if not df_12h.empty else baris_terpilih

    return {
        "pelabuhan": pelabuhan.capitalize(),
        "current": {
            "speed_ms": float(baris_terpilih['predicted_v_max']),
            "direction": derajat_ke_utara_selatan(baris_terpilih['arah_derajat']),
            "time_wib": baris_terpilih['datetime_wib'].strftime('%Y-%m-%d %H:%M')
        },
        "peak": {
            "speed_ms": float(baris_max['predicted_v_max']),
            "direction": derajat_ke_utara_selatan(baris_max['arah_derajat']),
            "time_wib": baris_max['datetime_wib'].strftime('%Y-%m-%d %H:%M')
        }
    }

# Tambahkan ini di bagian atas (di bawah deklarasi data_cache = {})
api_response_cache = {}

@app.get("/api/spatial-grid")
def get_spatial_grid(time_str: str = Query(None)):
    files = get_clean_files()
    if not files: raise HTTPException(status_code=404, detail="Data tidak tersedia")
    
    # KUNCI CACHE: Biar kalau jam yang sama dipanggil lagi, langsung instan!
    cache_key = f"grid_{files[0]}_{time_str}"
    if cache_key in api_response_cache:
        return api_response_cache[cache_key]
        
    df_spatial = load_data(files[0])
    
    if time_str: df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    else:
        waktu_unik = sorted(df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M').unique())
        df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == waktu_unik[0]]

    if df_hour.empty: raise HTTPException(status_code=404, detail="Waktu tidak ditemukan")

    lo1, lo2 = float(df_hour['lon'].min()), float(df_hour['lon'].max())
    la1, la2 = float(df_hour['lat'].max()), float(df_hour['lat'].min()) 
    nx, ny = GRID_NX, GRID_NY
    grid_lon, grid_lat = np.linspace(lo1, lo2, nx), np.linspace(la1, la2, ny) 
    dx, dy = (lo2 - lo1) / (nx - 1), (la1 - la2) / (ny - 1) 
    
    triang = tri.Triangulation(df_hour['lon'], df_hour['lat'])
    x_tri, y_tri = triang.x[triang.triangles], triang.y[triang.triangles]
    batas_resolusi = BATAS_RESOLUSI if 'BATAS_RESOLUSI' in globals() else 0.015
    mask = np.any([np.sqrt((x_tri[:, i] - x_tri[:, (i+1)%3])**2 + (y_tri[:, i] - y_tri[:, (i+1)%3])**2) > batas_resolusi for i in range(3)], axis=0)
    triang.set_mask(mask)
    
    interp_u = tri.LinearTriInterpolator(triang, df_hour['predicted_u'])
    interp_v = tri.LinearTriInterpolator(triang, df_hour['predicted_v'])
    Grid_lon, Grid_lat = np.meshgrid(grid_lon, grid_lat)
    U_grid, V_grid = interp_u(Grid_lon, Grid_lat), interp_v(Grid_lon, Grid_lat)
    
    u_data, v_data = [], []
    for i in range(ny):
        for j in range(nx):
            u_val, v_val = U_grid[i, j], V_grid[i, j]
            if np.ma.is_masked(u_val) or np.isnan(u_val): u_data.append(None); v_data.append(None)
            else: u_data.append(float(u_val)); v_data.append(float(v_val))
                
    result = [
        {"header": {"parameterCategory": 2, "parameterNumber": 2, "nx": nx, "ny": ny, "lo1": lo1, "lo2": lo2, "la1": la1, "la2": la2, "dx": dx, "dy": dy}, "data": u_data},
        {"header": {"parameterCategory": 2, "parameterNumber": 3, "nx": nx, "ny": ny, "lo1": lo1, "lo2": lo2, "la1": la1, "la2": la2, "dx": dx, "dy": dy}, "data": v_data}
    ]
    
    # Simpan hasil ke memori
    api_response_cache[cache_key] = result
    return result

@app.get("/api/map-background")
def get_map_background(time_str: str = Query(None)):
    files = get_clean_files()
    if not files: raise HTTPException(status_code=404, detail="Data tidak tersedia")
    
    # KUNCI CACHE
    cache_key = f"bg_{files[0]}_{time_str}"
    if cache_key in api_response_cache:
        return api_response_cache[cache_key]
        
    df_spatial = load_data(files[0])
    
    if time_str: df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    else:
        waktu_unik = sorted(df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M').unique())
        df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == waktu_unik[0]]

    if df_hour.empty: raise HTTPException(status_code=404, detail="Waktu tidak ditemukan")

    lon_min, lon_max = float(df_hour['lon'].min()), float(df_hour['lon'].max())
    lat_min, lat_max = float(df_hour['lat'].min()), float(df_hour['lat'].max())
    
    triang = tri.Triangulation(df_hour['lon'], df_hour['lat'])
    x_tri, y_tri = triang.x[triang.triangles], triang.y[triang.triangles]
    batas_resolusi = BATAS_RESOLUSI if 'BATAS_RESOLUSI' in globals() else 0.015
    mask = np.any([np.sqrt((x_tri[:, i] - x_tri[:, (i+1)%3])**2 + (y_tri[:, i] - y_tri[:, (i+1)%3])**2) > batas_resolusi for i in range(3)], axis=0)
    triang.set_mask(mask)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off'); ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    cmap_custom = LinearSegmentedColormap.from_list("custom", ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ffa500', '#ff0000', '#4b0082'])
    
    # =========================================================
    # TRIK PAMUNGKAS: RASTERISASI & BICUBIC BLUR (SEPERTI REFERENSI)
    # =========================================================
    # 1. Bikin kanvas piksel yang rapat (300x300 pixel)
    nx_bg, ny_bg = 300, 300
    grid_lon_bg, grid_lat_bg = np.meshgrid(np.linspace(lon_min, lon_max, nx_bg), 
                                           np.linspace(lat_min, lat_max, ny_bg))
    
    # 2. Tuangkan data dari segitiga-segitiga kasar ke atas kanvas piksel yang rapat
    interp_bg = tri.LinearTriInterpolator(triang, df_hour['predicted_v_max'])
    Z_bg = interp_bg(grid_lon_bg, grid_lat_bg)
    
    # 3. Render sebagai gambar "Heatmap" dengan blur alami (bicubic)
    ax.imshow(Z_bg, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', 
              cmap=cmap_custom, vmin=0, vmax=4.0, alpha=0.9, interpolation='bicubic')
    # =========================================================
    
    buf = io.BytesIO()
    # DPI bisa dibiarkan di 150 atau 300. Karena datanya sudah berupa raster (gambar), 
    # file size-nya akan otomatis lebih kecil dan enteng dirender server.
    plt.savefig(buf, format='png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
    buf.seek(0); plt.close(fig)
    
    result = {"bounds": [[lat_min, lon_min], [lat_max, lon_max]], "image": base64.b64encode(buf.getvalue()).decode('utf-8')}
    api_response_cache[cache_key] = result
    return result

@app.get("/api/timeseries")
def get_timeseries(lat: float, lon: float):
    files = get_clean_files()
    if not files: raise HTTPException(status_code=404, detail="Data tidak tersedia")
    df = load_data(files[0])
    
    df_coords = df[['lat', 'lon']].drop_duplicates()
    df_coords['jarak'] = (df_coords['lat'] - lat)**2 + (df_coords['lon'] - lon)**2
    
    if df_coords['jarak'].min() > 0.0001: raise HTTPException(status_code=404, detail="Titik di luar area")
    
    titik_terdekat = df_coords.loc[df_coords['jarak'].idxmin()]
    df_titik = df[(df['lat'] == float(titik_terdekat['lat'])) & (df['lon'] == float(titik_terdekat['lon']))].sort_values('datetime_wib')
    
    return {
        "lat": float(titik_terdekat['lat']), "lon": float(titik_terdekat['lon']),
        "waktu": df_titik['datetime_wib'].dt.strftime('%H:%M WIB').tolist(),
        "kecepatan": df_titik['predicted_v_max'].tolist()
    }

@app.get("/api/forecast-12h")
def get_forecast_12h(filename: str = None):
    try:
        files = get_clean_files()
        if not files: return {"waktu": [], "kecepatan": [], "anotasi_blok": [], "title_range": ""}
        
        target_file = next((f for f in files if os.path.basename(f).split("?")[0] == filename), files[0])
        df = load_data(target_file)
        df_area = df[(df['lat'] >= -8.2) & (df['lat'] <= -8.0) & (df['lon'] >= 114.3) & (df['lon'] <= 114.5)]
        
        last_hour = df_area['datetime'].max().hour
        start_hour_utc, end_hour_utc = (12, 24) if last_hour < 12 else (0, 12)
        df_shift = df_area[(df_area['datetime'].dt.hour >= start_hour_utc) & (df_area['datetime'].dt.hour < end_hour_utc)]
        
        idx_target = df_shift.groupby('datetime_wib')['predicted_v_max'].idxmax()
        df_trend = df_shift.loc[idx_target].sort_values('datetime_wib').reset_index(drop=True)
        
        title_range = format_rentang_waktu(df_trend['datetime_wib'])
        waktu_list = df_trend['datetime_wib'].dt.strftime('%H:%M').tolist()
        kecepatan_list = [round(float(x), 2) if pd.notna(x) else 0.0 for x in df_trend['predicted_v_max']]
        
        anotasi_blok = []
        for i in range(0, len(df_trend), 3):
            chunk = df_trend.iloc[i:i+3]
            if chunk.empty: continue
            max_v = float(chunk['predicted_v_max'].max())
            if math.isnan(max_v): max_v = 0.0
            kat_arah = chunk['arah_derajat'].apply(lambda d: "Utara" if (pd.notna(d) and (d >= 270 or d < 90)) else "Selatan")
            arah_dom = kat_arah.mode()[0] if not kat_arah.mode().empty else "-"
            
            anotasi_blok.append({"xMin": waktu_list[i], "xMax": waktu_list[min(i+3, len(waktu_list)-1)], "xCenter": waktu_list[min(i+1, len(waktu_list)-1)], "max_v": max_v, "arah": arah_dom})
            
        return {"waktu": waktu_list, "kecepatan": kecepatan_list, "anotasi_blok": anotasi_blok, "title_range": title_range}
    except Exception as e:
        print("Error Forecast:", e)
        return {"waktu": [], "title_range": ""}

@app.get("/api/validation-12h")
def get_validation_12h(filename: str = None):
    try:
        files = get_clean_files()
        if not files: return {"waktu": [], "title_range": ""}
        
        target_idx = next((i for i, f in enumerate(files) if os.path.basename(f).split("?")[0] == filename), 0)
        valid_idx = target_idx + 1 
        if valid_idx >= len(files): return {"waktu": [], "title_range": "(Baserun Sebelumnya Tidak Tersedia)"}
             
        target_file = files[valid_idx]
        df_pred = load_data(target_file)
        df_area = df_pred[(df_pred['lat'] >= -8.2) & (df_pred['lat'] <= -8.0) & (df_pred['lon'] >= 114.3) & (df_pred['lon'] <= 114.5)]
        
        last_hour = df_area['datetime'].max().hour
        start_hour_utc, end_hour_utc = (12, 24) if last_hour < 12 else (0, 12)
        df_shift = df_area[(df_area['datetime'].dt.hour >= start_hour_utc) & (df_area['datetime'].dt.hour < end_hour_utc)]
        
        idx_target = df_shift.groupby('datetime_wib')['predicted_v_max'].idxmax()
        df_trend = df_shift.loc[idx_target].sort_values('datetime_wib').reset_index(drop=True)
        title_range = format_rentang_waktu(df_trend['datetime_wib'])
        
        # Fetch file verifikasi dari url atau lokal
        path_verif = get_verif_csv_url()
        try:
            # Tambahkan kunci token agar bisa tembus repo private
            opsi_akses = {"Authorization": f"token {GITHUB_TOKEN}"} if str(path_verif).startswith("http") else None
            df_rr = pd.read_csv(path_verif, storage_options=opsi_akses)
            df_rr['waktu_utc'] = pd.to_datetime(df_rr['Waktu_UTC'])
            df_merge = pd.merge(df_trend[['datetime', 'datetime_wib', 'predicted_v_max']], df_rr[['waktu_utc', 'Real_Max_Speed_ms', 'Arah_Derajat']], left_on='datetime', right_on='waktu_utc', how='left')
        except:
            df_merge = df_trend[['datetime', 'datetime_wib', 'predicted_v_max']].copy()
            df_merge['Real_Max_Speed_ms'], df_merge['Arah_Derajat'] = np.nan, np.nan
            
        waktu_list = df_merge['datetime_wib'].dt.strftime('%H:%M').tolist()
        prediksi_list = [round(float(x), 2) if pd.notna(x) else 0.0 for x in df_merge['predicted_v_max']]
        observasi_list = [round(float(x), 2) if pd.notna(x) else None for x in df_merge['Real_Max_Speed_ms']]
        
        anotasi_blok = []
        for i in range(0, len(df_merge), 3):
            chunk = df_merge.iloc[i:i+3]
            if chunk.empty: continue
            obs_valid = chunk['Real_Max_Speed_ms'].dropna()
            max_v = float(obs_valid.max()) if not obs_valid.empty else None
            
            arah_dom = "-"
            arah_valid = chunk['Arah_Derajat'].dropna()
            if not arah_valid.empty:
                kat_arah = arah_valid.apply(lambda d: "Utara" if (d >= 270 or d < 90) else "Selatan")
                if not kat_arah.mode().empty:
                    arah_asli = kat_arah.mode()[0]
                    arah_dom = "Selatan" if arah_asli == "Utara" else ("Utara" if arah_asli == "Selatan" else "-")
            
            anotasi_blok.append({"xMin": waktu_list[i], "xMax": waktu_list[min(i+3, len(waktu_list)-1)], "xCenter": waktu_list[min(i+1, len(waktu_list)-1)], "max_v": max_v, "arah": arah_dom})
            
        return {"waktu": waktu_list, "prediksi": prediksi_list, "observasi": observasi_list, "anotasi_blok": anotasi_blok, "title_range": title_range}
    except Exception as e:
        print("Error Validation:", e)
        return {"waktu": [], "title_range": ""}

@app.get("/api/rmse-spatial")
def get_rmse_spatial(filename: str = None):
    try:
        files = get_clean_files()
        if not files: return {"speed_heatmap": [], "dir_heatmap": [], "avg_speed": 0, "avg_dir": 0}
        
        target_idx = next((i for i, f in enumerate(files) if os.path.basename(f).split("?")[0] == filename), 0)
        valid_idx = target_idx + 1
        if valid_idx >= len(files): return {"speed_heatmap": [], "dir_heatmap": [], "avg_speed": 0, "avg_dir": 0}
        
        target_file = files[valid_idx]
        df_pred = load_data(target_file)
        
        path_verif = get_verif_csv_url()
        try:
            # Tambahkan kunci token agar bisa tembus repo private
            opsi_akses = {"Authorization": f"token {GITHUB_TOKEN}"} if str(path_verif).startswith("http") else None
            df_obs = pd.read_csv(path_verif, storage_options=opsi_akses)
            df_obs['waktu_utc'] = pd.to_datetime(df_obs['Waktu_UTC'])
        except: return {"speed_heatmap": [], "dir_heatmap": [], "avg_speed": 0, "avg_dir": 0}
        
        df_area = df_pred[(df_pred['lat'] >= -8.2) & (df_pred['lat'] <= -8.0) & (df_pred['lon'] >= 114.3) & (df_pred['lon'] <= 114.5)]
        last_hour = df_area['datetime'].max().hour
        start_hour_utc, end_hour_utc = (12, 24) if last_hour < 12 else (0, 12)
        df_shift = df_area[(df_area['datetime'].dt.hour >= start_hour_utc) & (df_area['datetime'].dt.hour < end_hour_utc)]
        
        df_merged = pd.merge(df_shift, df_obs[['waktu_utc', 'Real_Max_Speed_ms', 'Arah_Derajat']], left_on='datetime', right_on='waktu_utc', how='inner')
        if df_merged.empty: return {"speed_heatmap": [], "dir_heatmap": [], "avg_speed": 0, "avg_dir": 0}
            
        grouped = df_merged.groupby(['lat', 'lon'])
        speed_heatmap, dir_heatmap = [], []
        total_speed_rmse, total_dir_rmse, count = 0, 0, 0
        
        for (lat, lon), group in grouped:
            speed_err_sq = ((group['predicted_v_max'] - group['Real_Max_Speed_ms']) ** 2).mean()
            rmse_speed = float(np.sqrt(speed_err_sq)) if pd.notna(speed_err_sq) else 0
            
            diff = (group['arah_derajat'] - group['Arah_Derajat'] + 180) % 360 - 180
            dir_err_sq = (diff ** 2).mean()
            rmse_dir = float(np.sqrt(dir_err_sq)) if pd.notna(dir_err_sq) else 0
            
            speed_heatmap.append([lat, lon, min(rmse_speed / 0.6, 1.0)])
            dir_heatmap.append([lat, lon, min(rmse_dir / 30.0, 1.0)])
            total_speed_rmse += rmse_speed; total_dir_rmse += rmse_dir; count += 1
            
        return {
            "speed_heatmap": speed_heatmap, "dir_heatmap": dir_heatmap,
            "avg_speed": round(total_speed_rmse / count, 2) if count > 0 else 0,
            "avg_dir": round(total_dir_rmse / count, 1) if count > 0 else 0
        }
    except Exception: return {"speed_heatmap": [], "dir_heatmap": [], "avg_speed": 0, "avg_dir": 0}