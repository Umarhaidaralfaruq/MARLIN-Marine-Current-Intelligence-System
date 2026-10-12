# =============================================================================
# FILE: main.py
# FUNGSI: API Backend MARLIN v2 (FastAPI)
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

# Mengambil variabel konfigurasi (pastikan file config.py tetap ada di folder yang sama)
from config import *

app = FastAPI(title="MARLIN API", description="API Backend untuk Marine Current Intelligence System")

# Setup CORS (SANGAT PENTING!)
# Ini mengizinkan frontend (seperti Netlify/Vercel) untuk mengambil data dari API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Saat production, ganti "*" dengan URL web lo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# FUNGSI HELPER (Diambil dari Marlin V1)
# ---------------------------------------------------------
def get_available_baseruns():
    # Mengambil list file CSV seperti di v1
    return sorted(glob.glob(os.path.join(PATHS["OUTPUT_PRED"], "prediksi_spasial*.csv")), reverse=True)

# Cache sederhana untuk memori lokal
data_cache = {}

def load_data(file_path):
    if file_path in data_cache:
        return data_cache[file_path]
        
    df = pd.read_csv(file_path).rename(columns={
        'datetime_prediction': 'datetime', 
        'latitude': 'lat', 
        'longitude': 'lon', 
        'current_speed_ms': 'predicted_v_max', 
        'v_velocity_ms': 'predicted_v', 
        'u_velocity_ms': 'predicted_u', 
        'current_direction_deg': 'arah_derajat'
    })
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime_wib'] = df['datetime'] + pd.Timedelta(hours=7)
    
    data_cache[file_path] = df
    return df

def ambil_index_target(group):
    # Logika statistik dari v1
    if METODE_STATISTIK == "P95":
        p95 = group['predicted_v_max'].quantile(0.95)
        if pd.isna(p95): return group.index[0]
        return (group['predicted_v_max'] - p95).abs().idxmin()
    return group['predicted_v_max'].idxmax()

def derajat_ke_utara_selatan(deg):
    return "" if pd.isna(deg) else ("Utara" if deg >= 270 or deg < 90 else "Selatan")

# ---------------------------------------------------------
# ENDPOINTS API (Jalur untuk dipanggil Frontend)
# ---------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "MARLIN API System Online. Akses /docs untuk melihat dokumentasi API."}

@app.get("/api/baseruns")
def get_baseruns():
    """Mengembalikan daftar file prediksi yang tersedia."""
    files = get_available_baseruns()
    if not files:
        raise HTTPException(status_code=404, detail="Tidak ada file prediksi ditemukan")
    
    # Format agar mudah dibaca frontend
    hasil = [{"filename": os.path.basename(f), "path": f} for f in files]
    return {"status": "success", "data": hasil}

@app.get("/api/summary")
def get_summary(pelabuhan: str = Query(..., description="Pilih: ketapang atau gilimanuk")):
    """
    Mengembalikan data ringkasan operasional (Kecepatan saat ini & puncak) 
    untuk pelabuhan tertentu (menggantikan summary card Streamlit).
    """
    files = get_available_baseruns()
    if not files:
        raise HTTPException(status_code=404, detail="Data tidak tersedia")
        
    df_spatial = load_data(files[0]) # Ambil file terbaru
    waktu_sekarang_wib = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=7)
    
    # Tentukan area berdasarkan parameter (Logika filter area dari v1)
    if pelabuhan.lower() == "ketapang":
        lat_min, lat_max, lon_min, lon_max = KET_LAT_MIN, KET_LAT_MAX, KET_LON_MIN, KET_LON_MAX
    elif pelabuhan.lower() == "gilimanuk":
        lat_min, lat_max, lon_min, lon_max = GIL_LAT_MIN, GIL_LAT_MAX, GIL_LON_MIN, GIL_LON_MAX
    else:
        raise HTTPException(status_code=400, detail="Pelabuhan tidak valid")

    df_area = df_spatial[(df_spatial['lat'] >= lat_min) & (df_spatial['lat'] <= lat_max) & 
                         (df_spatial['lon'] >= lon_min) & (df_spatial['lon'] <= lon_max)]
                         
    if df_area.empty:
        raise HTTPException(status_code=404, detail="Data area kosong")

    # Logika mencari index target seperti di v1
    idx_target = df_area.groupby('datetime_wib').apply(ambil_index_target)
    df_trend = df_area.loc[idx_target].reset_index(drop=True)
    
    df_trend['selisih_waktu'] = abs(df_trend['datetime_wib'] - waktu_sekarang_wib)
    baris_saat_ini = df_trend.loc[df_trend['selisih_waktu'].idxmin()]
    baris_max = df_trend.loc[df_trend['predicted_v_max'].idxmax()]

    # Kembalikan data dalam format JSON murni, BUKAN HTML/CSS
    return {
        "pelabuhan": pelabuhan.capitalize(),
        "current": {
            "speed_ms": float(baris_saat_ini['predicted_v_max']),
            "direction": derajat_ke_utara_selatan(baris_saat_ini['arah_derajat']),
            "time_wib": baris_saat_ini['datetime_wib'].strftime('%Y-%m-%d %H:%M')
        },
        "peak": {
            "speed_ms": float(baris_max['predicted_v_max']),
            "direction": derajat_ke_utara_selatan(baris_max['arah_derajat']),
            "time_wib": baris_max['datetime_wib'].strftime('%Y-%m-%d %H:%M')
        }
    }

@app.get("/api/spatial-grid")
def get_spatial_grid(time_str: str = Query(None, description="Format YYYY-MM-DD HH:MM")):
    """
    Menghasilkan data JSON grid U dan V yang akan dipakai oleh Leaflet-Velocity di Frontend.
    Menggantikan fungsi konversi_ke_velocity_json di v1.
    """
    files = get_available_baseruns()
    if not files:
        raise HTTPException(status_code=404, detail="Data tidak tersedia")
        
    df_spatial = load_data(files[0])
    
    if time_str:
        df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    else:
        # Jika tidak dikasih waktu, ambil waktu pertama
        waktu_unik = sorted(df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M').unique())
        df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == waktu_unik[0]]

    if df_hour.empty:
        raise HTTPException(status_code=404, detail="Waktu tidak ditemukan di data")

    # Logika konversi grid (Sama persis dengan konversi_ke_velocity_json di V1)
    lo1, lo2 = float(df_hour['lon'].min()), float(df_hour['lon'].max())
    la1, la2 = float(df_hour['lat'].max()), float(df_hour['lat'].min()) 
    nx, ny = GRID_NX, GRID_NY
    grid_lon, grid_lat = np.linspace(lo1, lo2, nx), np.linspace(la1, la2, ny) 
    dx, dy = (lo2 - lo1) / (nx - 1), (la1 - la2) / (ny - 1) 
    
    triang = tri.Triangulation(df_hour['lon'], df_hour['lat'])
    batas_resolusi = BATAS_RESOLUSI
    x_tri, y_tri = triang.x[triang.triangles], triang.y[triang.triangles]
    sisi_a = np.sqrt((x_tri[:, 0] - x_tri[:, 1])**2 + (y_tri[:, 0] - y_tri[:, 1])**2)
    sisi_b = np.sqrt((x_tri[:, 1] - x_tri[:, 2])**2 + (y_tri[:, 1] - y_tri[:, 2])**2)
    sisi_c = np.sqrt((x_tri[:, 2] - x_tri[:, 0])**2 + (y_tri[:, 2] - y_tri[:, 0])**2)
    triang.set_mask(np.any([sisi_a > batas_resolusi, sisi_b > batas_resolusi, sisi_c > batas_resolusi], axis=0))
    
    interp_u = tri.LinearTriInterpolator(triang, df_hour['predicted_u'])
    interp_v = tri.LinearTriInterpolator(triang, df_hour['predicted_v'])
    Grid_lon, Grid_lat = np.meshgrid(grid_lon, grid_lat)
    U_grid, V_grid = interp_u(Grid_lon, Grid_lat), interp_v(Grid_lon, Grid_lat)
    
    u_data, v_data = [], []
    for i in range(ny):
        for j in range(nx):
            u_val, v_val = U_grid[i, j], V_grid[i, j]
            if np.ma.is_masked(u_val) or np.isnan(u_val):
                u_data.append(None); v_data.append(None)
            else:
                u_data.append(float(u_val)); v_data.append(float(v_val))
                
    # Return sebagai list of dictionaries (JSON)
    return [
        {"header": {"parameterCategory": 2, "parameterNumber": 2, "nx": nx, "ny": ny, "lo1": lo1, "lo2": lo2, "la1": la1, "la2": la2, "dx": dx, "dy": dy}, "data": u_data},
        {"header": {"parameterCategory": 2, "parameterNumber": 3, "nx": nx, "ny": ny, "lo1": lo1, "lo2": lo2, "la1": la1, "la2": la2, "dx": dx, "dy": dy}, "data": v_data}
    ]


@app.get("/api/map-background")
def get_map_background(time_str: str = Query(None, description="Format YYYY-MM-DD HH:MM")):
    """Menghasilkan gambar overlay warna gradasi kecepatan arus"""
    files = get_available_baseruns()
    if not files:
        raise HTTPException(status_code=404, detail="Data tidak tersedia")
        
    df_spatial = load_data(files[0])
    
    if time_str:
        df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    else:
        waktu_unik = sorted(df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M').unique())
        df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == waktu_unik[0]]

    if df_hour.empty:
        raise HTTPException(status_code=404, detail="Waktu tidak ditemukan")

    lon_min, lon_max = float(df_hour['lon'].min()), float(df_hour['lon'].max())
    lat_min, lat_max = float(df_hour['lat'].min()), float(df_hour['lat'].max())
    
    # Proses Masking Segitiga (seperti di app.py lama)
    triang = tri.Triangulation(df_hour['lon'], df_hour['lat'])
    x_tri, y_tri = triang.x[triang.triangles], triang.y[triang.triangles]
    batas_resolusi = BATAS_RESOLUSI if 'BATAS_RESOLUSI' in globals() else 0.015
    mask = np.any([np.sqrt((x_tri[:, i] - x_tri[:, (i+1)%3])**2 + (y_tri[:, i] - y_tri[:, (i+1)%3])**2) > batas_resolusi for i in range(3)], axis=0)
    triang.set_mask(mask)
    
    # Proses Menggambar dengan Matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    vmax_plot = 4.0 # Batas max gradasi warna
    cmap_custom = LinearSegmentedColormap.from_list("custom", ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ffa500', '#ff0000', '#4b0082'])
    
    ax.tricontourf(triang, df_hour['predicted_v_max'], levels=np.linspace(0, vmax_plot, 100), cmap=cmap_custom, vmin=0, vmax=vmax_plot, extend='max', alpha=0.9, antialiased=True)
    
    # Ubah gambar jadi teks Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return {
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "image": image_base64
    }