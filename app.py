# =============================================================================
# FILE: app.py
# FUNGSI: Dashboard Interaktif Arus Selat Bali dengan Animasi Arah & Warna Kecepatan
# =============================================================================

import os
import glob
import io
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import plugins
import branca
import base64
from branca.element import Template, MacroElement
from streamlit_folium import st_folium
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime
from pathlib import Path
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.tri as tri
import json
import streamlit.components.v1 as components
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter
from plotly.subplots import make_subplots

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Peta Arus WebGL + Canvas</title>
    <meta charset="utf-8" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-velocity@1.9.0/dist/leaflet-velocity.min.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet-velocity@1.9.0/dist/leaflet-velocity.min.js"></script>
    <style>
        body { margin: 0; padding: 0; }
        #map { width: 100vw; height: 100vh; }
        .custom-div-icon { background: transparent; border: none; }
        /* Animasi Denyut untuk Lokasi Maksimum */
        .pulse-marker {
            background: #ff0000;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 0 0 rgba(255, 0, 0, 0.6);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
            70% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
        }
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        // 1. Inisialisasi Peta (Tema Terang/Positron)
        var map = L.map('map').setView([__LAT_CENTER__, __LON_CENTER__], 12.5);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap &copy; CARTO'
        }).addTo(map);

        // 3. Render Kanvas Warna Background (Image Overlay)
        var imageUrl = 'data:image/png;base64,__IMAGE_BASE64__';
        var imageBounds = [[__LAT_MIN__, __LON_MIN__], [__LAT_MAX__, __LON_MAX__]];
        L.imageOverlay(imageUrl, imageBounds, {opacity: 0.75}).addTo(map);

        // 4. Render Label Angka Spasial
        var labelsData = __LABELS_JSON__;
        labelsData.forEach(function(lbl) {
            var icon = L.divIcon({
                className: 'custom-div-icon',
                html: '<div style="font-size: 11px; color: black; font-weight: bold; text-shadow: 2px 2px 3px white, -2px -2px 3px white;">' + lbl.val + '</div>',
                iconSize: [30, 15],
                iconAnchor: [15, 7]
            });
            L.marker([lbl.lat, lbl.lon], {icon: icon, interactive: false}).addTo(map);
        });

        // 5. Render WebGL Particle (Garis Arus Putih Mulus)
        var velocityLayer = L.velocityLayer({
            displayValues: true,
            displayOptions: {
                velocityType: 'Kecepatan Arus',
                displayPosition: 'bottomleft',
                displayEmptyString: 'Tidak ada data'
            },
            data: __DATA_JSON__,
            maxVelocity: __MAX_VELOCITY__, // UBAH INI
            velocityScale: 0.05,           
            particleAge: 200,
            particleMultiplier: 0.002,     
            colorScale: ['#ffffff']        
        });
        velocityLayer.addTo(map);

        // ========================================================
        // 6. MENAMBAHKAN LEGENDA KUSTOM ALA MARITIM
        // ========================================================
        var legend = L.control({position: 'bottomright'});
        legend.onAdd = function (map) {
            var div = L.DomUtil.create('div', 'info legend');
            // Kotak luar legenda
            div.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
            div.style.padding = '10px 15px';
            div.style.borderRadius = '8px';
            div.style.border = '1px solid #ccc';
            div.style.boxShadow = '2px 2px 5px rgba(0,0,0,0.2)';
            div.style.fontFamily = 'Arial, sans-serif';
            
            // Elemen di dalam legenda (Bentuk kapsul warna)
            div.innerHTML = 
                '<div style="margin-bottom: 8px; font-weight: bold; font-size: 12px; color: black; text-align: center;">Kecepatan Arus (__SATUAN__)</div>' +
                '<div style="background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ffa500, #ff0000, #4b0082); ' +
                'width: 250px; height: 15px; border-radius: 10px; box-shadow: inset 0 0 2px rgba(0,0,0,0.5);"></div>' +
                '<div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: 5px; color: black; font-weight: bold; padding: 0 5px;">' +
                    '<span>0</span>' +
                    '<span>__LEG_1__</span>' +
                    '<span>__LEG_2__</span>' +
                    '<span>__LEG_3__</span>' +
                    '<span>__LEG_4__</span>' +
                '</div>';
            return div;
        };
        legend.addTo(map);
    </script>
</body>
</html>
"""

def konversi_ke_velocity_json(df_hour):
    """
    Mengubah DataFrame Pandas menjadi format JSON standar Windy.com / Leaflet-Velocity.
    """
    # Membulatkan koordinat untuk menghindari bug desimal (floating point)
    df_hour = df_hour.copy()
    df_hour['lon'] = df_hour['lon'].round(4)
    df_hour['lat'] = df_hour['lat'].round(4)
    
    # Mengambil nilai unik grid (Lat harus diurutkan dari Utara ke Selatan / Descending)
    lons = np.sort(df_hour['lon'].unique())
    lats = np.sort(df_hour['lat'].unique())[::-1] 
    
    nx, ny = len(lons), len(lats)
    lo1, lo2 = float(lons[0]), float(lons[-1])
    la1, la2 = float(lats[0]), float(lats[-1])
    
    dx = (lo2 - lo1) / (nx - 1) if nx > 1 else 0
    dy = (la1 - la2) / (ny - 1) if ny > 1 else 0
    
    # Buat dictionary untuk pencarian cepat
    data_dict = df_hour.set_index(['lat', 'lon'])[['predicted_u', 'predicted_v']].to_dict('index')
    
    u_data, v_data = [], []
    
    # Isi array data (Jika di titik itu daratan/NaN, isi dengan 0.0)
    for lat in lats:
        for lon in lons:
            if (lat, lon) in data_dict and not pd.isna(data_dict[(lat, lon)]['predicted_u']):
                u_data.append(float(data_dict[(lat, lon)]['predicted_u']))
                v_data.append(float(data_dict[(lat, lon)]['predicted_v']))
            else:
                u_data.append(0.0)
                v_data.append(0.0)
                
    # Bungkus dalam format standar
    velocity_json = [
        {
            "header": {
                "parameterCategory": 2, "parameterNumber": 2, # Parameter U (Timur-Barat)
                "nx": nx, "ny": ny, "lo1": lo1, "lo2": lo2, "la1": la1, "la2": la2, "dx": dx, "dy": dy
            },
            "data": u_data
        },
        {
            "header": {
                "parameterCategory": 2, "parameterNumber": 3, # Parameter V (Utara-Selatan)
                "nx": nx, "ny": ny, "lo1": lo1, "lo2": lo2, "la1": la1, "la2": la2, "dx": dx, "dy": dy
            },
            "data": v_data
        }
    ]
    return json.dumps(velocity_json)

# ==========================================
# 0. KONFIGURASI TEMA MATPLOTLIB (GAYA BMKG)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Batasan Area Operasional (Ketapang - Gilimanuk)
LAT_MIN, LAT_MAX = -8.209, -8.160
LON_MIN, LON_MAX = 114.387, 114.435

# ==========================================
# 1. SETUP HALAMAN WEBSITE
# ==========================================
ikon_bmkg = Image.open("logo_bmkg.png")

# Kode pelindung: Cek dulu apakah filenya benar-benar ada di server
if os.path.exists("logo_BMKG.png"):
    ikon_bmkg = Image.open("logo_bmkg.png")
    st.set_page_config(page_title="Dashboard Arus Selat Bali", page_icon=ikon_bmkg, layout="wide")
else:
    # Kalau file gagal ditemukan (entah karena salah ketik/beda folder), 
    # pakai emotikon saja sementara biar web tetap nyala dan nggak error!
    st.set_page_config(page_title="Dashboard Arus Selat Bali", page_icon="🌊", layout="wide")

# --- TRIK NINJA: HILANGKAN WATERMARK & MENU STREAMLIT ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
# --------------------------------------------------------

st.title("🌊 Dashboard Prediksi Arus Laut Selat Bali")
st.markdown("Prediksi spasial dan deret waktu 24 jam ke depan berbasis Machine Learning (LSTM).")

# Koordinat Anchor (Sesuai script Anda)
ANCHOR_LAT = -8.20880659
ANCHOR_LON = 114.4111511 

# Sesuaikan dengan lokasi folder penyimpanan CSV dari config Anda
# Ubah path ini jika folder output Anda berbeda
FOLDER_OUTPUT = Path("Arsip_Prediksi")

# ==========================================
# 2. FUNGSI PENCARI & PEMUAT DATA (SISTEM BASERUN)
# ==========================================
def get_available_baseruns():
    # Mencari semua file prediksi spasial
    search_pattern = os.path.join(FOLDER_OUTPUT, "prediksi_spasial*.csv")
    list_of_files = glob.glob(search_pattern)
    
    # Urutkan file berdasarkan waktu pembuatan, yang terbaru di atas
    list_of_files = sorted(list_of_files, reverse=True)
    return list_of_files

@st.cache_data(ttl=3600) 
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # --- TRANSLATOR KOLOM DARI PREDICT.PY KE APP.PY ---
    df = df.rename(columns={
        'datetime_prediction': 'datetime',
        'latitude': 'lat',
        'longitude': 'lon',
        'current_speed_ms': 'predicted_v_max',
        'v_velocity_ms': 'predicted_v',
        'u_velocity_ms': 'predicted_u',
        'current_direction_deg': 'arah_derajat'
    })
    # --------------------------------------------------
    
    # Konversi waktu ke datetime dan ubah ke WIB (+7)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime_wib'] = df['datetime'] + pd.Timedelta(hours=7)
    return df

# ==========================================
# 3. MEMUAT DATA & VALIDASI
# ==========================================
list_of_files = get_available_baseruns()

if not list_of_files:
    st.error(f"❌ Tidak ada file prediksi ditemukan di folder: {FOLDER_OUTPUT}")
    st.stop()

# Membuat dictionary nama file
file_options = {os.path.basename(f): f for f in list_of_files}

st.divider()

# Membuat 3 Kolom untuk area atas (Rasio kolom bisa disesuaikan, ini 1:1:1)
col_m1, col_m2, col_m3 = st.columns(3)

# ---------------------------------------------------------
# LANGKAH A: Tampilkan Dropdown & Pemilih Satuan
# ---------------------------------------------------------
selected_filename = col_m2.selectbox(
    "🗂️ Pilih Baserun Data Prediksi:",
    options=list(file_options.keys())
)

# --- PINDAHKAN PEMILIH SATUAN KE KOLOM 3 (KANAN) ---
satuan_pilihan = col_m3.radio("📏 Pilih Satuan Kecepatan:", ["m/s", "Knot"], horizontal=True)

# Tombol refresh mungil di bawah dropdown (di Kolom 2)
if col_m2.button("🔄 Perbarui Daftar File", use_container_width=True):
    load_data.clear() 
    st.rerun()

# ---------------------------------------------------------
# LANGKAH B: Memuat data sesuai pilihan dari Kolom 2
# ---------------------------------------------------------
selected_filepath = file_options[selected_filename]
df_spatial = load_data(selected_filepath)

# --- Logika Konversi ---
faktor_konversi = 1.94384 if satuan_pilihan == "Knot" else 1.0
satuan_teks = satuan_pilihan
batas_waspada = 1.2 * faktor_konversi 

# Terapkan konversi ke dataframe
df_spatial['predicted_v_max'] = df_spatial['predicted_v_max'] * faktor_konversi
df_spatial['predicted_u'] = df_spatial['predicted_u'] * faktor_konversi
df_spatial['predicted_v'] = df_spatial['predicted_v'] * faktor_konversi

waktu_unik = sorted(df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M').unique())

# ---------------------------------------------------------
# LANGKAH C: Menghitung & Menampilkan Metrik di Kolom 1
# ---------------------------------------------------------
df_filtered_area = df_spatial[
    (df_spatial['lat'] >= LAT_MIN) & (df_spatial['lat'] <= LAT_MAX) &
    (df_spatial['lon'] >= LON_MIN) & (df_spatial['lon'] <= LON_MAX)
]

# --- FUNGSI PENCARI P95 ---
def get_p95_idx(group):
    p95 = group['predicted_v_max'].quantile(0.95)
    if pd.isna(p95): return group.index[0]
    return (group['predicted_v_max'] - p95).abs().idxmin()

idx_p95_trend = df_filtered_area.groupby('datetime_wib').apply(get_p95_idx)
df_trend_p95 = df_filtered_area.loc[idx_p95_trend].reset_index(drop=True)

baris_max = df_trend_p95.loc[df_trend_p95['predicted_v_max'].idxmax()]
max_speed_global = baris_max['predicted_v_max']
waktu_ekstrem = baris_max['datetime_wib'].strftime('%H:%M WIB')

status_teks = "Waspada" if max_speed_global > batas_waspada else "Aman"
keterangan_delta = f"Pukul {waktu_ekstrem} ({status_teks})"

# Menaruh Kartu Kecepatan di Kolom 1 (Kiri)
col_m1.metric(
    label="🌪️ Kecepatan Puncak P95 (24J ke Depan)", 
    value=f"{max_speed_global:.2f} {satuan_teks}", 
    delta=keterangan_delta,
    delta_color="inverse" if max_speed_global > batas_waspada else "normal"
)

# HAPUS BARIS INI (Keterangan Total Waktu Prediksi yang lama):
# col_m3.metric(label="⏱️ Total Waktu Prediksi", value=f"{len(waktu_unik)} Jam")

st.divider()

def hitung_vorticity(df_hour):
    """
    Menghitung pusaran (vorticity) dari komponen arus U dan V.
    """
    # 1. Bentuk ulang data menjadi matriks 2D (Grid Lintang x Bujur)
    grid_u = df_hour.pivot_table(index='lat', columns='lon', values='predicted_u')
    grid_v = df_hour.pivot_table(index='lat', columns='lon', values='predicted_v')
    
    # 2. Faktor konversi derajat lintang/bujur ke meter (Perkiraan di Selat Bali)
    # 1 derajat lintang ~ 111.32 km. 1 derajat bujur di -8.2 ~ 110.18 km.
    dx_meter = 110180
    dy_meter = 111320
    
    # 3. Hitung selisih kecepatan antar titik (Gradien spasial)
    # Gunakan np.gradient untuk menghitung turunan parsial
    du_dy, du_dx = np.gradient(grid_u.values)
    dv_dy, dv_dx = np.gradient(grid_v.values)
    
    # Bagi dengan jarak aslinya dalam meter
    du_dy = du_dy / dy_meter
    dv_dx = dv_dx / dx_meter
    
    # 4. Hitung Vorticity (Rotasi Z)
    vorticity_grid = dv_dx - du_dy
    
    # Kembalikan formatnya menjadi list/tabel seperti aslinya
    df_vort = pd.DataFrame(vorticity_grid, index=grid_u.index, columns=grid_u.columns)
    df_vort = df_vort.reset_index().melt(id_vars='lat', value_name='vorticity')
    
    # Gabungkan ke dataframe asal, dan ambil nilai absolutnya
    # (Karena kita hanya peduli kekuatan pusarannya, bukan arah putarannya)
    df_result = pd.merge(df_hour, df_vort, on=['lat', 'lon'], how='left')
    df_result['vort_abs'] = df_result['vorticity'].abs()
    
    return df_result

# ==========================================
# 5. LAYOUT UTAMA (PETA & GRAFIK)
# ==========================================
col1, col2 = st.columns([1.1, 1]) 

# ========================================================
# FUNGSI CACHE UNTUK BACKGROUND PETA (MATPLOTLIB + PIL)
# ========================================================
@st.cache_data(show_spinner=False, ttl=3600)
def generate_map_background(df_hour_json, lat_min_data, lat_max_data, lon_min_data, lon_max_data, vmax_plot):
    # Kita menggunakan JSON/Dict untuk parameter cache agar hashing Streamlit lebih cepat
    df_hour = pd.DataFrame(df_hour_json)
    
    triang = tri.Triangulation(df_hour['lon'], df_hour['lat'])
    batas_resolusi = 0.010 
    
    x_tri = triang.x[triang.triangles]
    y_tri = triang.y[triang.triangles]
    sisi_a = np.sqrt((x_tri[:, 0] - x_tri[:, 1])**2 + (y_tri[:, 0] - y_tri[:, 1])**2)
    sisi_b = np.sqrt((x_tri[:, 1] - x_tri[:, 2])**2 + (y_tri[:, 1] - y_tri[:, 2])**2)
    sisi_c = np.sqrt((x_tri[:, 2] - x_tri[:, 0])**2 + (y_tri[:, 2] - y_tri[:, 0])**2)
    
    mask = np.any([sisi_a > batas_resolusi, sisi_b > batas_resolusi, sisi_c > batas_resolusi], axis=0)
    triang.set_mask(mask)

    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ['blue', 'green', 'yellow', 'orange', 'red'])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    ax.set_xlim(lon_min_data, lon_max_data) 
    ax.set_ylim(lat_min_data, lat_max_data)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    # OPTIMASI: Turunkan levels dari 10000 ke 1000. 
    # Secara visual tidak ada bedanya, tapi rendering 10x lebih cepat!
    ax.tricontourf(
        triang, df_hour['predicted_v_max'], 
        levels=np.linspace(0.0, vmax_plot, 1000), 
        cmap=custom_cmap, vmin=0.0, vmax=vmax_plot, extend='max', alpha=0.9
    )

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    plt.close(fig)

    # Membulatkan sudut dengan PIL
    img = Image.open(buf)
    r, g, b, a = img.split()
    a_blur = a.filter(ImageFilter.GaussianBlur(radius=10))
    a_thresh = a_blur.point(lambda p: 255 if p > 120 else 0)
    a_final = a_thresh.filter(ImageFilter.GaussianBlur(radius=1))
    img_rounded = Image.merge('RGBA', (r, g, b, a_final))

    buf_final = io.BytesIO()
    img_rounded.save(buf_final, format='PNG')
    return base64.b64encode(buf_final.getvalue()).decode('utf-8')

with col1:
    st.subheader("🗺️ Peta Spasial Arus")
    
    # --------------------------------------------------------
    # 1. Slider Navigasi Waktu 
    # --------------------------------------------------------
    st.markdown("**⏱️ Pilih Waktu Prediksi:**")
    
    # --- PERBAIKAN KODE: KONDISIONAL JAM TERAKHIR ---
    # 1. Ambil waktu saat ini, jadikan tz-naive, tambah 7 jam (WIB), dan bulatkan ke jam (floor)
    waktu_sekarang_wib = (pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=7)).floor('h')
    
    # 2. Ubah formatnya menjadi string agar sama persis dengan format di waktu_unik
    waktu_sekarang_str = waktu_sekarang_wib.strftime('%Y-%m-%d %H:%M')
    
    # 3. Logika penentuan jam default
    if waktu_sekarang_str in waktu_unik:
        # Jika jam saat ini tersedia di data, gunakan jam tersebut
        waktu_default = waktu_sekarang_str
    else:
        # Jika jam saat ini tidak tersedia (misal baserun lama), ambil waktu paling akhir di daftar
        waktu_default = waktu_unik[-1]
    # ------------------------------------------------------------

    selected_time_str = st.select_slider(
        "Geser untuk mengubah waktu", 
        options=waktu_unik,
        value=waktu_default,
        label_visibility="collapsed"
    )
    
    st.markdown(f"<h4 style='text-align: center; margin-top: -10px; color: white;'>{selected_time_str} WIB</h4>", unsafe_allow_html=True)
    
    df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == selected_time_str]
    
    lon_min_data, lon_max_data = float(df_hour['lon'].min()), float(df_hour['lon'].max())
    lat_min_data, lat_max_data = float(df_hour['lat'].min()), float(df_hour['lat'].max())

    # --------------------------------------------------------
    # 2. Panggil Fungsi Caching Gambar Background
    # --------------------------------------------------------
    df_hour_dict = df_hour[['lat', 'lon', 'predicted_v_max']].to_dict('list')
    vmax_dinamis = 3.0 * faktor_konversi
    image_base64 = generate_map_background(df_hour_dict, lat_min_data, lat_max_data, lon_min_data, lon_max_data, vmax_dinamis)

    # --------------------------------------------------------
    # 3. Label Spasial Merata
    # --------------------------------------------------------
    placed_labels = []
    labels_list = []
    MIN_LABEL_DIST = 0.01

    for idx, row in df_hour.iterrows():
        kec = row['predicted_v_max']
        if pd.isna(kec): continue
        lat, lon = float(row['lat']), float(row['lon'])
        
        too_close = False
        for (plat, plon) in placed_labels:
            if np.sqrt((lat - plat)**2 + (lon - plon)**2) < MIN_LABEL_DIST:
                too_close = True
                break
                
        if not too_close:
            labels_list.append({"lat": lat, "lon": lon, "val": f"{kec:.2f}"})
            placed_labels.append((lat, lon))

    # ========================================================
    # 4. RENDER PETA HTML (VERSI BERSIH & STABIL)
    # ========================================================
    lat_tengah = (lat_min_data + lat_max_data) / 2
    lon_tengah = (lon_min_data + lon_max_data) / 2
    
    html_siap_render = HTML_TEMPLATE.replace("__DATA_JSON__", konversi_ke_velocity_json(df_hour))
    html_siap_render = html_siap_render.replace("__LABELS_JSON__", json.dumps(labels_list))
    html_siap_render = html_siap_render.replace("__IMAGE_BASE64__", image_base64)
    html_siap_render = html_siap_render.replace("__LAT_MIN__", str(lat_min_data))
    html_siap_render = html_siap_render.replace("__LON_MIN__", str(lon_min_data))
    html_siap_render = html_siap_render.replace("__LAT_MAX__", str(lat_max_data))
    html_siap_render = html_siap_render.replace("__LON_MAX__", str(lon_max_data))
    html_siap_render = html_siap_render.replace("__LAT_CENTER__", str(lat_tengah))
    html_siap_render = html_siap_render.replace("__LON_CENTER__", str(lon_tengah))
    html_siap_render = html_siap_render.replace("__MAX_VELOCITY__", str(8.0 * faktor_konversi))
    html_siap_render = html_siap_render.replace("__SATUAN__", satuan_teks)
    html_siap_render = html_siap_render.replace("__LEG_1__", f"{round(1.0 * faktor_konversi)}")
    html_siap_render = html_siap_render.replace("__LEG_2__", f"{round(2.0 * faktor_konversi)}")
    html_siap_render = html_siap_render.replace("__LEG_3__", f"{round(3.0 * faktor_konversi)}")
    html_siap_render = html_siap_render.replace("__LEG_4__", f"{round(4.0 * faktor_konversi)}")
    
    components.html(html_siap_render, height=650)
    
    st.caption(f"🗺️ **Panduan:** Warna background menunjukkan kecepatan ({satuan_teks}). Animasi garis putih menunjukkan arah arus.")

with col2:
    
    def derajat_ke_panah(deg):
        if pd.isna(deg): 
            return ""
        # Mengonversi derajat (0-360) menjadi simbol panah mata angin instan
        panah = ['↑ U', '↗ TL', '→ T', '↘ TG', '↓ S', '↙ BD', '← B', '↖ BL']
        idx = int((deg + 22.5) % 360 // 45)
        return panah[idx]
        
    def derajat_ke_arah_lengkap(deg):
        if pd.isna(deg): 
            return ""
        # Menerjemahkan derajat menjadi nama arah mata angin lengkap dalam bahasa Indonesia
        arah = ['Utara', 'Timur Laut', 'Timur', 'Tenggara', 'Selatan', 'Barat Daya', 'Barat', 'Barat Laut']
        idx = int((deg + 22.5) % 360 // 45)
        return arah[idx]
    
    st.subheader("📈 Kecepatan & Arah Arus 24 Jam Ke Depan")
    
    # Mengambil dari data P95 yang sudah dikalkulasi di atas
    df_trend = df_trend_p95.copy()
    
    # Tambahkan kolom panah arah mata angin berdasarkan derajatnya
    df_trend['arah_panah'] = df_trend['arah_derajat'].apply(derajat_ke_panah)
    
    # Plotly Line Chart Utama (24 Jam) dengan Modul Teks Panah
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_trend['datetime_wib'], 
        y=df_trend['predicted_v_max'],
        mode='lines+markers+text',          # Aktifkan fitur teks di kurva
        name='Arus Maksimum',
        line=dict(color='black', width=2),
        marker=dict(size=8, color='crimson'),
        text=df_trend['arah_panah'],        # Set panah sebagai teks tampil
        textposition="top center",          # Posisikan persis di atas lingkaran merah
        hovertemplate=(
            "<b>Waktu:</b> %{x}<br>"
            "<b>Kecepatan:</b> %{y:.2f} " + satuan_teks + "<br>"
            "<b>Arah Dominan:</b> %{text}<extra></extra>"
        )
    ))
    
    # Garis Batas Waspada
    fig.add_hline(y=batas_waspada, line_dash="dash", line_color="orange", annotation_text=f"Batas Waspada ({batas_waspada:.2f} {satuan_teks})")
    
    
    fig.update_layout(
        xaxis_title="Waktu (WIB)",
        yaxis_title=f"Kecepatan Maksimum ({satuan_teks})",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # FOKUS OPERASIONAL SHIFT JAGA (12 JAM)
    # ==========================================
    # Fungsi penentu shift menggunakan waktu UTC asli Anda
    def get_shift_range_now(df_prediksi_24jam):
        last_time = df_prediksi_24jam['datetime'].max()
        last_hour = last_time.hour 
        if last_hour < 12: 
            return 12, 24, "Shift Malam"
        else: 
            return 0, 12, "Shift Pagi"

    start_hour_utc, end_hour_utc, shift_name = get_shift_range_now(df_spatial)
    
    mask_shift = (df_spatial['datetime'].dt.hour >= start_hour_utc) & (df_spatial['datetime'].dt.hour < end_hour_utc)
    df_shift_raw = df_spatial.loc[mask_shift & 
                                  (df_spatial['lat'] >= LAT_MIN) & (df_spatial['lat'] <= LAT_MAX) &
                                  (df_spatial['lon'] >= LON_MIN) & (df_spatial['lon'] <= LON_MAX)]

    st.divider()
    st.subheader(f"🎯 Kecepatan & Arah Arus 12 Jam Ke Depan ({shift_name})")

    if not df_shift_raw.empty:
        # 1. Ambil data spasial P95 per jam untuk tren shift
        idx_p95_shift = df_shift_raw.groupby('datetime_wib').apply(get_p95_idx)
        df_shift = df_shift_raw.loc[idx_p95_shift].reset_index(drop=True)
        
        # Mapping arah panah untuk grafik dan arah lengkap untuk teks narasi
        df_shift['arah_panah'] = df_shift['arah_derajat'].apply(derajat_ke_panah)
        df_shift['arah_lengkap'] = df_shift['arah_derajat'].apply(derajat_ke_arah_lengkap)

        # ========================================================
        # BARU: HITUNG ARAH DOMINAN (MODE) SEPANJANG SHIFT
        # ========================================================
        def penentu_utara_selatan(deg):
            if pd.isna(deg): 
                return None
            # Mengelompokkan derajat: 270° - 89.9° (Utara) dan 90° - 269.9° (Selatan)
            if deg >= 270 or deg < 90:
                return "utara"
            else:
                return "selatan"

        # Membuat kolom sementara untuk mendeteksi utara/selatan
        df_shift['arah_bipolar'] = df_shift['arah_derajat'].apply(penentu_utara_selatan)

        if not df_shift['arah_bipolar'].empty:
            # .mode()[0] akan mengambil arah (utara/selatan) yang paling sering muncul
            arah_dominan_shift = df_shift['arah_bipolar'].mode()[0]
        else:
            arah_dominan_shift = "tidak diketahui"

        # 2. Cari data waktu dan nilai kecepatan absolut tertinggi di shift ini
        baris_max_shift = df_shift.loc[df_shift['predicted_v_max'].idxmax()]
        max_speed_shift = baris_max_shift['predicted_v_max']
        waktu_max_shift = baris_max_shift['datetime_wib'].strftime('%H:%M WIB')
        
        # ========================================================
        # 3. NARASI PERINGATAN DINAMIS (SESUAI REQUEST)
        # ========================================================
        if max_speed_shift > 1.2:
            st.error(
                f"🚨 **PERINGATAN DINI** \n"
                f"Kecepatan Arus Maksimal **{max_speed_shift:.2f} {satuan_teks}** diprediksi terjadi pada pukul **{waktu_max_shift}** "
                f"dengan arah arus dominan bergerak ke **{arah_dominan_shift}**."
            )
        else:
            st.success(
                f"✅ **ARUS AMAN** \n"
                f"Kecepatan Arus Maksimal hanya **{max_speed_shift:.2f} {satuan_teks}** pada pukul **{waktu_max_shift}** "
                f"dengan arah arus dominan bergerak ke **{arah_dominan_shift}**."
            )
            
        # Grafik Zoom Khusus Shift dengan Teks Panah
        fig_shift = go.Figure()
        fig_shift.add_trace(go.Scatter(
            x=df_shift['datetime_wib'], 
            y=df_shift['predicted_v_max'],
            mode='lines+markers+text',
            name=f'Arus {shift_name}',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='crimson', line=dict(color='black', width=1)),
            text=df_shift['arah_panah'],
            textposition="top center",
            hovertemplate=(
                "<b>Jam Operasional:</b> %{x}<br>"
                "<b>Kecepatan:</b> %{y:.2f} " + satuan_teks + "<br>"
                "<b>Arah:</b> %{text}<extra></extra>"
            )
        ))
        
        fig_shift.add_hline(y=batas_waspada, line_dash="dash", line_color="orange", annotation_text=f"Batas Waspada ({batas_waspada:.2f} {satuan_teks})")
        
        fig_shift.add_annotation(
            x=baris_max_shift['datetime_wib'],
            y=max_speed_shift,
            text=f"Maks: {max_speed_shift:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowcolor="red",
            ax=0,
            ay=-40,
            font=dict(color="red", weight="bold")
        )
        
        fig_shift.update_layout(
            xaxis_title="Jam Operasional (WIB)",
            yaxis_title=f"Kecepatan Maksimum ({satuan_teks})",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        
        st.plotly_chart(fig_shift, use_container_width=True)
    else:
        st.info(f"Data prediksi untuk {shift_name} belum tersedia di jendela waktu saat ini.")
        
# ==========================================
# 5. GENERATOR LAPORAN BMKG (SIAP DOWNLOAD)
# ==========================================
st.divider()
st.subheader("🖨️ Download Grafik")

def create_bmkg_report(df_source, title_text):
    df_plot = df_source.copy()
    if 'datetime_utc' not in df_plot.columns:
        df_plot['datetime_utc'] = df_plot['datetime_wib'] - pd.Timedelta(hours=7)
        
    # KUNCI SEJAJAR & ADA MARGIN:
    x_min_asli = df_plot['datetime_utc'].min()
    x_max_asli = df_plot['datetime_utc'].max()
    x_min_buffer = x_min_asli - pd.Timedelta(hours=1)
    x_max_buffer = x_max_asli + pd.Timedelta(hours=1)
    extended_ticks_utc = pd.date_range(start=x_min_buffer, end=x_max_buffer, freq='h')

    # Inisialisasi Kanvas Gambar 
    fig_bmkg, ax0 = plt.subplots(figsize=(14, 7), dpi=100)
    fig_bmkg.subplots_adjust(left=0.08, right=0.95, top=0.7, bottom=0.15) 

    line_color = 'black' 
    grid_color = '#A9CCE3' 
    
    # 1. Judul Grafik
    ax0.set_title(title_text, fontsize=16, pad=60)
    
    # 2. Plotting Data Arus
    ax0.plot(df_plot['datetime_utc'], df_plot['predicted_v_max'], color=line_color, linewidth=2, marker='o', markersize=6, markerfacecolor=line_color, markeredgecolor='white')

    # 3. Kustomisasi Sumbu Y (Level Kecepatan)
    ax0.set_ylabel(f"Kecepatan Arus ({satuan_teks})", fontsize=13, weight='bold')
    y_max = max(2.0 * faktor_konversi, df_plot['predicted_v_max'].max() + (0.5 * faktor_konversi))
    ax0.set_ylim(0, y_max)
    ax0.grid(color=grid_color, linestyle='--', linewidth=0.8)

    # 4. Kustomisasi Sumbu X BAWAH (Waktu UTC)
    ax0.set_xlim(x_min_buffer, x_max_buffer)
    ax0.set_xlabel("Waktu (UTC)", fontsize=13, weight='bold', labelpad=10)
    ax0.set_xticks(extended_ticks_utc)
    ax0.set_xticklabels([t.strftime('%H:%M') for t in extended_ticks_utc], rotation=0, ha='center', fontsize=11, weight='bold')

    # 5. Kustomisasi Sumbu X ATAS (Waktu WIB) - TRICK BMKG
    ax0_twin = ax0.twiny()
    ax0_twin.set_xlabel("Waktu (WIB)", fontsize=13, weight='bold', labelpad=10)
    ax0_twin.set_xlim(x_min_buffer, x_max_buffer)
    ax0_twin.set_xticks(extended_ticks_utc)
    
    extended_ticks_wib = extended_ticks_utc + pd.Timedelta(hours=7)
    ax0_twin.set_xticklabels([t.strftime('%H:%M') for t in extended_ticks_wib], rotation=0, ha='center', fontsize=11, weight='bold')

    # 6. Menambahkan Label Angka dan Panah Arah di Setiap Titik 
    for idx, row in df_plot.iterrows():
        # KHUSUS LAPORAN BMKG: Cek derajatnya dan ubah jadi Utara/Selatan saja
        deg = row.get('arah_derajat', pd.NA)
        if pd.isna(deg):
            panah_bmkg = ""
        elif deg >= 270 or deg < 90:
            panah_bmkg = "↑ U"
        else:
            panah_bmkg = "↓ S"
            
        label_text = f"{panah_bmkg}\n{row['predicted_v_max']:.2f}"
        
        # xytext digeser ke (0, 15) agar teks dua baris tidak menabrak garis plot
        ax0.annotate(label_text, (row['datetime_utc'], row['predicted_v_max']), textcoords="offset points", xytext=(0,15), ha='center', fontsize=10, weight='bold')

    # 7. Metadata Header BMKG
    header_text_left = "Kantor Layanan Informasi Meteorologi\nMaritim Ketapang, BMKG\nBMKG"
    fig_bmkg.text(0.11, 0.95, header_text_left, fontsize=12, weight='bold', color='black', ha='left', va='top', linespacing=1.4)
    
    # --- PERUBAHAN: Translasi Manual Bulan ke Bahasa Indonesia ---
    bulan_indo = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni',
        7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    
    issue_time_wib = datetime.now() 
    hari = issue_time_wib.strftime('%d')
    bulan = bulan_indo[issue_time_wib.month]
    tahun = issue_time_wib.strftime('%Y')
    
    issue_date_str = f"{hari} {bulan} {tahun}"
    issue_time_str = issue_time_wib.strftime('%H:%M WIB')
    
    fig_bmkg.text(0.95, 0.95, f"Dikeluarkan pada {issue_time_str}, {issue_date_str}", fontsize=12, weight='bold', color='black', ha='right', va='top')

    # 8. Logo BMKG 
    LOGO_PATH = "logo_BMKG.png"
    if os.path.exists(LOGO_PATH):
        try:
            arr_logo = plt.imread(LOGO_PATH)
            imagebox = OffsetImage(arr_logo, zoom=0.08) 
            logo_loc = AnnotationBbox(imagebox, (0.02, 0.97), xycoords='figure fraction', frameon=False, box_alignment=(0, 1))
            ax0.add_artist(logo_loc)
        except:
            pass

    return fig_bmkg

# ==========================================
# TOMBOL RENDER & DOWNLOAD (BERDAMPINGAN)
# ==========================================
col_rep1, col_rep2 = st.columns(2)

# ----- LAPORAN 12 JAM (SHIFT) -----
with col_rep1:
    st.info(f"📋 **Grafik 12 Jam ({shift_name})**")
    if st.button("📊 Render Grafik 12 Jam", use_container_width=True):
        with st.spinner("Membuat grafik 12 jam..."):
            fig_12j = create_bmkg_report(df_shift, f"Prediksi Kecepatan Arus Ketapang-Gilimanuk 12 Jam Kedepan")
            st.pyplot(fig_12j)
            
            with io.BytesIO() as buf:
                fig_12j.savefig(buf, format="png", dpi=150)
                st.download_button(
                    label="📥 Download PNG (12 Jam)",
                    data=buf.getvalue(),
                    file_name=f"Grafik_Arus_12J_{shift_name.replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            plt.close(fig_12j)

# ----- LAPORAN 24 JAM (FULL) -----
with col_rep2:
    st.success("📋 **Grafik 24 Jam (Full Trend)**")
    if st.button("📊 Render Grafik 24 Jam", use_container_width=True):
        with st.spinner("Membuat grafik 24 jam..."):
            fig_24j = create_bmkg_report(df_trend, "Prediksi Kecepatan Arus Ketapang-Gilimanuk 24 Jam Kedepan")
            st.pyplot(fig_24j)
            
            with io.BytesIO() as buf:
                fig_24j.savefig(buf, format="png", dpi=150)
                st.download_button(
                    label="📥 Download PNG (24 Jam)",
                    data=buf.getvalue(),
                    file_name="Grafik_Arus_24J.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            plt.close(fig_24j)