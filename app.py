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
LAT_MIN, LAT_MAX = -8.209, -8.160084
LON_MIN, LON_MAX = 114.387089, 114.435215

# ==========================================
# 1. SETUP HALAMAN WEBSITE
# ==========================================

# --- PERBAIKAN: GUNAKAN HURUF KECIL UNTUK SEMUA NAMA LOGO ---
logo_filename = "Logo_BMKG.png"
logo_seacast = "logo_bmkg2.png"

# Pengecekan file secara aman
if os.path.exists(logo_seacast):
    ikon = Image.open(logo_seacast)
elif os.path.exists("logo_BMKG.png"):
    ikon = Image.open("logo_BMKG.png")
else:
    ikon = "🌊"

st.set_page_config(page_title="MARLIN", page_icon=ikon, layout="wide")

# --- TRIK NINJA: HILANGKAN WATERMARK, MENU, DAN BADGE HOSTING STREAMLIT ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            /* Menghilangkan tombol melayang Hosted with Streamlit di pojok kanan bawah */
            .viewerBadge {display: none !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
# --------------------------------------------------------

def get_base64_logo(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None

logo_base64 = get_base64_logo(logo_seacast)

if logo_base64:
    # Render HTML: RATA KIRI SEPENUHNYA agar tidak dibaca sebagai blok kode
    html_header = f"""<div style="display: flex; flex-direction: row; align-items: center; margin-bottom: 25px;">
<img src="data:image/png;base64,{logo_base64}" style="height: 110px; width: auto; margin-right: 25px;">
<div style="display: flex; flex-direction: column; justify-content: center;">
<h1 style="margin: 0; padding: 0; font-size: 2.8rem; font-weight: bold; line-height: 1.1;">MARLIN</h1>
<h3 style="margin: 5px 0 0 0; padding: 0; font-size: 1.4rem; font-weight: 500; opacity: 0.9;">Marine Current Intelligence System</h3>
<p style="margin: 10px 0 0 0; font-size: 1.1rem; opacity: 0.8; width: 100%;">Sistem Prediksi Arus Laut Berbasis Machine Learning di Perairan Selat Bali.</p>
</div>
</div>"""
    
    st.markdown(html_header, unsafe_allow_html=True)
else:
    st.markdown("<h1 style='margin-bottom: 0;'>MARLIN</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0; opacity: 0.9;'>Marine Current Intelligence System</h3>", unsafe_allow_html=True)
    st.markdown("Sistem Prediksi Arus Laut Berbasis Machine Learning di Perairan Selat Bali.")

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

# ========================================================
# KODE BARU: STATUS UPDATE BERDASARKAN ISI DATA (BASERUN)
# Ditaruh di sini agar df_spatial sudah terbaca oleh sistem
# ========================================================
waktu_baserun_wib = df_spatial['datetime_wib'].min() - pd.Timedelta(hours=1)
waktu_sekarang_wib = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=7)

selisih_jam = (waktu_sekarang_wib - waktu_baserun_wib).total_seconds() / 3600

if selisih_jam <= 24: 
    status_icon = "🟢"
    status_teks = "Terupdate"
    teks_warna = "#28a745" # Hijau
else:
    status_icon = "🔴"
    status_teks = "Tidak Update"
    teks_warna = "#dc3545" # Merah

# Kita tetap melempar render UI-nya ke Kolom 3 (col_m3)
col_m3.markdown(
    f"<div style='margin-top: 10px; font-size: 14px; font-weight: bold; color: {teks_warna};'>"
    f"{status_icon} Status: {status_teks}</div>", 
    unsafe_allow_html=True
)
col_m3.caption(f"📅 Data Terakhir Tersedia: {waktu_baserun_wib.strftime('%d-%m-%Y %H:%M')} WIB")
# ========================================================


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
    label="🌪️ Kecepatan Arus Maksimal 24 Jam ke Depan", 
    value=f"{max_speed_global:.2f} {satuan_teks}", 
    delta=keterangan_delta,
    delta_color="inverse" if max_speed_global > batas_waspada else "normal"
)

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
    
    st.caption(f"🗺️ **Panduan:** Arahkan kursor ke area yang diinginkan untuk melihat kecepatan dan arah arus di area tersebut.")

with col2:
    
    def derajat_ke_panah(deg):
        if pd.isna(deg): 
            return ""
        panah = ['↑ U', '↗ TL', '→ T', '↘ TG', '↓ S', '↙ BD', '← B', '↖ BL']
        idx = int((deg + 22.5) % 360 // 45)
        return panah[idx]
        
    def derajat_ke_arah_lengkap(deg):
        if pd.isna(deg): 
            return ""
        arah = ['Utara', 'Timur Laut', 'Timur', 'Tenggara', 'Selatan', 'Barat Daya', 'Barat', 'Barat Laut']
        idx = int((deg + 22.5) % 360 // 45)
        return arah[idx]
    
    # =========================================================================
    # DATA 24 JAM TETAP DISIMPAN (TIDAK DITAMPILKAN) UNTUK TOMBOL DOWNLOAD
    # =========================================================================
    df_trend = df_trend_p95.copy()
    df_trend['arah_panah'] = df_trend['arah_derajat'].apply(derajat_ke_panah)
    
    # ==========================================
    # 1. GRAFIK FOKUS OPERASIONAL SHIFT JAGA (12 JAM) - NAIK KE ATAS
    # ==========================================
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

    st.subheader(f"📈 Kecepatan & Arah Arus 12 Jam Ke Depan ({shift_name})")

    if not df_shift_raw.empty:
        idx_p95_shift = df_shift_raw.groupby('datetime_wib').apply(get_p95_idx)
        df_shift = df_shift_raw.loc[idx_p95_shift].reset_index(drop=True)
        
        df_shift['arah_panah'] = df_shift['arah_derajat'].apply(derajat_ke_panah)
        df_shift['arah_lengkap'] = df_shift['arah_derajat'].apply(derajat_ke_arah_lengkap)

        def penentu_utara_selatan(deg):
            if pd.isna(deg): return None
            if deg >= 270 or deg < 90: return "utara"
            else: return "selatan"

        df_shift['arah_bipolar'] = df_shift['arah_derajat'].apply(penentu_utara_selatan)

        if not df_shift['arah_bipolar'].empty:
            arah_dominan_shift = df_shift['arah_bipolar'].mode()[0]
        else:
            arah_dominan_shift = "tidak diketahui"

        baris_max_shift = df_shift.loc[df_shift['predicted_v_max'].idxmax()]
        max_speed_shift = baris_max_shift['predicted_v_max']
        waktu_max_shift = baris_max_shift['datetime_wib'].strftime('%H:%M WIB')
        
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

        # ==========================================
        # 2. TABEL ARUS KETAPANG & GILIMANUK (MENGISI POSISI LAMA)
        # ==========================================
        st.divider()
        st.subheader(f"📊 Kecepatan Arus ({shift_name}) Sekitar Pelabuhan")

        LAT_MIN_KET, LAT_MAX_KET = -8.209161, -8.151693
        LON_MIN_KET, LON_MAX_KET = 114.38673, 114.414975

        LAT_MIN_GIL, LAT_MAX_GIL = -8.21003, -8.16198
        LON_MIN_GIL, LON_MAX_GIL = 114.41354, 114.43416

        # FUNGSI BARU: Pembulatan murni ke Utara / Selatan
        def derajat_ke_utara_selatan(deg):
            if pd.isna(deg): return ""
            if deg >= 270 or deg < 90:
                return "Utara"
            else:
                return "Selatan"

        def buat_tabel_area(df_raw, lat_min, lat_max, lon_min, lon_max):
            df_area = df_raw[
                (df_raw['lat'] >= lat_min) & (df_raw['lat'] <= lat_max) &
                (df_raw['lon'] >= lon_min) & (df_raw['lon'] <= lon_max)
            ]
            if df_area.empty: return pd.DataFrame()
                
            idx_p95_area = df_area.groupby('datetime_wib').apply(get_p95_idx)
            df_trend_area = df_area.loc[idx_p95_area].reset_index(drop=True)
            
            tabel = pd.DataFrame({
                'Waktu (WIB)': df_trend_area['datetime_wib'].dt.strftime('%H:%M'),
                # Biarkan berbentuk float di sini agar logika warna peringatan tidak error
                f'Kecepatan ({satuan_teks})': df_trend_area['predicted_v_max'], 
                'Arah Dominan': df_trend_area['arah_derajat'].apply(derajat_ke_utara_selatan) # Panggil fungsi baru
            })
            return tabel

        df_shift_full = df_spatial.loc[mask_shift]
        tabel_ketapang = buat_tabel_area(df_shift_full, LAT_MIN_KET, LAT_MAX_KET, LON_MIN_KET, LON_MAX_KET)
        tabel_gilimanuk = buat_tabel_area(df_shift_full, LAT_MIN_GIL, LAT_MAX_GIL, LON_MIN_GIL, LON_MAX_GIL)

        col_ket, col_gil = st.columns(2)
        
        # Logika warna merah untuk waspada
        def style_kecepatan(val):
            return 'color: red; font-weight: bold;' if val > batas_waspada else ''

        with col_ket:
            st.markdown("**🚢 Area Pelabuhan Ketapang**")
            if not tabel_ketapang.empty:
                # Format 2 angka di belakang koma dilakukan di level Styler UI
                styled_ket = tabel_ketapang.style.format({f'Kecepatan ({satuan_teks})': "{:.2f}"}).map(
                    style_kecepatan, subset=[f'Kecepatan ({satuan_teks})']
                )
                st.dataframe(styled_ket, hide_index=True, use_container_width=True)
            else:
                st.info("Data tidak tersedia.")

        with col_gil:
            st.markdown("**🚢 Area Pelabuhan Gilimanuk**")
            if not tabel_gilimanuk.empty:
                styled_gil = tabel_gilimanuk.style.format({f'Kecepatan ({satuan_teks})': "{:.2f}"}).map(
                    style_kecepatan, subset=[f'Kecepatan ({satuan_teks})']
                )
                st.dataframe(styled_gil, hide_index=True, use_container_width=True)
            else:
                st.info("Data tidak tersedia.")

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

    # --- PERBAIKAN PENTING: PENGGUNAAN LOGO YANG AMAN DI MATPLOTLIB ---
    if os.path.exists(logo_filename): # logo_filename sekarang sudah diset seragam di atas menjadi "logo_bmkg.png"
        try:
            arr_logo = plt.imread(logo_filename)
            imagebox = OffsetImage(arr_logo, zoom=0.08) 
            logo_loc = AnnotationBbox(imagebox, (0.02, 0.97), xycoords='figure fraction', frameon=False, box_alignment=(0, 1))
            ax0.add_artist(logo_loc)
        except Exception as e:
            print(f"Gagal memuat logo untuk Matplotlib: {e}")

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
            
# =============================================================================
# 6. TABEL VERIFIKASI HISTORIS MODEL VS DATA REAL (ALL BASERUNS - 12 JAM)
# =============================================================================
st.divider()

# KODE BARU: Menggunakan Expander agar bisa di-hide/munculkan
with st.expander("🔬 Klik Untuk Menampilkan Verifikasi Model vs HF Radar 🔗", expanded=False):

    with st.spinner("Merekap riwayat seluruh baserun prediksi..."):
    
        # ---------------------------------------------------------
        # LANGKAH 1: PRE-LOAD DATA REAL (AGAR BISA MATCHING 12 TITIK)
        # ---------------------------------------------------------
        from pathlib import Path
        BASE_DIR = Path(__file__).parent.resolve()
        path_verif_csv = BASE_DIR / "Hasil_Validasi" / "Validasi_Real_Max_12H.csv" 
        
        df_real_hourly = pd.DataFrame()
        df_real_agg = pd.DataFrame()
        
        if path_verif_csv.exists():
            df_real_raw = pd.read_csv(path_verif_csv)
            
            if 'Waktu_UTC' in df_real_raw.columns:
                df_real_raw['waktu_utc'] = pd.to_datetime(df_real_raw['Waktu_UTC'])
                df_real_raw['Tanggal'] = df_real_raw['waktu_utc'].dt.strftime('%Y-%m-%d')
                df_real_raw['Shift'] = df_real_raw['waktu_utc'].dt.hour.apply(lambda h: "Pagi" if h < 12 else "Malam")
                
                # Konversi satuan Data Real sejak awal (m/s atau Knot)
                if satuan_teks == "Knot":
                    df_real_raw['Real_Max_Speed_ms'] = df_real_raw['Real_Max_Speed_ms'] * 1.94384
                
                # Simpan data mentah per-jam untuk kalkulasi RMSE/MBE 12 Titik
                df_real_hourly = df_real_raw[['waktu_utc', 'Real_Max_Speed_ms']].copy()
                
                # Buat data agregasi untuk mencari nilai Maksimal, Waktu Kejadian & Ketersediaan Data
                idx_max_real = df_real_raw.groupby(['Tanggal', 'Shift'])['Real_Max_Speed_ms'].idxmax()
                df_real_max = df_real_raw.loc[idx_max_real, ['Tanggal', 'Shift', 'waktu_utc', 'Real_Max_Speed_ms']].copy()
                
                # Ubah UTC ke WIB khusus untuk tampilan waktu puncak observasi
                df_real_max['Waktu_Max_Real'] = (df_real_max['waktu_utc'] + pd.Timedelta(hours=7)).dt.strftime('%H:%M WIB')
                df_real_max.rename(columns={'Real_Max_Speed_ms': 'Real_Max_Speed_Max'}, inplace=True)
                
                # Hitung jumlah jam tersedia
                df_real_count = df_real_raw.groupby(['Tanggal', 'Shift']).size().reset_index(name='Jumlah_Jam')
                
                # Gabungkan kembali
                df_real_agg = pd.merge(df_real_max, df_real_count, on=['Tanggal', 'Shift'])
                
                # Filter minimal 6 jam data
                df_real_agg = df_real_agg[df_real_agg['Jumlah_Jam'] >= 6].copy()
                
                def qc_shift(jml):
                    if jml >= 9: return ">75% (Sangat Baik)"
                    else: return "50% - 75% (Cukup)"
                        
                df_real_agg['Kualitas_Data'] = df_real_agg['Jumlah_Jam'].apply(qc_shift)

        # ---------------------------------------------------------
        # LANGKAH 2: LOOPING DATA PREDIKSI (BASERUNS)
        # ---------------------------------------------------------
        rekap_prediksi = []
        
        def cek_shift_target(df_prediksi_24jam):
            last_time = df_prediksi_24jam['datetime'].max()
            if last_time.hour < 12: return 12, 24, "Malam"
            else: return 0, 12, "Pagi"

        def get_p95_idx_local(group):
            p95 = group['predicted_v_max'].quantile(0.95)
            if pd.isna(p95): return group.index[0]
            return (group['predicted_v_max'] - p95).abs().idxmin()

        for filepath in list_of_files:
            df_temp = load_data(filepath)
            start_hour_utc, end_hour_utc, shift_nama = cek_shift_target(df_temp)
            
            mask_shift = (
                (df_temp['datetime'].dt.hour >= start_hour_utc) & 
                (df_temp['datetime'].dt.hour < end_hour_utc) &
                (df_temp['lat'] >= LAT_MIN) & (df_temp['lat'] <= LAT_MAX) &
                (df_temp['lon'] >= LON_MIN) & (df_temp['lon'] <= LON_MAX)
            )
            df_temp_shift = df_temp.loc[mask_shift]
            
            if df_temp_shift.empty:
                continue
                
            # Ambil 12 titik (P95 tiap jamnya)
            idx_p95 = df_temp_shift.groupby('datetime_wib').apply(get_p95_idx_local)
            df_p95_shift = df_temp_shift.loc[idx_p95].copy()
            
            tanggal_str = df_temp_shift['datetime'].dt.strftime('%Y-%m-%d').mode()[0]
            
            # Cari nilai prediksi maksimum dan waktu kejadiannya di shift tersebut
            df_p95_shift['pred_speed'] = df_p95_shift['predicted_v_max'] * faktor_konversi
            baris_max_pred = df_p95_shift.loc[df_p95_shift['pred_speed'].idxmax()]
            
            max_pred_shift = baris_max_pred['pred_speed']
            waktu_max_pred_str = baris_max_pred['datetime_wib'].strftime('%H:%M WIB')
            
            shift_mbe = np.nan
            shift_rmse = np.nan
            
            # --- PERHITUNGAN RMSE & MBE DARI 12 TITIK (PER JAM) ---
            raw_errors_shift = [] 
            chart_data_dict = {} # KODE BARU: Wadah untuk menyimpan data plot
            
            if not df_real_hourly.empty:
                # KODE BARU: Left Join agar nilai Prediksi tetap utuh meskipun observasi bolong
                df_chart = pd.merge(
                    df_p95_shift[['datetime', 'datetime_wib', 'pred_speed']], 
                    df_real_hourly[['waktu_utc', 'Real_Max_Speed_ms']], 
                    left_on='datetime', 
                    right_on='waktu_utc', 
                    how='left'
                )
                
                # Ekstrak data mentah untuk dikirim ke popup grafik
                df_chart['jam_wib'] = df_chart['datetime_wib'].dt.strftime('%H:%M')
                chart_data_dict = {
                    'waktu': df_chart['jam_wib'].tolist(),
                    'prediksi': df_chart['pred_speed'].tolist(),
                    'observasi': df_chart['Real_Max_Speed_ms'].tolist()
                }
                
                # Hapus data yang observasinya kosong (Hanya untuk hitung error metrik)
                df_merged = df_chart.dropna(subset=['Real_Max_Speed_ms', 'pred_speed'])
                
                if not df_merged.empty:
                    errors = df_merged['Real_Max_Speed_ms'] - df_merged['pred_speed']
                    shift_mbe = errors.mean()
                    shift_rmse = np.sqrt((errors**2).mean())
                    raw_errors_shift = errors.tolist() 
            
            rekap_prediksi.append({
                'Tanggal': tanggal_str,
                'Shift': shift_nama,
                f'Prediksi Arus Maks ({satuan_teks})': max_pred_shift,
                f'MBE ({satuan_teks})': shift_mbe,
                f'RMSE ({satuan_teks})': shift_rmse,
                'Baserun': os.path.basename(filepath),
                'raw_errors': raw_errors_shift,
                'chart_data': chart_data_dict # KODE BARU: Bawa data grafik
            })
            
        df_rekap_pred = pd.DataFrame(rekap_prediksi)

        # ---------------------------------------------------------
        # LANGKAH 3: GABUNGKAN DENGAN NILAI MAX REAL & RAPIKAN TABEL
        # ---------------------------------------------------------
        if not df_rekap_pred.empty and not df_real_agg.empty:
            df_verif = pd.merge(
                df_rekap_pred, 
                df_real_agg[['Tanggal', 'Shift', 'Real_Max_Speed_Max', 'Waktu_Max_Real', 'Kualitas_Data']], 
                on=['Tanggal', 'Shift'], 
                how='left' 
            )
            
            # Hapus nilai MBE dan RMSE jika Data Real (Real_Max_Speed_Max) kosong
            mask_kosong = df_verif['Real_Max_Speed_Max'].isna()
            df_verif.loc[mask_kosong, f'MBE ({satuan_teks})'] = np.nan
            df_verif.loc[mask_kosong, f'RMSE ({satuan_teks})'] = np.nan
            df_verif.loc[mask_kosong, 'Waktu_Max_Real'] = np.nan
        else:
            df_verif = df_rekap_pred.copy() if not df_rekap_pred.empty else pd.DataFrame()
            if not df_verif.empty:
                df_verif['Real_Max_Speed_Max'] = np.nan
                df_verif['Waktu_Max_Real'] = np.nan
                df_verif['Kualitas_Data'] = "Tidak Ada Data CSV"

        if not df_verif.empty:
            df_verif.rename(columns={
                'Real_Max_Speed_Max': f'Observasi Arus Maks ({satuan_teks})',
                'Waktu_Max_Real': 'Waktu Puncak Observasi',
                'Kualitas_Data': 'Ketersediaan Data'
            }, inplace=True)
            
            df_verif['Ketersediaan Data'] = df_verif['Ketersediaan Data'].fillna("Data Tidak/Belum Tersedia")
            df_verif['Waktu Puncak Observasi'] = df_verif['Waktu Puncak Observasi'].fillna("-")

            # Urutkan dari yang terbaru
            df_verif = df_verif.sort_values(by=['Tanggal', 'Baserun'], ascending=[False, False])

            # Ambil SELURUH baris di mana data Prediksi, Real, dan RMSE sepenuhnya tersedia
            valid_data = df_verif.dropna(subset=[
                f'RMSE ({satuan_teks})', 
                f'Observasi Arus Maks ({satuan_teks})'
            ])
            
            if len(valid_data) > 0:
                # --- KODE BARU: MENGHITUNG METRIK KESELURUHAN (GLOBAL) ---
                # Lebur semua error per jam dari SELURUH shift menjadi satu array besar
                all_errors = []
                for err_list in valid_data['raw_errors']:
                    if isinstance(err_list, list) and len(err_list) > 0:
                        all_errors.extend(err_list)
                        
                all_errors = np.array(all_errors)
                
                # Hitung Global MBE dan Global RMSE dari sekumpulan titik jam tersebut
                mbe_score = all_errors.mean() if len(all_errors) > 0 else 0.0
                rmse_score = np.sqrt((all_errors**2).mean()) if len(all_errors) > 0 else 0.0
                # ---------------------------------------------------------
                
                if rmse_score <= 0.4:
                    delta_teks_rmse = "Akurasi Baik & Aman"
                    color_rmse = "normal"
                else:
                    delta_teks_rmse = "Waspada Error Ekstrem"
                    color_rmse = "inverse"
                    
                if mbe_score > 0.1: 
                    delta_teks_mbe = "Model Cenderung Underestimate"
                    color_mbe = "off" 
                elif mbe_score < -0.1: 
                    delta_teks_mbe = "Model Cenderung Overestimate"
                    color_mbe = "off" 
                else:
                    delta_teks_mbe = "Bias Seimbang / Sangat Presisi"
                    color_mbe = "normal" 
            else:
                mbe_score = 0.0
                rmse_score = 0.0
                delta_teks_rmse = "Menunggu Observasi"
                delta_teks_mbe = "Menunggu Observasi"
                color_rmse = "off"
                color_mbe = "off"

            # Format Angka
            df_verif[f'Prediksi Arus Maks ({satuan_teks})'] = df_verif[f'Prediksi Arus Maks ({satuan_teks})'].map('{:.2f}'.format)
            df_verif[f'Observasi Arus Maks ({satuan_teks})'] = df_verif[f'Observasi Arus Maks ({satuan_teks})'].apply(lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '-')
            df_verif[f'MBE ({satuan_teks})'] = df_verif[f'MBE ({satuan_teks})'].apply(lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '-')
            df_verif[f'RMSE ({satuan_teks})'] = df_verif[f'RMSE ({satuan_teks})'].apply(lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '-')

            # Susun Kolom (Hanya menyisakan Waktu Puncak Observasi)
            kolom_fix = [
                'Tanggal', 'Shift', f'Prediksi Arus Maks ({satuan_teks})',
                f'Observasi Arus Maks ({satuan_teks})', 'Waktu Puncak Observasi',
                f'MBE ({satuan_teks})', f'RMSE ({satuan_teks})', 'Ketersediaan Data'
            ]
            df_verif_tampil = df_verif[kolom_fix]

            # =========================================================
            # FUNGSI POP-UP GRAFIK 
            # =========================================================
            # Tambahkan parameter width="large" agar jendela lebih lebar
            @st.dialog("📈 Grafik Perbandingan Shift", width="large")
            def tampilkan_popup_grafik(tanggal, shift, chart_data):
                if not chart_data or 'waktu' not in chart_data:
                    st.warning("Data observasi grafik tidak tersedia untuk shift ini.")
                    return
                    
                fig_popup = go.Figure()
                fig_popup.add_trace(go.Scatter(
                    x=chart_data['waktu'], y=chart_data['prediksi'],
                    mode='lines+markers', name='Prediksi Model', line=dict(color='blue', width=2)
                ))
                fig_popup.add_trace(go.Scatter(
                    x=chart_data['waktu'], y=chart_data['observasi'],
                    mode='lines+markers', name='Observasi HF Radar', line=dict(color='red', width=2)
                ))
                
                fig_popup.update_layout(
                    title=f"Tanggal: {tanggal} | Shift: {shift}",
                    xaxis_title="Waktu (WIB)",
                    yaxis_title=f"Kecepatan Arus ({satuan_teks})",
                    hovermode="x unified",
                    margin=dict(l=0, r=0, t=60, b=0), # Margin atas dilebarkan
                    # Legenda digeser ke tengah atas agar tidak menabrak teks
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5) 
                )
                st.plotly_chart(fig_popup, use_container_width=True)

            # ---------------------------------------------------------
            # LANGKAH 4: RENDER UI KARTU METRIK & TABEL 
            # ---------------------------------------------------------
            col_v1, col_v2, col_v3 = st.columns(3)

            with col_v1:
                st.metric(
                    label="📌 Rekap Shift Tersedia", 
                    value=f"{len(df_verif)} Shift", 
                    delta=f"{len(valid_data)} Tervalidasi Data Real", 
                    delta_color="normal"
                )

            with col_v2:
                st.metric(
                    label="🧭 MBE Keseluruhan",  
                    value=f"{mbe_score:.2f} {satuan_teks}", 
                    delta=delta_teks_mbe, 
                    delta_color=color_mbe if not valid_data.empty else "off"
                )

            with col_v3:
                st.metric(
                    label="📉 RMSE Keseluruhan", 
                    value=f"{rmse_score:.2f} {satuan_teks}", 
                    delta=delta_teks_rmse, 
                    delta_color=color_rmse if not valid_data.empty else "off"
                )

            def warnai_ketersediaan(val):
                if isinstance(val, str):
                    if 'Sangat Baik' in val: return 'color: #28a745;'
                    elif 'Cukup' in val: return 'color: #ffc107;'
                    elif 'Menunggu' in val or 'Tidak' in val: return 'color: #dc3545;'
                return ''

            # Render tabel secara normal (tanpa checkbox)
            styled_df = df_verif_tampil.style.map(warnai_ketersediaan, subset=['Ketersediaan Data'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # =========================================================
            # UI BARU: PEMANGGIL GRAFIK (Dropdown + Button)
            # =========================================================
            st.markdown("---")
            st.markdown("**📈 Lihat Detail Grafik Perbandingan Per Shift**")
            
            # Gabungkan teks Tanggal dan Shift untuk opsi dropdown
            opsi_pilihan = df_verif['Tanggal'].astype(str) + " - Shift " + df_verif['Shift'].astype(str)
            
            col_pilih1, col_pilih2 = st.columns([3, 1])
            with col_pilih1:
                pilihan_shift = st.selectbox(
                    "Pilih jadwal untuk melihat grafik:", 
                    options=opsi_pilihan.tolist(), 
                    label_visibility="collapsed"
                )
                
            with col_pilih2:
                if st.button("Lihat Grafik", use_container_width=True):
                    # Cari baris data berdasarkan pilihan dropdown
                    idx = opsi_pilihan.tolist().index(pilihan_shift)
                    baris = df_verif.iloc[idx]
                    
                    # Panggil popup grafik
                    tampilkan_popup_grafik(baris['Tanggal'], baris['Shift'], baris['chart_data'])
        else:
            st.warning("Belum ada data rekap prediksi yang dapat ditampilkan.")