# =============================================================================
# FILE: app.py
# FUNGSI: Dashboard Interaktif Arus Selat Bali (Redesign Phase 2: Tab & Multi-Port)
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
import re
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
from streamlit_autorefresh import st_autorefresh

# ==========================================
# [BLOK KONFIGURASI GLOBAL] 
# ==========================================
AREA_LAT_MIN, AREA_LAT_MAX = -8.209161, -8.160084
AREA_LON_MIN, AREA_LON_MAX = 114.387089, 114.435215

KET_LAT_MIN, KET_LAT_MAX = -8.209161, -8.151693
KET_LON_MIN, KET_LON_MAX = 114.38673, 114.414975

GIL_LAT_MIN, GIL_LAT_MAX = -8.21003, -8.16198
GIL_LON_MIN, GIL_LON_MAX = 114.41354, 114.43416

FAKTOR_KNOT = 1.94384
METODE_STATISTIK = "MAX"  # "MAX" atau "P95"

# --- KONFIGURASI STATUS, THRESHOLD, DAN GAYA (UBAH DI SINI SAJA) ---
# Kamu bisa bebas menambah/menghapus/mengubah status di sini.
# Sistem akan otomatis membaca batas_ms dan mengurutkannya.
KONFIG_STATUS = {
    "AWAS":    {"batas_ms": 2.5, "icon": "🔴", "bg": "#dc3545", "text_color": "#ffffff"},
    "WASPADA": {"batas_ms": 2.0, "icon": "🟠", "bg": "#fd7e14", "text_color": "#ffffff"},
    "SIAGA":   {"batas_ms": 1.5, "icon": "🟡", "bg": "#ffc107", "text_color": "#000000"},
    "AMAN":    {"batas_ms": 0.0, "icon": "🟢", "bg": "#28a745", "text_color": "#ffffff"} 
}

# Auto-generate label untuk dipakai di tabel
for k, v in KONFIG_STATUS.items():
    v["label"] = f"{v['icon']} {k}"

def ambil_index_target(group):
    if METODE_STATISTIK == "P95":
        p95 = group['predicted_v_max'].quantile(0.95)
        if pd.isna(p95): return group.index[0]
        return (group['predicted_v_max'] - p95).abs().idxmin()
    return group['predicted_v_max'].idxmax()

# ==========================================
# HTML TEMPLATE PETA LEAFLET
# ==========================================
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
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map');
        map.fitBounds([ [__LAT_MIN__, __LON_MIN__], [__LAT_MAX__, __LON_MAX__] ]);
        
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        }).addTo(map);

        var imageUrl = 'data:image/png;base64,__IMAGE_BASE64__';
        var imageBounds = [[__LAT_MIN__, __LON_MIN__], [__LAT_MAX__, __LON_MAX__]];
        L.imageOverlay(imageUrl, imageBounds, {opacity: 0.75}).addTo(map);

        var labelsData = __LABELS_JSON__;
        labelsData.forEach(function(lbl) {
            var icon = L.divIcon({
                className: 'custom-div-icon',
                html: '<div style="font-size: 11px; color: white; font-weight: bold; text-shadow: 1px 1px 3px black, -1px -1px 3px black, 0px 0px 4px black;">' + lbl.val + '</div>',
                iconSize: [30, 15],
                iconAnchor: [15, 7]
            });
            L.marker([lbl.lat, lbl.lon], {icon: icon, interactive: false}).addTo(map);
        });

        var velocityLayer = L.velocityLayer({
            displayValues: true,
            displayOptions: { velocityType: 'Kecepatan Arus', displayPosition: 'bottomleft', displayEmptyString: 'Tidak ada data' },
            data: __DATA_JSON__,
            maxVelocity: __MAX_VELOCITY__,
            velocityScale: 0.015,         
            particleAge: 300,             
            particleMultiplier: 0.002,    // Dibuat cukup padat agar glow menyatu
            colorScale: ['#ffffff']       // Tetap putih murni
        });
        velocityLayer.addTo(map);

        var legend = L.control({position: 'bottomright'});
        legend.onAdd = function (map) {
            var div = L.DomUtil.create('div', 'info legend');
            div.style.backgroundColor = 'rgba(15, 23, 42, 0.9)'; 
            div.style.padding = '10px 15px';
            div.style.borderRadius = '8px';
            div.style.border = '1px solid rgba(148,163,184,0.3)';
            div.style.boxShadow = '2px 2px 5px rgba(0,0,0,0.5)';
            div.style.fontFamily = 'Arial, sans-serif';
            div.innerHTML = 
                '<div style="margin-bottom: 8px; font-weight: bold; font-size: 12px; color: #f8fafc; text-align: center;">Kecepatan Arus (__SATUAN__)</div>' +
                '<div style="background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ffa500, #ff0000, #4b0082); ' +
                'width: 250px; height: 15px; border-radius: 10px; box-shadow: inset 0 0 2px rgba(0,0,0,0.8);"></div>' +
                '<div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: 5px; color: #cbd5e1; font-weight: bold; padding: 0 5px;">' +
                    '<span>0</span>' +
                    '<span>__LEG_1__</span><span>__LEG_2__</span><span>__LEG_3__</span><span>__LEG_4__</span>' +
                '</div>';
            return div;
        };
        legend.addTo(map);
    </script>
</body>
</html>
"""

def konversi_ke_velocity_json(df_hour):
    lo1, lo2 = float(df_hour['lon'].min()), float(df_hour['lon'].max())
    la1, la2 = float(df_hour['lat'].max()), float(df_hour['lat'].min()) 
    nx, ny = 60, 60
    grid_lon, grid_lat = np.linspace(lo1, lo2, nx), np.linspace(la1, la2, ny) 
    dx, dy = (lo2 - lo1) / (nx - 1), (la1 - la2) / (ny - 1) 
    
    triang = tri.Triangulation(df_hour['lon'], df_hour['lat'])
    batas_resolusi = 0.011
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
                
    return json.dumps([
        {"header": {"parameterCategory": 2, "parameterNumber": 2, "nx": nx, "ny": ny, "lo1": lo1, "lo2": lo2, "la1": la1, "la2": la2, "dx": dx, "dy": dy}, "data": u_data},
        {"header": {"parameterCategory": 2, "parameterNumber": 3, "nx": nx, "ny": ny, "lo1": lo1, "lo2": lo2, "la1": la1, "la2": la2, "dx": dx, "dy": dy}, "data": v_data}
    ])

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial'], 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})

# ==========================================
# SETUP HALAMAN WEBSITE
# ==========================================
BASE_DIR = Path(__file__).parent.resolve()
FOLDER_OUTPUT = BASE_DIR / "Arsip_Prediksi"

logo_filename = "logo_bmkg.png"
logo_seacast = "logo_bmkg2.png"
ikon = Image.open(logo_seacast) if os.path.exists(logo_seacast) else (Image.open("logo_bmkg.png") if os.path.exists("logo_bmkg.png") else "🌊")

st.set_page_config(page_title="MARLIN Dashboard", page_icon=ikon, layout="wide")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;} 
        header {visibility: hidden;} 
        .viewerBadge {display: none !important;}
        
        /* Ini kode untuk memangkas jarak atas dan bawah */
        .block-container {
            padding-top: 2rem !important; 
            padding-bottom: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)
st_autorefresh(interval=1800000, limit=None, key="auto_refresh_marlin")

def get_base64_logo(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None

def get_available_baseruns():
    return sorted(glob.glob(os.path.join(FOLDER_OUTPUT, "prediksi_spasial*.csv")), reverse=True)

@st.cache_data(ttl=3600) 
def load_data(file_path):
    df = pd.read_csv(file_path).rename(columns={'datetime_prediction': 'datetime', 'latitude': 'lat', 'longitude': 'lon', 'current_speed_ms': 'predicted_v_max', 'v_velocity_ms': 'predicted_v', 'u_velocity_ms': 'predicted_u', 'current_direction_deg': 'arah_derajat'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime_wib'] = df['datetime'] + pd.Timedelta(hours=7)
    return df

def derajat_ke_arah_lengkap(deg):
    if pd.isna(deg): return ""
    return ['Utara', 'Timur Laut', 'Timur', 'Tenggara', 'Selatan', 'Barat Daya', 'Barat', 'Barat Laut'][int((deg + 22.5) % 360 // 45)]

def derajat_ke_panah(deg):
    if pd.isna(deg): return ""
    return ['↑ U', '↗ TL', '→ T', '↘ TG', '↓ S', '↙ BD', '← B', '↖ BL'][int((deg + 22.5) % 360 // 45)]

def derajat_ke_utara_selatan(deg):
    return "" if pd.isna(deg) else ("Utara" if deg >= 270 or deg < 90 else "Selatan")

# --- FUNGSI STATUS DINAMIS ---
def get_status_arus(speed, faktor_konv):
    # Cari status terendah sebagai nilai default/fallback (Batas_ms = 0)
    status_terendah = min(KONFIG_STATUS.items(), key=lambda x: x[1]['batas_ms'])[0]
    
    if pd.isna(speed): 
        return {"status": status_terendah, **KONFIG_STATUS[status_terendah]}
    
    # Mengurutkan konfigurasi dari nilai batas tertinggi ke terendah
    sorted_status = sorted(KONFIG_STATUS.items(), key=lambda x: x[1]['batas_ms'], reverse=True)
    
    for nama_status, config in sorted_status:
        # Pengecekan otomatis mengalikan threshold dengan faktor satuan (Knot/ms)
        if speed >= (config['batas_ms'] * faktor_konv):
            return {"status": nama_status, **config}
            
    return {"status": status_terendah, **KONFIG_STATUS[status_terendah]}

list_of_files = get_available_baseruns()
if not list_of_files:
    st.error(f"❌ Tidak ada file prediksi ditemukan di folder: {FOLDER_OUTPUT}")
    st.stop()

def format_nama_baserun(filepath):
    match = re.search(r'prediksi_spasial_(\d{8})_(\d{4})\.csv', os.path.basename(filepath))
    return (pd.to_datetime(f"{match.group(1)}{match.group(2)}", format="%Y%m%d%H%M") + pd.Timedelta(hours=7)).strftime("%Y-%m-%d %H:%M WIB") if match else os.path.basename(filepath)

file_options = {format_nama_baserun(f): f for f in list_of_files}

# ==========================================
# 1. HEADER DENGAN LOGO BMKG
# ==========================================
logo_base64 = get_base64_logo(logo_seacast)
col_head1, col_head2 = st.columns([3, 1])

with col_head1:
    if logo_base64:
        html_header = f"""<div style="display: flex; flex-direction: row; align-items: center; margin-bottom: 5px;">
        <img src="data:image/png;base64,{logo_base64}" style="height: 60px; width: auto; margin-right: 15px;">
        <div style="display: flex; flex-direction: column; justify-content: center;">
        <h1 style="margin: 0; padding: 0; font-size: 2rem; font-weight: bold; line-height: 1.1;">MARLIN</h1>
        <p style="margin: 0; font-size: 1rem; opacity: 0.8;">Marine Current Intelligence System</p>
        </div></div>"""
        st.markdown(html_header, unsafe_allow_html=True)
    else:
        st.markdown("## MARLIN \n *Marine Current Intelligence System*")

st.markdown("<br>**Kontrol Data & Pengaturan**", unsafe_allow_html=True)
col_dc1, col_dc2, col_dc3 = st.columns([2, 1, 1])

selected_filename = st.session_state.get('sel_file', list(file_options.keys())[0])
if selected_filename not in file_options: selected_filename = list(file_options.keys())[0]
satuan_pilihan = st.session_state.get('sel_unit', 'm/s')

with col_dc1:
    selected_filename = st.selectbox("File Prediksi", options=list(file_options.keys()), index=list(file_options.keys()).index(selected_filename), label_visibility="collapsed")
with col_dc2:
    if st.button("↻ Perbarui Data", use_container_width=True): load_data.clear(); st.rerun()
with col_dc3:
    satuan_pilihan = st.radio("Satuan Utama:", ["m/s", "Knot"], index=["m/s", "Knot"].index(satuan_pilihan), horizontal=True, label_visibility="collapsed")

if st.session_state.get('sel_file') != selected_filename or st.session_state.get('sel_unit') != satuan_pilihan:
    st.session_state['sel_file'], st.session_state['sel_unit'] = selected_filename, satuan_pilihan
    st.rerun()

# ==========================================
# PERSIAPAN DATA GLOBAL & 24 JAM
# ==========================================
df_spatial = load_data(file_options[selected_filename])
faktor_konversi = FAKTOR_KNOT if satuan_pilihan == "Knot" else 1.0
satuan_teks = satuan_pilihan

df_spatial['predicted_v_max'] *= faktor_konversi
df_spatial['predicted_u'] *= faktor_konversi
df_spatial['predicted_v'] *= faktor_konversi

# Perhitungan trend 24 jam untuk grafik
df_filtered_area_24 = df_spatial[
    (df_spatial['lat'] >= AREA_LAT_MIN) & (df_spatial['lat'] <= AREA_LAT_MAX) &
    (df_spatial['lon'] >= AREA_LON_MIN) & (df_spatial['lon'] <= AREA_LON_MAX)
]
idx_target_trend_24 = df_filtered_area_24.groupby('datetime_wib').apply(ambil_index_target)
df_trend_stat = df_filtered_area_24.loc[idx_target_trend_24].reset_index(drop=True)

waktu_sekarang_wib = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=7)
last_hour = df_spatial['datetime'].max().hour 
start_hour_utc, end_hour_utc, shift_name_global = (12, 24, "Shift Malam") if last_hour < 12 else (0, 12, "Shift Pagi")

with col_head2:
    selisih_jam = (waktu_sekarang_wib - (df_spatial['datetime_wib'].min() - pd.Timedelta(hours=1))).total_seconds() / 3600
    st.markdown(f"<div style='text-align: right; font-weight: bold; color: {'#28a745' if selisih_jam <= 24 else '#dc3545'};'>{'🟢 DATA UPDATE' if selisih_jam <= 24 else '🔴 TIDAK UPDATE'}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Fungsi Ambil Data Area Spesifik
def get_summary_area_data(df_raw, lat_min, lat_max, lon_min, lon_max):
    df_area = df_raw[(df_raw['lat'] >= lat_min) & (df_raw['lat'] <= lat_max) & (df_raw['lon'] >= lon_min) & (df_raw['lon'] <= lon_max)]
    if df_area.empty: return None
    
    idx_target = df_area.groupby('datetime_wib').apply(ambil_index_target)
    df_trend = df_area.loc[idx_target].reset_index(drop=True)
    
    df_trend['selisih_waktu'] = abs(df_trend['datetime_wib'] - waktu_sekarang_wib)
    baris_saat_ini = df_trend.loc[df_trend['selisih_waktu'].idxmin()]
    
    waktu_utc = df_trend['datetime_wib'] - pd.Timedelta(hours=7)
    mask_shift = (waktu_utc.dt.hour >= start_hour_utc) & (waktu_utc.dt.hour < end_hour_utc)
    df_shift = df_trend[mask_shift]
    
    baris_max = df_shift.loc[df_shift['predicted_v_max'].idxmax()] if not df_shift.empty else baris_saat_ini
    
    return {
        'current_speed': baris_saat_ini['predicted_v_max'],
        'current_dir': derajat_ke_utara_selatan(baris_saat_ini['arah_derajat']) or "-",
        'current_time': baris_saat_ini['datetime_wib'].strftime('%H:%M WIB'),
        'peak_speed': baris_max['predicted_v_max'],
        'peak_dir': derajat_ke_utara_selatan(baris_max['arah_derajat']) or "-",
        'peak_time': baris_max['datetime_wib'].strftime('%H:%M WIB')
    }

def create_summary_card(title, data):
    if not data: return f"<div>Data {title} tidak tersedia</div>"
    alt_satuan = "Knot" if satuan_teks == "m/s" else "m/s"
    f_pembagi = FAKTOR_KNOT if satuan_teks == "m/s" else (1 / FAKTOR_KNOT)
    curr_alt, peak_alt = data['current_speed'] * f_pembagi, data['peak_speed'] * f_pembagi
    status = get_status_arus(data['current_speed'], faktor_konversi)
    
    return f"""
    <div style="background: linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.7) 100%); border: 1px solid rgba(148,163,184,0.2); border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(148,163,184,0.2); padding-bottom: 8px; margin-bottom: 12px;">
            <h4 style="color: #cbd5e1; margin: 0; font-size: 1rem; font-weight: 600; text-transform: uppercase;">Pelabuhan {title}</h4>
            <div style="background-color: {status['bg']}; color: {status['text_color']}; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">{status['icon']} {status['status']}</div>
        </div>
        <div style="display: flex; gap: 15px;">
            <div style="flex: 1; border-right: 1px solid rgba(148,163,184,0.2); padding-right: 10px;">
                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 5px;">Kecepatan Saat Ini ({data['current_time']})</div>
                <div style="color: #f8fafc; font-size: 1.8rem; font-weight: bold; line-height: 1;">{data['current_speed']:.2f} <span style="font-size: 1rem; color: #94a3b8; font-weight: normal;">{satuan_teks}</span></div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">{curr_alt:.2f} {alt_satuan} • Arah: <b>{data['current_dir']}</b></div>
            </div>
            <div style="flex: 1;">
                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 5px;">Kecepatan Maksimal - 12 Jam</div>
                <div style="color: #f8fafc; font-size: 1.8rem; font-weight: bold; line-height: 1;">{data['peak_speed']:.2f} <span style="font-size: 1rem; color: #94a3b8; font-weight: normal;">{satuan_teks}</span></div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">{peak_alt:.2f} {alt_satuan} • Arah: <b>{data['peak_dir']}</b> ({data['peak_time']})</div>
            </div>
        </div>
    </div>
    """

# ==========================================
# PEMBUATAN STRUKTUR TAB
# ==========================================
tab1, tab2, tab3 = st.tabs(["📍 Ringkasan Operasional", "📈 Prakiraan 12 Jam", "✅ Verifikasi Model"])

# ------------------------------------------
# TAB 1: RINGKASAN OPERASIONAL & SPASIAL
# ------------------------------------------
with tab1:
    col_sum_ket, col_sum_gil = st.columns(2)
    with col_sum_ket:
        data_ket = get_summary_area_data(df_spatial, KET_LAT_MIN, KET_LAT_MAX, KET_LON_MIN, KET_LON_MAX)
        st.markdown(create_summary_card("Ketapang", data_ket), unsafe_allow_html=True)
    with col_sum_gil:
        data_gil = get_summary_area_data(df_spatial, GIL_LAT_MIN, GIL_LAT_MAX, GIL_LON_MIN, GIL_LON_MAX)
        st.markdown(create_summary_card("Gilimanuk", data_gil), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 5px; font-weight: bold; margin-top: 10px;'>PETA SPASIAL ARUS SELAT BALI</div>", unsafe_allow_html=True)
    st.markdown("<style>.stSlider, .stSlider div, .stSlider span, .stSlider p { color: transparent !important; } .stSlider [data-testid='stThumbValue'], .stSlider [data-testid='stTickBarMin'], .stSlider [data-testid='stTickBarMax'], .stSlider [data-testid='stTickBar'] { display: none !important; opacity: 0 !important; }</style>", unsafe_allow_html=True)
    
    waktu_unik = sorted(df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M').unique())
    waktu_sekarang_str = waktu_sekarang_wib.floor('h').strftime('%Y-%m-%d %H:%M')
    waktu_default = waktu_sekarang_str if waktu_sekarang_str in waktu_unik else waktu_unik[-1]

    selected_time_str = st.select_slider("Waktu Prediksi", options=waktu_unik, value=waktu_default, label_visibility="collapsed")
    dt_obj = datetime.strptime(selected_time_str, '%Y-%m-%d %H:%M')
    bulan_indo = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MEI', 6: 'JUN', 7: 'JUL', 8: 'AGU', 9: 'SEP', 10: 'OKT', 11: 'NOV', 12: 'DES'}
    st.markdown(f"<div style='text-align: center; background-color: rgba(128, 128, 128, 0.1); padding: 4px; border-radius: 4px; font-weight: bold; font-size: 14px; margin-bottom: 5px;'>⏱️ {dt_obj.day:02d} {bulan_indo[dt_obj.month]} {dt_obj.year} • {dt_obj.strftime('%H:%M')} WIB</div>", unsafe_allow_html=True)
    
    df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == selected_time_str]
    lon_min_data, lon_max_data = float(df_hour['lon'].min()), float(df_hour['lon'].max())
    lat_min_data, lat_max_data = float(df_hour['lat'].min()), float(df_hour['lat'].max())

    @st.cache_data(show_spinner=False, ttl=3600)
    def generate_map_background(df_hour_json, lat_min, lat_max, lon_min, lon_max, vmax_plot):
        df_h = pd.DataFrame(df_hour_json)
        triang = tri.Triangulation(df_h['lon'], df_h['lat'])
        x_tri, y_tri = triang.x[triang.triangles], triang.y[triang.triangles]
        mask = np.any([np.sqrt((x_tri[:, i] - x_tri[:, (i+1)%3])**2 + (y_tri[:, i] - y_tri[:, (i+1)%3])**2) > 0.011 for i in range(3)], axis=0)
        triang.set_mask(mask)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off'); ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
        fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        ax.tricontourf(triang, df_h['predicted_v_max'], levels=np.linspace(0, vmax_plot, 100), cmap=LinearSegmentedColormap.from_list("custom", ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ffa500', '#ff0000', '#4b0082']), vmin=0, vmax=vmax_plot, extend='max', alpha=0.9, antialiased=True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
        buf.seek(0); plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    vmax_dinamis = 4.0 * faktor_konversi
    image_base64 = generate_map_background(df_hour[['lat', 'lon', 'predicted_v_max']].to_dict('list'), lat_min_data, lat_max_data, lon_min_data, lon_max_data, vmax_dinamis)
    placed_labels, labels_list = [], []
    for _, row in df_hour.iloc[::3].iterrows():
        kec, lat, lon = row['predicted_v_max'], float(row['lat']), float(row['lon'])
        if pd.isna(kec): continue
        if not any(np.sqrt((lat - plat)**2 + (lon - plon)**2) < 0.015 for (plat, plon) in placed_labels):
            labels_list.append({"lat": lat, "lon": lon, "val": f"{kec:.2f}"}); placed_labels.append((lat, lon))

    html_siap_render = HTML_TEMPLATE.replace("__DATA_JSON__", konversi_ke_velocity_json(df_hour)).replace("__LABELS_JSON__", json.dumps(labels_list)).replace("__IMAGE_BASE64__", image_base64).replace("__LAT_MIN__", str(lat_min_data)).replace("__LON_MIN__", str(lon_min_data)).replace("__LAT_MAX__", str(lat_max_data)).replace("__LON_MAX__", str(lon_max_data)).replace("__MAX_VELOCITY__", str(8.0 * faktor_konversi)).replace("__SATUAN__", satuan_teks).replace("__LEG_1__", f"{round(1.0 * faktor_konversi)}").replace("__LEG_2__", f"{round(2.0 * faktor_konversi)}").replace("__LEG_3__", f"{round(3.0 * faktor_konversi)}").replace("__LEG_4__", f"{round(4.0 * faktor_konversi)}")
    components.html(html_siap_render, height=550)

# ------------------------------------------
# DATA PRAKIRAAN 12 JAM (DIBUTUHKAN TAB 2 & 3)
# ------------------------------------------
mask_shift = (df_spatial['datetime'].dt.hour >= start_hour_utc) & (df_spatial['datetime'].dt.hour < end_hour_utc)
df_shift_raw = df_spatial.loc[mask_shift & (df_spatial['lat'] >= AREA_LAT_MIN) & (df_spatial['lat'] <= AREA_LAT_MAX) & (df_spatial['lon'] >= AREA_LON_MIN) & (df_spatial['lon'] <= AREA_LON_MAX)]
if not df_shift_raw.empty:
    idx_target_shift = df_shift_raw.groupby('datetime_wib').apply(ambil_index_target)
    df_shift = df_shift_raw.loc[idx_target_shift].reset_index(drop=True)
    df_shift['arah_panah'] = df_shift['arah_derajat'].apply(derajat_ke_panah)
else:
    df_shift = pd.DataFrame()

# ------------------------------------------
# TAB 2: PRAKIRAAN & PERENCANAAN 12 JAM
# ------------------------------------------
with tab2:
    if not df_shift.empty:
        st.markdown(f"<div style='margin-bottom: 5px; font-weight: bold;'>GRAFIK KECEPATAN & ARAH — 12 JAM KEDEPAN</div>", unsafe_allow_html=True)
        fig_shift = go.Figure()
        fig_shift.add_trace(go.Scatter(
            x=df_shift['datetime_wib'], y=df_shift['predicted_v_max'], mode='lines+markers', name='Prediksi Arus',
            line=dict(color='#0d6efd', width=3), marker=dict(size=8, color='#0d6efd', line=dict(color='white', width=1)),
            customdata=df_shift['arah_panah'], hovertemplate="<b>%{x}</b><br>Kecepatan: %{y:.2f} " + satuan_teks + "<br>Arah: %{customdata}<extra></extra>"
        ))
        
        ymax_plot = df_shift['predicted_v_max'].max() + (0.3 * faktor_konversi)
        
        # Gambar garis peringatan dinamis 
        sorted_thresholds = sorted([item for item in KONFIG_STATUS.items() if item[1]['batas_ms'] > 0], key=lambda x: x[1]['batas_ms'])
        
        for i, (nama_status, config) in enumerate(sorted_thresholds):
            batas_terkonversi = config['batas_ms'] * faktor_konversi
            if ymax_plot >= batas_terkonversi:
                dash_style = ["dot", "dash", "solid"][i % 3] # Gaya garis berubah-ubah
                fig_shift.add_hline(
                    y=batas_terkonversi, line_dash=dash_style, line_color=config["bg"], 
                    annotation_text=f"{nama_status} >= {batas_terkonversi:.2f}", 
                    annotation_position="top left", annotation_font_color=config["bg"]
                )
                ymax_plot = max(ymax_plot, batas_terkonversi + (0.2 * faktor_konversi))

        fig_shift.update_layout(
            margin=dict(l=0, r=0, t=10, b=20), xaxis_title=None, yaxis_title=f"Kecepatan ({satuan_teks})",
            yaxis=dict(range=[0, ymax_plot]), height=380, showlegend=False, hovermode="x unified",
            hoverlabel=dict(bgcolor="#1e293b", font_color="#f8fafc", bordercolor="#334155"),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_shift.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig_shift.update_xaxes(showgrid=False, showspikes=True, spikecolor="#94a3b8", spikethickness=1, spikedash="dash")
        st.plotly_chart(fig_shift, use_container_width=True)
    else:
        st.info("Data prediksi untuk shift ini belum tersedia.")

    st.divider()
    st.markdown("### DETAIL KONDISI PELABUHAN")
    col_ket_tbl, col_gil_tbl = st.columns(2)

    def buat_tabel_area(df_raw, lat_min, lat_max, lon_min, lon_max):
        df_area = df_raw[(df_raw['lat'] >= lat_min) & (df_raw['lat'] <= lat_max) & (df_raw['lon'] >= lon_min) & (df_raw['lon'] <= lon_max)]
        if df_area.empty: return pd.DataFrame()
        idx_t = df_area.groupby('datetime_wib').apply(ambil_index_target)
        df_tr = df_area.loc[idx_t].reset_index(drop=True)
        def fmt_speed(val):
            inf = get_status_arus(val, faktor_konversi)
            status_terendah = min(KONFIG_STATUS.items(), key=lambda x: x[1]['batas_ms'])[0]
            return f"{val:.2f}" if inf["status"] == status_terendah else f"{val:.2f} {inf['label']}"
        return pd.DataFrame({'Waktu (WIB)': df_tr['datetime_wib'].dt.strftime('%H:%M'), f'Kecepatan ({satuan_teks})': df_tr['predicted_v_max'].apply(fmt_speed), 'Arah Dominan': df_tr['arah_derajat'].apply(derajat_ke_utara_selatan)})

    def style_tabel_peringatan(val):
        if isinstance(val, str):
            for nama_status, config in KONFIG_STATUS.items():
                if config["batas_ms"] > 0 and config["label"] in val:
                    return f'color: {config["bg"]}; font-weight: bold;'
        return ''

    with col_ket_tbl:
        st.markdown("**Area Pelabuhan Ketapang**")
        tb_ket = buat_tabel_area(df_shift_raw, KET_LAT_MIN, KET_LAT_MAX, KET_LON_MIN, KET_LON_MAX)
        if not tb_ket.empty: st.dataframe(tb_ket.style.map(style_tabel_peringatan, subset=[f'Kecepatan ({satuan_teks})']), hide_index=True, use_container_width=True)
        else: st.caption("Data Ketapang tidak tersedia.")

    with col_gil_tbl:
        st.markdown("**Area Pelabuhan Gilimanuk**")
        tb_gil = buat_tabel_area(df_shift_raw, GIL_LAT_MIN, GIL_LAT_MAX, GIL_LON_MIN, GIL_LON_MAX)
        if not tb_gil.empty: st.dataframe(tb_gil.style.map(style_tabel_peringatan, subset=[f'Kecepatan ({satuan_teks})']), hide_index=True, use_container_width=True)
        else: st.caption("Data Gilimanuk tidak tersedia.")

    st.divider()
    st.markdown("### UNDUH LAPORAN GRAFIK BMKG")
    def create_bmkg_report(df_source, title_text):
        df_plot = df_source.copy()
        if 'datetime_utc' not in df_plot.columns: df_plot['datetime_utc'] = df_plot['datetime_wib'] - pd.Timedelta(hours=7)
        x_min, x_max = df_plot['datetime_utc'].min() - pd.Timedelta(hours=1), df_plot['datetime_utc'].max() + pd.Timedelta(hours=1)
        frekuensi = '2h' if ((x_max - x_min).total_seconds() / 3600) > 15 else 'h'
        ext_ticks = pd.date_range(start=x_min, end=x_max, freq=frekuensi)

        fig_bmkg, ax0 = plt.subplots(figsize=(14, 7), dpi=100)
        fig_bmkg.subplots_adjust(left=0.08, right=0.95, top=0.7, bottom=0.15) 
        ax0.set_title(title_text, fontsize=16, pad=60)
        ax0.plot(df_plot['datetime_utc'], df_plot['predicted_v_max'], color='black', linewidth=2, marker='o', markersize=6, markerfacecolor='black', markeredgecolor='white')
        ax0.set_ylabel(f"Kecepatan Arus ({satuan_teks})", fontsize=13, weight='bold')
        batas_tertinggi = max(KONFIG_STATUS.values(), key=lambda x: x['batas_ms'])['batas_ms'] * faktor_konversi
        ax0.set_ylim(0, max(batas_tertinggi, df_plot['predicted_v_max'].max() + (0.5 * faktor_konversi)))
        ax0.grid(color='#A9CCE3', linestyle='--', linewidth=0.8); ax0.set_xlim(x_min, x_max)
        ax0.set_xlabel("Waktu (UTC)", fontsize=13, weight='bold', labelpad=10)
        ax0.set_xticks(ext_ticks); ax0.set_xticklabels([t.strftime('%H:%M') for t in ext_ticks], rotation=0, fontsize=11, weight='bold')

        ax0_twin = ax0.twiny()
        ax0_twin.set_xlabel("Waktu (WIB)", fontsize=13, weight='bold', labelpad=10)
        ax0_twin.set_xlim(x_min, x_max); ax0_twin.set_xticks(ext_ticks)
        ax0_twin.set_xticklabels([t.strftime('%H:%M') for t in (ext_ticks + pd.Timedelta(hours=7))], rotation=0, fontsize=11, weight='bold')

        for _, row in df_plot.iterrows():
            panah = "" if pd.isna(row.get('arah_derajat')) else ("↑ U" if row['arah_derajat'] >= 270 or row['arah_derajat'] < 90 else "↓ S")
            ax0.annotate(f"{panah}\n{row['predicted_v_max']:.2f}", (row['datetime_utc'], row['predicted_v_max']), textcoords="offset points", xytext=(0,15), ha='center', fontsize=10, weight='bold')

        fig_bmkg.text(0.11, 0.95, "Kantor Layanan Informasi Meteorologi Maritim\nKetapang - Gilimanuk\nBanyuwangi", fontsize=12, weight='bold', ha='left', va='top', linespacing=1.4)
        isu_tm = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=7)
        fig_bmkg.text(0.95, 0.95, f"Dikeluarkan pada {isu_tm.strftime('%H:%M')} WIB, {isu_tm.strftime('%d-%m-%Y')}", fontsize=12, weight='bold', ha='right', va='top')
        
        # Mengembalikan fitur logo AnnotationBbox
        logo_path = BASE_DIR / "Logo_BMKG.png"
        if not logo_path.exists(): logo_path = BASE_DIR / "logo_bmkg2.png"
        if not logo_path.exists(): logo_path = BASE_DIR / "logo_bmkg.png"
        if logo_path.exists():
            try:
                ax0.add_artist(AnnotationBbox(OffsetImage(plt.imread(str(logo_path)), zoom=0.08), (0.02, 0.97), xycoords='figure fraction', frameon=False, box_alignment=(0, 1)))
            except: pass 
            
        return fig_bmkg

    # Mengembalikan tombol unduh 12 jam & 24 jam bersampingan
    col_exp1, col_exp2, _ = st.columns([1, 1, 2])
    with col_exp1:
        if st.button("Unduh Grafik 12 Jam", use_container_width=True):
            if not df_shift.empty:
                f_12 = create_bmkg_report(df_shift, f"Prediksi Arus 12 Jam (Mode: {METODE_STATISTIK})")
                buf_12 = io.BytesIO(); f_12.savefig(buf_12, format="png", dpi=150); plt.close(f_12)
                st.download_button("Klik Untuk Menyimpan (12 Jam)", data=buf_12.getvalue(), file_name="Grafik_12J.png", mime="image/png", use_container_width=True)
            else: st.error("Data 12 Jam belum tersedia.")
            
    with col_exp2:
        if st.button("Unduh Grafik 24 Jam", use_container_width=True):
            if not df_trend_stat.empty:
                f_24 = create_bmkg_report(df_trend_stat, f"Prediksi Arus 24 Jam (Mode: {METODE_STATISTIK})")
                buf_24 = io.BytesIO(); f_24.savefig(buf_24, format="png", dpi=150); plt.close(f_24)
                st.download_button("Klik Untuk Menyimpan (24 Jam)", data=buf_24.getvalue(), file_name="Grafik_24J.png", mime="image/png", use_container_width=True)
            else: st.error("Data 24 Jam belum tersedia.")

# ------------------------------------------
# TAB 3: VERIFIKASI MODEL & ANALISA
# ------------------------------------------
with tab3:
    st.markdown("### Akurasi & Validasi Model")
    with st.spinner("Merekap riwayat validasi HF Radar..."):
        path_verif_csv = BASE_DIR / "Hasil_Validasi" / "Validasi_Real_Max_12H.csv" 
        df_real_hourly, df_real_agg = pd.DataFrame(), pd.DataFrame()
        
        if path_verif_csv.exists():
            df_rr = pd.read_csv(path_verif_csv)
            if 'Waktu_UTC' in df_rr.columns:
                df_rr['waktu_utc'] = pd.to_datetime(df_rr['Waktu_UTC'])
                df_rr['Tanggal'] = df_rr['waktu_utc'].dt.strftime('%Y-%m-%d')
                df_rr['Shift'] = df_rr['waktu_utc'].dt.hour.apply(lambda h: "Shift Pagi" if h < 12 else "Shift Malam")
                if satuan_teks == "Knot": df_rr['Real_Max_Speed_ms'] *= FAKTOR_KNOT
                
                df_real_hourly = df_rr[['waktu_utc', 'Real_Max_Speed_ms']].copy()
                idx_m = df_rr.groupby(['Tanggal', 'Shift'])['Real_Max_Speed_ms'].idxmax()
                df_real_max = df_rr.loc[idx_m, ['Tanggal', 'Shift', 'waktu_utc', 'Real_Max_Speed_ms']].copy()
                df_real_max['Waktu_Max_Real'] = (df_real_max['waktu_utc'] + pd.Timedelta(hours=7)).dt.strftime('%H:%M WIB')
                df_real_max.rename(columns={'Real_Max_Speed_ms': 'Real_Max_Speed_Max'}, inplace=True)
                
                df_real_agg = pd.merge(df_real_max, df_rr.groupby(['Tanggal', 'Shift']).size().reset_index(name='Jumlah_Jam'), on=['Tanggal', 'Shift'])
                df_real_agg = df_real_agg[df_real_agg['Jumlah_Jam'] >= 6].copy()
                df_real_agg['Kualitas_Data'] = df_real_agg['Jumlah_Jam'].apply(lambda j: ">75% (Sangat Baik)" if j >= 9 else "50% - 75% (Cukup)")

        rekap_prediksi = []
        for filepath in list_of_files:
            df_temp = load_data(filepath)
            s_hr_utc, e_hr_utc, shft_nm = (12, 24, "Shift Malam") if df_temp['datetime'].max().hour < 12 else (0, 12, "Shift Pagi")
            mask_local = ((df_temp['datetime'].dt.hour >= s_hr_utc) & (df_temp['datetime'].dt.hour < e_hr_utc) & (df_temp['lat'] >= AREA_LAT_MIN) & (df_temp['lat'] <= AREA_LAT_MAX) & (df_temp['lon'] >= AREA_LON_MIN) & (df_temp['lon'] <= AREA_LON_MAX))
            df_ts = df_temp.loc[mask_local]
            if df_ts.empty: continue
                
            idx_tgt = df_ts.groupby('datetime_wib').apply(ambil_index_target)
            df_stat = df_ts.loc[idx_tgt].copy()
            df_stat['pred_speed'] = df_stat['predicted_v_max'] * faktor_konversi
            
            baris_max_pred = df_stat.loc[df_stat['pred_speed'].idxmax()]
            s_mbe, s_rmse, r_err, chrt_dict = np.nan, np.nan, [], {}
            
            if not df_real_hourly.empty:
                df_chrt = pd.merge(df_stat[['datetime', 'datetime_wib', 'pred_speed']], df_real_hourly[['waktu_utc', 'Real_Max_Speed_ms']], left_on='datetime', right_on='waktu_utc', how='left')
                chrt_dict = {'waktu': df_chrt['datetime_wib'].dt.strftime('%H:%M').tolist(), 'prediksi': df_chrt['pred_speed'].tolist(), 'observasi': df_chrt['Real_Max_Speed_ms'].tolist()}
                df_mgd = df_chrt.dropna(subset=['Real_Max_Speed_ms', 'pred_speed'])
                if not df_mgd.empty:
                    err = df_mgd['Real_Max_Speed_ms'] - df_mgd['pred_speed']
                    s_mbe, s_rmse, r_err = err.mean(), np.sqrt((err**2).mean()), err.tolist()
            
            rekap_prediksi.append({
                'Tanggal': df_ts['datetime'].dt.strftime('%Y-%m-%d').mode()[0], 'Shift': shft_nm, 
                f'Prediksi ({satuan_teks})': baris_max_pred['pred_speed'],
                f'MBE ({satuan_teks})': s_mbe, f'RMSE ({satuan_teks})': s_rmse, 'Baserun': os.path.basename(filepath),
                'raw_errors': r_err, 'chart_data': chrt_dict
            })
            
        df_rekap_pred = pd.DataFrame(rekap_prediksi)

        if not df_rekap_pred.empty and not df_real_agg.empty:
            df_verif = pd.merge(df_rekap_pred, df_real_agg[['Tanggal', 'Shift', 'Real_Max_Speed_Max', 'Waktu_Max_Real', 'Kualitas_Data']], on=['Tanggal', 'Shift'], how='left')
            df_verif.loc[df_verif['Real_Max_Speed_Max'].isna(), [f'MBE ({satuan_teks})', f'RMSE ({satuan_teks})', 'Waktu_Max_Real']] = np.nan
        else:
            df_verif = df_rekap_pred.copy() if not df_rekap_pred.empty else pd.DataFrame()
            if not df_verif.empty: df_verif['Real_Max_Speed_Max'], df_verif['Waktu_Max_Real'], df_verif['Kualitas_Data'] = np.nan, np.nan, "Tidak Ada Data CSV"

        if not df_verif.empty:
            df_verif.rename(columns={'Real_Max_Speed_Max': f'Observasi ({satuan_teks})', 'Waktu_Max_Real': 'Waktu Puncak', 'Kualitas_Data': 'Ketersediaan Data'}, inplace=True)
            df_verif['Ketersediaan Data'] = df_verif['Ketersediaan Data'].fillna("Data Tidak/Belum Tersedia")
            df_verif['Waktu Puncak'] = df_verif['Waktu Puncak'].fillna("-")
            df_verif = df_verif.sort_values(by=['Tanggal', 'Baserun'], ascending=[False, False])

            valid_data = df_verif.dropna(subset=[f'RMSE ({satuan_teks})', f'Observasi ({satuan_teks})'])
            if len(valid_data) > 0:
                all_err = np.array([e for err_lst in valid_data['raw_errors'] if isinstance(err_lst, list) for e in err_lst])
                mbe_sc, rmse_sc = (all_err.mean(), np.sqrt((all_err**2).mean())) if len(all_err) > 0 else (0.0, 0.0)
                dt_rmse, cr_rmse = ("Akurasi Sangat Baik", "normal") if rmse_sc <= 0.4 else ("Waspada Error", "inverse")
                if mbe_sc > 0.1: dt_mbe, cr_mbe = "Cenderung Underestimate", "off"
                elif mbe_sc < -0.1: dt_mbe, cr_mbe = "Cenderung Overestimate", "off"
                else: dt_mbe, cr_mbe = "Sangat Presisi", "normal"
            else: mbe_sc, rmse_sc, dt_rmse, dt_mbe, cr_rmse, cr_mbe = 0.0, 0.0, "N/A", "N/A", "off", "off"

            df_verif[f'Prediksi ({satuan_teks})'] = df_verif[f'Prediksi ({satuan_teks})'].map('{:.2f}'.format)
            for c in [f'Observasi ({satuan_teks})', f'MBE ({satuan_teks})', f'RMSE ({satuan_teks})']:
                df_verif[c] = df_verif[c].apply(lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '-')

            df_v_show = df_verif[['Tanggal', 'Shift', f'Prediksi ({satuan_teks})', f'Observasi ({satuan_teks})', 'Waktu Puncak', f'MBE ({satuan_teks})', f'RMSE ({satuan_teks})', 'Ketersediaan Data']]

            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1: st.metric("Shift Available", f"{len(valid_data)}", f"Total {len(df_verif)}")
            with col_v2: st.metric("MBE Global", f"{mbe_sc:.2f} {satuan_teks}", dt_mbe, delta_color=cr_mbe)
            with col_v3: st.metric("RMSE Global", f"{rmse_sc:.2f} {satuan_teks}", dt_rmse, delta_color=cr_rmse)

            def clr_avail(val):
                if isinstance(val, str):
                    if 'Sangat Baik' in val: return 'color: #28a745;'
                    if 'Cukup' in val: return 'color: #ffc107;'
                    if 'Menunggu' in val or 'Tidak' in val: return 'color: #dc3545;'
                return ''

            st.dataframe(df_v_show.style.map(clr_avail, subset=['Ketersediaan Data']), use_container_width=True, hide_index=True)
            
            @st.dialog("📈 Grafik Perbandingan Shift", width="large")
            def popup_grafik(tgl, shf, cd):
                if not cd or 'waktu' not in cd: return st.warning("Data observasi tidak tersedia.")
                fg = go.Figure()
                fg.add_trace(go.Scatter(x=cd['waktu'], y=cd['prediksi'], mode='lines+markers', name='Model', line=dict(color='blue', width=2)))
                fg.add_trace(go.Scatter(x=cd['waktu'], y=cd['observasi'], mode='lines+markers', name='HF Radar', line=dict(color='red', width=2)))
                fg.update_layout(title=f"{tgl} | {shf}", xaxis_title="Waktu", yaxis_title=satuan_teks, hovermode="x unified")
                st.plotly_chart(fg, use_container_width=True)

            col_p1, col_p2, _ = st.columns([2, 1, 1])
            opsi_shift = df_verif['Tanggal'].astype(str) + " - " + df_verif['Shift'].astype(str)
            with col_p1: pil = st.selectbox("Pilih jadwal:", options=opsi_shift.tolist(), label_visibility="collapsed")
            with col_p2:
                if st.button("Lihat Grafik", use_container_width=True):
                    idx = opsi_shift.tolist().index(pil)
                    popup_grafik(df_verif.iloc[idx]['Tanggal'], df_verif.iloc[idx]['Shift'], df_verif.iloc[idx]['chart_data'])
        else: st.warning("Belum ada data rekap.")
