# =============================================================================
# FILE: app.py
# FUNGSI: Dashboard Interaktif Arus Selat Bali (Redesign Phase 1: Structure)
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

# ==========================================
# [BLOK KONFIGURASI GLOBAL] 
# Mencegah Hardcode, Magic Numbers, dan Mengatur Metode Statistik
# ==========================================

# 1. KOORDINAT AREA UTAMA (Bounding Box Keseluruhan)
AREA_LAT_MIN, AREA_LAT_MAX = -8.209161, -8.160084
AREA_LON_MIN, AREA_LON_MAX = 114.387089, 114.435215

# 2. KOORDINAT PELABUHAN KETAPANG 
KET_LAT_MIN = max(-8.209161, AREA_LAT_MIN)
KET_LAT_MAX = min(-8.151693, AREA_LAT_MAX)
KET_LON_MIN = max(114.38673, AREA_LON_MIN)
KET_LON_MAX = min(114.414975, AREA_LON_MAX)

# 3. KOORDINAT PELABUHAN GILIMANUK
GIL_LAT_MIN = max(-8.21003, AREA_LAT_MIN)
GIL_LAT_MAX = min(-8.160084, AREA_LAT_MAX)
GIL_LON_MIN = max(114.41354, AREA_LON_MIN)
GIL_LON_MAX = min(114.435215, AREA_LON_MAX)

# 4. AMBANG BATAS KECEPATAN (Satuan m/s)
BATAS_WASPADA_MS = 1.2
BATAS_AWAS_MS = 2.0
FAKTOR_KNOT = 1.94384

# 5. METODE STATISTIK PENGAMBILAN DATA SPASIAL
METODE_STATISTIK = "MAX"  # Ubah menjadi "P95" jika ingin menggunakan persentil 95

def ambil_index_target(group):
    """Fungsi sentral untuk mengambil index berdasarkan metode statistik yang dipilih."""
    if METODE_STATISTIK == "P95":
        p95 = group['predicted_v_max'].quantile(0.95)
        if pd.isna(p95): return group.index[0]
        return (group['predicted_v_max'] - p95).abs().idxmin()
    else: # Default ke MAX
        return group['predicted_v_max'].idxmax()

# ==========================================
# [BAGIAN AWAL TETAP SAMA: HTML Template, Fungsi Velocity, Konfigurasi Tema]
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
        
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap &copy; CARTO'
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
            velocityScale: 0.08,            
            particleAge: 120,           
            particleMultiplier: 0.001, 
            colorScale: ['#ffffff']        
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
    lo1 = float(df_hour['lon'].min())
    lo2 = float(df_hour['lon'].max())
    la1 = float(df_hour['lat'].max()) 
    la2 = float(df_hour['lat'].min()) 
    
    nx, ny = 60, 60
    grid_lon = np.linspace(lo1, lo2, nx)
    grid_lat = np.linspace(la1, la2, ny) 
    dx = (lo2 - lo1) / (nx - 1)
    dy = (la1 - la2) / (ny - 1) 
    
    triang = tri.Triangulation(df_hour['lon'], df_hour['lat'])
    batas_resolusi = 0.011
    
    x_tri = triang.x[triang.triangles]
    y_tri = triang.y[triang.triangles]
    sisi_a = np.sqrt((x_tri[:, 0] - x_tri[:, 1])**2 + (y_tri[:, 0] - y_tri[:, 1])**2)
    sisi_b = np.sqrt((x_tri[:, 1] - x_tri[:, 2])**2 + (y_tri[:, 1] - y_tri[:, 2])**2)
    sisi_c = np.sqrt((x_tri[:, 2] - x_tri[:, 0])**2 + (y_tri[:, 2] - y_tri[:, 0])**2)
    
    mask = np.any([sisi_a > batas_resolusi, sisi_b > batas_resolusi, sisi_c > batas_resolusi], axis=0)
    triang.set_mask(mask)
    
    interp_u = tri.LinearTriInterpolator(triang, df_hour['predicted_u'])
    interp_v = tri.LinearTriInterpolator(triang, df_hour['predicted_v'])
    
    Grid_lon, Grid_lat = np.meshgrid(grid_lon, grid_lat)
    U_grid = interp_u(Grid_lon, Grid_lat)
    V_grid = interp_v(Grid_lon, Grid_lat)
    
    u_data, v_data = [], []
    for i in range(ny):
        for j in range(nx):
            u_val = U_grid[i, j]
            v_val = V_grid[i, j]
            if np.ma.is_masked(u_val) or np.isnan(u_val):
                u_data.append(None)
                v_data.append(None)
            else:
                u_data.append(float(u_val))
                v_data.append(float(v_val))
                
    velocity_json = [
        {"header": {"parameterCategory": 2, "parameterNumber": 2, "nx": nx, "ny": ny, "lo1": lo1, "lo2": lo2, "la1": la1, "la2": la2, "dx": dx, "dy": dy}, "data": u_data},
        {"header": {"parameterCategory": 2, "parameterNumber": 3, "nx": nx, "ny": ny, "lo1": lo1, "lo2": lo2, "la1": la1, "la2": la2, "dx": dx, "dy": dy}, "data": v_data}
    ]
    return json.dumps(velocity_json)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# ==========================================
# SETUP HALAMAN WEBSITE
# ==========================================
logo_filename = "logo_bmkg.png"
logo_seacast = "logo_bmkg2.png"
ikon = Image.open(logo_seacast) if os.path.exists(logo_seacast) else (Image.open("logo_bmkg.png") if os.path.exists("logo_bmkg.png") else "🌊")

st.set_page_config(page_title="MARLIN Dashboard", page_icon=ikon, layout="wide")
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .viewerBadge {display: none !important;}</style>""", unsafe_allow_html=True)

def get_base64_logo(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None

BASE_DIR = Path(__file__).parent.resolve()
FOLDER_OUTPUT = BASE_DIR / "Arsip_Prediksi"

def get_available_baseruns():
    search_pattern = os.path.join(FOLDER_OUTPUT, "prediksi_spasial*.csv")
    return sorted(glob.glob(search_pattern), reverse=True)

@st.cache_data(ttl=3600) 
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={'datetime_prediction': 'datetime', 'latitude': 'lat', 'longitude': 'lon', 'current_speed_ms': 'predicted_v_max', 'v_velocity_ms': 'predicted_v', 'u_velocity_ms': 'predicted_u', 'current_direction_deg': 'arah_derajat'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime_wib'] = df['datetime'] + pd.Timedelta(hours=7)
    return df

def derajat_ke_arah_lengkap(deg):
    if pd.isna(deg): return ""
    arah = ['Utara', 'Timur Laut', 'Timur', 'Tenggara', 'Selatan', 'Barat Daya', 'Barat', 'Barat Laut']
    return arah[int((deg + 22.5) % 360 // 45)]

def derajat_ke_panah(deg):
    if pd.isna(deg): return ""
    panah = ['↑ U', '↗ TL', '→ T', '↘ TG', '↓ S', '↙ BD', '← B', '↖ BL']
    return panah[int((deg + 22.5) % 360 // 45)]

# ==========================================
# STATUS ARUS (SATU SUMBER KEBENARAN)
# ==========================================
STATUS_STYLE = {
    "AWAS":    {"icon": "🔴", "bg": "#dc3545", "text_color": "#ffffff"},
    "WASPADA": {"icon": "🟡", "bg": "#ffc107", "text_color": "#000000"},
    "NORMAL":  {"icon": "🟢", "bg": "#28a745", "text_color": "#ffffff"},
}
for _status_key, _style in STATUS_STYLE.items():
    _style["label"] = f"{_style['icon']} {_status_key}"

def get_status_arus(speed, batas_waspada, batas_awas):
    if pd.isna(speed):
        status = "NORMAL"
    elif speed >= batas_awas:
        status = "AWAS"
    elif speed >= batas_waspada:
        status = "WASPADA"
    else:
        status = "NORMAL"

    style = STATUS_STYLE[status]
    return {
        "status": status,
        "icon": style["icon"],
        "bg": style["bg"],
        "text_color": style["text_color"],
        "label": style["label"],
    }

list_of_files = get_available_baseruns()
if not list_of_files:
    st.error(f"❌ Tidak ada file prediksi ditemukan di folder: {FOLDER_OUTPUT}")
    st.stop()

def format_nama_baserun(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'prediksi_spasial_(\d{8})_(\d{4})\.csv', filename)
    if match:
        tanggal_str = match.group(1) 
        jam_str = match.group(2)     
        waktu_utc = pd.to_datetime(f"{tanggal_str}{jam_str}", format="%Y%m%d%H%M")
        waktu_wib = waktu_utc + pd.Timedelta(hours=7)
        return waktu_wib.strftime("%Y-%m-%d %H:%M WIB")
    return filename 

file_options = {format_nama_baserun(f): f for f in list_of_files}

# ==========================================
# 1. HEADER & SYSTEM STATUS
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

# ==========================================
# PERSIAPAN DATA SUMMARY
# ==========================================
selected_filename = st.session_state.get('sel_file', list(file_options.keys())[0])
if selected_filename not in file_options:
    selected_filename = list(file_options.keys())[0]

satuan_pilihan = st.session_state.get('sel_unit', 'm/s')
df_spatial = load_data(file_options[selected_filename])
faktor_konversi = FAKTOR_KNOT if satuan_pilihan == "Knot" else 1.0
satuan_teks = satuan_pilihan

batas_waspada = BATAS_WASPADA_MS * faktor_konversi 
batas_awas = BATAS_AWAS_MS * faktor_konversi

df_spatial['predicted_v_max'] = df_spatial['predicted_v_max'] * faktor_konversi
df_spatial['predicted_u'] = df_spatial['predicted_u'] * faktor_konversi
df_spatial['predicted_v'] = df_spatial['predicted_v'] * faktor_konversi

def get_shift_range_now(df_prediksi_24jam):
    last_hour = df_prediksi_24jam['datetime'].max().hour 
    return (12, 24, "Shift Malam") if last_hour < 12 else (0, 12, "Shift Pagi")

start_hour_utc, end_hour_utc, shift_name_global = get_shift_range_now(df_spatial)

# Agregasi Spasial Menggunakan Variabel Fungsi `ambil_index_target`
df_filtered_area_24 = df_spatial[
    (df_spatial['lat'] >= AREA_LAT_MIN) & (df_spatial['lat'] <= AREA_LAT_MAX) &
    (df_spatial['lon'] >= AREA_LON_MIN) & (df_spatial['lon'] <= AREA_LON_MAX)
]
idx_target_trend_24 = df_filtered_area_24.groupby('datetime_wib').apply(ambil_index_target)
df_trend_stat = df_filtered_area_24.loc[idx_target_trend_24].reset_index(drop=True)

waktu_sekarang_wib = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=7)

if not df_trend_stat.empty:
    df_trend_stat['selisih_waktu'] = abs(df_trend_stat['datetime_wib'] - waktu_sekarang_wib)
    baris_saat_ini = df_trend_stat.loc[df_trend_stat['selisih_waktu'].idxmin()]
    
    kecepatan_saat_ini = baris_saat_ini['predicted_v_max']
    jam_saat_ini_str = baris_saat_ini['datetime_wib'].strftime('%H:%M WIB')
    
    waktu_utc = df_trend_stat['datetime_wib'] - pd.Timedelta(hours=7)
    jam_terakhir = waktu_utc.max().hour
    start_h = 12 if jam_terakhir < 12 else 0
    end_h = 24 if jam_terakhir < 12 else 12
    
    mask_shift = (waktu_utc.dt.hour >= start_h) & (waktu_utc.dt.hour < end_h)
    df_12j = df_trend_stat[mask_shift]
    
    if not df_12j.empty:
        baris_max = df_12j.loc[df_12j['predicted_v_max'].idxmax()] # Puncak Waktu Tetap max absolut
        max_speed_global = baris_max['predicted_v_max']
        waktu_ekstrem = baris_max['datetime_wib'].strftime('%H:%M WIB')
        waktu_puncak_dt = baris_max['datetime_wib']
        selisih_detik = (waktu_puncak_dt - waktu_sekarang_wib).total_seconds()
        
        if selisih_detik < -1800: 
            teks_selisih, warna_selisih = "Sudah terlewati", "#28a745"
        elif -1800 <= selisih_detik <= 1800: 
            teks_selisih, warna_selisih = "Sedang terjadi", "#dc3545"
        else:
            jam_lagi = int(selisih_detik // 3600)
            menit_lagi = int((selisih_detik % 3600) // 60)
            teks_selisih = f"{jam_lagi}j {menit_lagi}m lagi" if jam_lagi > 0 else f"{menit_lagi}m lagi"
            warna_selisih = "#ffc107"
            
        info_puncak_html = f"<div style='font-size: 0.9rem; color: {warna_selisih}; font-weight: 600; margin-top: 5px;'>{teks_selisih}</div>"
    else:
        max_speed_global = 0.0
        waktu_ekstrem, info_puncak_html = "--:--", ""
else:
    kecepatan_saat_ini, max_speed_global = 0.0, 0.0
    jam_saat_ini_str, waktu_ekstrem, info_puncak_html = "--:--", "--:--", ""

waktu_baserun_wib = df_spatial['datetime_wib'].min() - pd.Timedelta(hours=1)
selisih_jam = (waktu_sekarang_wib - waktu_baserun_wib).total_seconds() / 3600

with col_head2:
    status_icon_data, status_teks_data = ("🟢", "DATA UPDATE") if selisih_jam <= 24 else ("🔴", "TIDAK UPDATE")
    teks_warna_data = "#28a745" if selisih_jam <= 24 else "#dc3545"
    st.markdown(f"<div style='text-align: right; margin-top: 20px; font-weight: bold; color: {teks_warna_data};'>{status_icon_data} {status_teks_data}</div>", unsafe_allow_html=True)

st.divider()

# ==========================================
# 2. RINGKASAN OPERASIONAL (P0)
# ==========================================
status_ringkasan = get_status_arus(kecepatan_saat_ini, batas_waspada, batas_awas)

html_summary = f"""<div style="background: linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.7) 100%); border: 1px solid rgba(148,163,184,0.2); border-radius: 8px; padding: 20px 25px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);">
<h3 style="color: #cbd5e1; margin-top: 0; margin-bottom: 20px; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px;">Ringkasan Operasional</h3>
<div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
<div style="flex: 1; min-width: 150px; border-right: 1px solid rgba(148,163,184,0.2); padding-right: 15px;">
<div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 5px;">Kecepatan Saat Ini <span style="font-size: 0.75rem; color: #cbd5e1;">({jam_saat_ini_str})</span></div>
<div style="color: #f8fafc; font-size: 2.2rem; font-weight: bold; line-height: 1;">{kecepatan_saat_ini:.2f} <span style="font-size: 1.1rem; color: #94a3b8; font-weight: normal;">{satuan_teks}</span></div>
</div>
<div style="flex: 1; min-width: 150px; border-right: 1px solid rgba(148,163,184,0.2); padding-right: 15px;">
<div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 5px;">Kecepatan Puncak - 12 Jam</div>
<div style="color: #f8fafc; font-size: 2.2rem; font-weight: bold; line-height: 1;">{max_speed_global:.2f} <span style="font-size: 1.1rem; color: #94a3b8; font-weight: normal;">{satuan_teks}</span></div>
</div>
<div style="flex: 1; min-width: 150px; border-right: 1px solid rgba(148,163,184,0.2); padding-right: 15px;">
<div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 5px;">Waktu Puncak - 12 Jam</div>
<div style="color: #f8fafc; font-size: 2.2rem; font-weight: bold; line-height: 1;">{waktu_ekstrem[:5]} <span style="font-size: 1.1rem; color: #94a3b8; font-weight: normal;">WIB</span></div>
{info_puncak_html}
</div>
<div style="flex: 1; min-width: 150px;">
<div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 10px;">Status Kondisi Saat Ini</div>
<div style="display: inline-block; background-color: {status_ringkasan['bg']}; color: {status_ringkasan['text_color']}; padding: 6px 16px; border-radius: 20px; font-size: 1.1rem; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">{status_ringkasan['icon']} {status_ringkasan['status']}</div>
</div>
</div>
</div>"""
st.markdown(html_summary, unsafe_allow_html=True)

# ==========================================
# 3. KONTROL DATA
# ==========================================
st.markdown("**Kontrol Data**")
col_dc1, col_dc2, col_dc3 = st.columns([2, 1, 1])

with col_dc1:
    selected_filename = st.selectbox("File Prediksi", options=list(file_options.keys()), label_visibility="collapsed")
with col_dc2:
    if st.button("↻ Perbarui Data", use_container_width=True):
        load_data.clear() 
        st.rerun()
with col_dc3:
    satuan_pilihan = st.radio("Satuan:", ["m/s", "Knot"], horizontal=True, label_visibility="collapsed")

if st.session_state.get('sel_file') != selected_filename or st.session_state.get('sel_unit') != satuan_pilihan:
    st.session_state['sel_file'], st.session_state['sel_unit'] = selected_filename, satuan_pilihan
    st.rerun()

st.divider()

# ==========================================
# 4. PEMANTAUAN UTAMA
# ==========================================
st.markdown("### PEMANTAUAN ARUS SPASIAL DAN TEMPORAL")
st.markdown("<style>.stSlider, .stSlider div, .stSlider span, .stSlider p { color: transparent !important; } .stSlider [data-testid='stThumbValue'], .stSlider [data-testid='stTickBarMin'], .stSlider [data-testid='stTickBarMax'], .stSlider [data-testid='stTickBar'] { display: none !important; opacity: 0 !important; }</style>", unsafe_allow_html=True)

col_map, col_chart = st.columns([1.3, 1])

with col_map:
    waktu_unik = sorted(df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M').unique())
    waktu_sekarang_str = waktu_sekarang_wib.floor('h').strftime('%Y-%m-%d %H:%M')
    waktu_default = waktu_sekarang_str if waktu_sekarang_str in waktu_unik else waktu_unik[-1]

    st.markdown("<div style='margin-bottom: 5px; font-weight: bold;'>PETA SPASIAL ARUS</div>", unsafe_allow_html=True)
    selected_time_str = st.select_slider("Waktu Prediksi", options=waktu_unik, value=waktu_default, label_visibility="collapsed")
    
    bulan_indo = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MEI', 6: 'JUN', 7: 'JUL', 8: 'AGU', 9: 'SEP', 10: 'OKT', 11: 'NOV', 12: 'DES'}
    dt_obj = datetime.strptime(selected_time_str, '%Y-%m-%d %H:%M')
    formatted_time = f"{dt_obj.day:02d} {bulan_indo[dt_obj.month]} {dt_obj.year} • {dt_obj.strftime('%H:%M')} WIB"
    st.markdown(f"<div style='text-align: center; background-color: rgba(128, 128, 128, 0.1); padding: 4px; border-radius: 4px; font-weight: bold; font-size: 14px; margin-bottom: 5px;'>⏱️ {formatted_time}</div>", unsafe_allow_html=True)
    
    df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == selected_time_str]
    lon_min_data, lon_max_data = float(df_hour['lon'].min()), float(df_hour['lon'].max())
    lat_min_data, lat_max_data = float(df_hour['lat'].min()), float(df_hour['lat'].max())

    @st.cache_data(show_spinner=False, ttl=3600)
    def generate_map_background(df_hour_json, lat_min_data, lat_max_data, lon_min_data, lon_max_data, vmax_plot):
        df_hour = pd.DataFrame(df_hour_json)
        triang = tri.Triangulation(df_hour['lon'], df_hour['lat'])
        batas_resolusi = 0.011
        x_tri, y_tri = triang.x[triang.triangles], triang.y[triang.triangles]
        sisi_a = np.sqrt((x_tri[:, 0] - x_tri[:, 1])**2 + (y_tri[:, 0] - y_tri[:, 1])**2)
        sisi_b = np.sqrt((x_tri[:, 1] - x_tri[:, 2])**2 + (y_tri[:, 1] - y_tri[:, 2])**2)
        sisi_c = np.sqrt((x_tri[:, 2] - x_tri[:, 0])**2 + (y_tri[:, 2] - y_tri[:, 0])**2)
        triang.set_mask(np.any([sisi_a > batas_resolusi, sisi_b > batas_resolusi, sisi_c > batas_resolusi], axis=0))
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ffa500', '#ff0000', '#4b0082'])
    
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        ax.set_xlim(lon_min_data, lon_max_data) 
        ax.set_ylim(lat_min_data, lat_max_data)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        ax.tricontourf(triang, df_hour['predicted_v_max'], 
               levels=np.linspace(0.0, vmax_plot, 1000), # <-- Cukup 100 aja udah mulus banget
               cmap=custom_cmap, 
               vmin=0.0, vmax=vmax_plot, 
               extend='max', 
               alpha=0.9,
               antialiased=True)
    
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    vmax_dinamis = 4.0 * faktor_konversi
    image_base64 = generate_map_background(df_hour[['lat', 'lon', 'predicted_v_max']].to_dict('list'), lat_min_data, lat_max_data, lon_min_data, lon_max_data, vmax_dinamis)

    placed_labels, labels_list = [], []
    
    # TRIK 1: Potong data dari awal, ambil setiap baris ke-3 saja (ubah angka 3 kalau mau makin renggang)
    df_label_renggang = df_hour.iloc[::3] 
    
    for idx, row in df_label_renggang.iterrows():
        kec = row['predicted_v_max']
        if pd.isna(kec): continue
        lat, lon = float(row['lat']), float(row['lon'])
        
        # TRIK 2: Besarkan radius radius bentrokan (clash) dari 0.01 menjadi 0.015 atau 0.02
        if not any(np.sqrt((lat - plat)**2 + (lon - plon)**2) < 0.015 for (plat, plon) in placed_labels):
            labels_list.append({"lat": lat, "lon": lon, "val": f"{kec:.2f}"})
            placed_labels.append((lat, lon))
        if pd.isna(kec): continue
        lat, lon = float(row['lat']), float(row['lon'])
        if not any(np.sqrt((lat - plat)**2 + (lon - plon)**2) < 0.01 for (plat, plon) in placed_labels):
            labels_list.append({"lat": lat, "lon": lon, "val": f"{kec:.2f}"})
            placed_labels.append((lat, lon))

    lat_tengah, lon_tengah = (lat_min_data + lat_max_data) / 2, (lon_min_data + lon_max_data) / 2
    html_siap_render = HTML_TEMPLATE.replace("__DATA_JSON__", konversi_ke_velocity_json(df_hour)).replace("__LABELS_JSON__", json.dumps(labels_list)).replace("__IMAGE_BASE64__", image_base64).replace("__LAT_MIN__", str(lat_min_data)).replace("__LON_MIN__", str(lon_min_data)).replace("__LAT_MAX__", str(lat_max_data)).replace("__LON_MAX__", str(lon_max_data)).replace("__LAT_CENTER__", str(lat_tengah)).replace("__LON_CENTER__", str(lon_tengah)).replace("__MAX_VELOCITY__", str(8.0 * faktor_konversi)).replace("__SATUAN__", satuan_teks).replace("__LEG_1__", f"{round(1.0 * faktor_konversi)}").replace("__LEG_2__", f"{round(2.0 * faktor_konversi)}").replace("__LEG_3__", f"{round(3.0 * faktor_konversi)}").replace("__LEG_4__", f"{round(4.0 * faktor_konversi)}")
    components.html(html_siap_render, height=480)

with col_chart:
    mask_shift = (df_spatial['datetime'].dt.hour >= start_hour_utc) & (df_spatial['datetime'].dt.hour < end_hour_utc)
    df_shift_raw = df_spatial.loc[mask_shift & (df_spatial['lat'] >= AREA_LAT_MIN) & (df_spatial['lat'] <= AREA_LAT_MAX) & (df_spatial['lon'] >= AREA_LON_MIN) & (df_spatial['lon'] <= AREA_LON_MAX)]
    st.markdown(f"<div style='margin-bottom: 5px; font-weight: bold;'>PRAKIRAAN KECEPATAN & ARAH — 12 JAM</div>", unsafe_allow_html=True)

    if not df_shift_raw.empty:
        # PENGGUNAAN FUNGSI SENTRAL: apply(ambil_index_target)
        idx_target_shift = df_shift_raw.groupby('datetime_wib').apply(ambil_index_target)
        df_shift = df_shift_raw.loc[idx_target_shift].reset_index(drop=True)
        df_shift['arah_panah'] = df_shift['arah_derajat'].apply(derajat_ke_panah)
        
        baris_max_shift = df_shift.loc[df_shift['predicted_v_max'].idxmax()]
        max_speed_shift = baris_max_shift['predicted_v_max']
        waktu_max_shift = baris_max_shift['datetime_wib'].strftime('%H:%M WIB')
        arah_mode = df_shift['arah_derajat'].apply(derajat_ke_arah_lengkap).mode()
        arah_dominan = arah_mode[0] if not arah_mode.empty else "Variabel"
        
        status_prakiraan = get_status_arus(max_speed_shift, batas_waspada, batas_awas)

        html_alert = f"""<div style="background: linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.7) 100%); border: 1px solid rgba(148,163,184,0.2); border-left: 5px solid {status_prakiraan['bg']}; padding: 15px 20px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <div style="color: #cbd5e1; font-size: 0.95rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Status Prakiraan</div>
        <div style="background-color: {status_prakiraan['bg']}; color: {status_prakiraan['text_color']}; padding: 4px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">{status_prakiraan['icon']} {status_prakiraan['status']}</div>
        </div>
        <div style="display: flex; gap: 20px;">
        <div>
        <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 3px;">Puncak Kecepatan</div>
        <div style="color: #f8fafc; font-size: 1.2rem; font-weight: bold;">{max_speed_shift:.2f} <span style="font-size: 0.9rem; color: #94a3b8; font-weight: normal;">{satuan_teks}</span> <span style="font-size: 0.85rem; color: #94a3b8; font-weight: normal;">({waktu_max_shift})</span></div>
        </div>
        <div>
        <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 3px;">Arah Dominan</div>
        <div style="color: #f8fafc; font-size: 1.2rem; font-weight: bold;">{arah_dominan}</div>
        </div>
        </div>
        </div>"""
        st.markdown(html_alert, unsafe_allow_html=True)

        fig_shift = go.Figure()
        fig_shift.add_trace(go.Scatter(
            x=df_shift['datetime_wib'], y=df_shift['predicted_v_max'], mode='lines+markers', name='Prediksi Arus',
            line=dict(color='#0d6efd', width=3), marker=dict(size=8, color='#0d6efd', line=dict(color='white', width=1)),
            customdata=df_shift['arah_panah'], hovertemplate="<b>%{x}</b><br>Kecepatan: %{y:.2f} " + satuan_teks + "<br>Arah: %{customdata}<extra></extra>"
        ))
        
        ymax_plot = df_shift['predicted_v_max'].max() + (0.3 * faktor_konversi)
        if ymax_plot >= batas_waspada:
            fig_shift.add_hline(y=batas_waspada, line_dash="dash", line_color=STATUS_STYLE["WASPADA"]["bg"], annotation_text=f"Waspada > {batas_waspada:.2f}", annotation_position="top left", annotation_font_color=STATUS_STYLE["WASPADA"]["bg"])
            ymax_plot = max(ymax_plot, batas_waspada + (0.2 * faktor_konversi))
        if ymax_plot >= batas_awas:
            fig_shift.add_hline(y=batas_awas, line_dash="solid", line_color=STATUS_STYLE["AWAS"]["bg"], annotation_text=f"Awas > {batas_awas:.2f}", annotation_position="top left", annotation_font_color=STATUS_STYLE["AWAS"]["bg"])
            ymax_plot = max(ymax_plot, batas_awas + (0.2 * faktor_konversi))

        fig_shift.update_layout(
            margin=dict(l=0, r=0, t=10, b=20), 
            xaxis_title=None, 
            yaxis_title=f"Kecepatan ({satuan_teks})",
            yaxis=dict(range=[0, ymax_plot]), 
            height=340, 
            showlegend=False, 
            hovermode="x unified",
            
            # --- TAMBAHKAN KODE INI UNTUK MENGHILANGKAN PUTIH-PUTIH ---
            hoverlabel=dict(
                bgcolor="#1e293b",       # Background box gelap (slate tua)
                font_color="#f8fafc",    # Teks warna putih terang
                bordercolor="#334155"    # Garis tepi / border warna gelap biar nyaru
            ),
            # ----------------------------------------------------------
            
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_shift.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig_shift.update_xaxes(
            showgrid=False,
            showspikes=True,             # Pastikan garis hover aktif
            spikecolor="#94a3b8",        # Ubah warna garis jadi abu-abu kalem (tidak putih mencolok)
            spikethickness=1,            # Ketebalan garis
            spikedash="dash"             # Format garis putus-putus
        )
        st.plotly_chart(fig_shift, use_container_width=True)
    else:
        st.info("Data prediksi untuk shift ini belum tersedia.")

st.divider()

# ==========================================
# 5. KONDISI PELABUHAN
# ==========================================
st.markdown("### KONDISI PELABUHAN")
col_ket, col_gil = st.columns(2)

def derajat_ke_utara_selatan(deg):
    return "" if pd.isna(deg) else ("Utara" if deg >= 270 or deg < 90 else "Selatan")

def buat_tabel_area(df_raw, lat_min, lat_max, lon_min, lon_max):
    df_area = df_raw[(df_raw['lat'] >= lat_min) & (df_raw['lat'] <= lat_max) & (df_raw['lon'] >= lon_min) & (df_raw['lon'] <= lon_max)]
    if df_area.empty: return pd.DataFrame()
    
    # PENGGUNAAN FUNGSI SENTRAL: apply(ambil_index_target)
    idx_target_area = df_area.groupby('datetime_wib').apply(ambil_index_target)
    df_trend_area = df_area.loc[idx_target_area].reset_index(drop=True)
    
    def format_speed_badge(val):
        info = get_status_arus(val, batas_waspada, batas_awas)
        return f"{val:.2f}" if info["status"] == "NORMAL" else f"{val:.2f} {info['label']}"

    return pd.DataFrame({
        'Waktu (WIB)': df_trend_area['datetime_wib'].dt.strftime('%H:%M'),
        f'Kecepatan ({satuan_teks})': df_trend_area['predicted_v_max'].apply(format_speed_badge), 
        'Arah Dominan': df_trend_area['arah_derajat'].apply(derajat_ke_utara_selatan)
    })

def style_tabel_peringatan(val):
    if isinstance(val, str):
        if STATUS_STYLE["AWAS"]["label"] in val: return f'color: {STATUS_STYLE["AWAS"]["bg"]}; font-weight: bold;'
        elif STATUS_STYLE["WASPADA"]["label"] in val: return f'color: {STATUS_STYLE["WASPADA"]["bg"]}; font-weight: bold;'
    return ''

df_shift_full = df_shift_raw if 'df_shift_raw' in locals() else pd.DataFrame()

with col_ket:
    st.markdown("**Area Pelabuhan Ketapang**")
    tabel_ketapang = buat_tabel_area(df_shift_full, KET_LAT_MIN, KET_LAT_MAX, KET_LON_MIN, KET_LON_MAX)
    if not tabel_ketapang.empty:
        st.dataframe(tabel_ketapang.style.map(style_tabel_peringatan, subset=[f'Kecepatan ({satuan_teks})']), hide_index=True, use_container_width=True)
    else:
        st.caption("Data area Ketapang tidak tersedia untuk shift ini.")

with col_gil:
    st.markdown("**Area Pelabuhan Gilimanuk**")
    tabel_gilimanuk = buat_tabel_area(df_shift_full, GIL_LAT_MIN, GIL_LAT_MAX, GIL_LON_MIN, GIL_LON_MAX)
    if not tabel_gilimanuk.empty:
        st.dataframe(tabel_gilimanuk.style.map(style_tabel_peringatan, subset=[f'Kecepatan ({satuan_teks})']), hide_index=True, use_container_width=True)
    else:
        st.caption("Data area Gilimanuk tidak tersedia untuk shift ini.")

st.divider()

# ==========================================
# 6. UNDUH GRAFIK
# ==========================================
st.markdown("### UNDUH GRAFIK")

def create_bmkg_report(df_source, title_text):
    df_plot = df_source.copy()
    if 'datetime_utc' not in df_plot.columns: df_plot['datetime_utc'] = df_plot['datetime_wib'] - pd.Timedelta(hours=7)
        
    x_min_buffer = df_plot['datetime_utc'].min() - pd.Timedelta(hours=1)
    x_max_buffer = df_plot['datetime_utc'].max() + pd.Timedelta(hours=1)
    frekuensi = '2h' if ((x_max_buffer - x_min_buffer).total_seconds() / 3600) > 15 else 'h'
    extended_ticks_utc = pd.date_range(start=x_min_buffer, end=x_max_buffer, freq=frekuensi)

    fig_bmkg, ax0 = plt.subplots(figsize=(14, 7), dpi=100)
    fig_bmkg.subplots_adjust(left=0.08, right=0.95, top=0.7, bottom=0.15) 
    line_color, grid_color = 'black', '#A9CCE3' 
    
    ax0.set_title(title_text, fontsize=16, pad=60)
    ax0.plot(df_plot['datetime_utc'], df_plot['predicted_v_max'], color=line_color, linewidth=2, marker='o', markersize=6, markerfacecolor=line_color, markeredgecolor='white')
    ax0.set_ylabel(f"Kecepatan Arus ({satuan_teks})", fontsize=13, weight='bold')
    ax0.set_ylim(0, max(BATAS_AWAS_MS * faktor_konversi, df_plot['predicted_v_max'].max() + (0.5 * faktor_konversi)))
    ax0.grid(color=grid_color, linestyle='--', linewidth=0.8)
    ax0.set_xlim(x_min_buffer, x_max_buffer)
    ax0.set_xlabel("Waktu (UTC)", fontsize=13, weight='bold', labelpad=10)
    ax0.set_xticks(extended_ticks_utc)
    ax0.set_xticklabels([t.strftime('%H:%M') for t in extended_ticks_utc], rotation=0, ha='center', fontsize=11, weight='bold')

    ax0_twin = ax0.twiny()
    ax0_twin.set_xlabel("Waktu (WIB)", fontsize=13, weight='bold', labelpad=10)
    ax0_twin.set_xlim(x_min_buffer, x_max_buffer)
    ax0_twin.set_xticks(extended_ticks_utc)
    ax0_twin.set_xticklabels([t.strftime('%H:%M') for t in (extended_ticks_utc + pd.Timedelta(hours=7))], rotation=0, ha='center', fontsize=11, weight='bold')

    for idx, row in df_plot.iterrows():
        deg = row.get('arah_derajat', pd.NA)
        panah_bmkg = "" if pd.isna(deg) else ("↑ U" if deg >= 270 or deg < 90 else "↓ S")
        ax0.annotate(f"{panah_bmkg}\n{row['predicted_v_max']:.2f}", (row['datetime_utc'], row['predicted_v_max']), textcoords="offset points", xytext=(0,15), ha='center', fontsize=10, weight='bold')

    fig_bmkg.text(0.11, 0.95, "Kantor Layanan Informasi Meteorologi Maritim\nKetapang - Gilimanuk\nBanyuwangi", fontsize=12, weight='bold', color='black', ha='left', va='top', linespacing=1.4)
    bulan_indo_bmkg = {1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'}
    issue_time_wib = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=7)
    fig_bmkg.text(0.95, 0.95, f"Dikeluarkan pada {issue_time_wib.strftime('%H:%M')} WIB, {issue_time_wib.strftime('%d')} {bulan_indo_bmkg[issue_time_wib.month]} {issue_time_wib.strftime('%Y')}", fontsize=12, weight='bold', color='black', ha='right', va='top')

    logo_path = BASE_DIR / "Logo_BMKG.png"
    if not logo_path.exists(): logo_path = BASE_DIR / "logo_bmkg2.png"
    if not logo_path.exists(): logo_path = BASE_DIR / "logo_bmkg.png"
    if logo_path.exists():
        try:
            ax0.add_artist(AnnotationBbox(OffsetImage(plt.imread(str(logo_path)), zoom=0.08), (0.02, 0.97), xycoords='figure fraction', frameon=False, box_alignment=(0, 1)))
        except: pass 
    return fig_bmkg

col_exp1, col_exp2, col_exp_empty = st.columns([1, 1, 2])
with col_exp1:
    if st.button("Unduh Grafik 12 Jam", use_container_width=True):
        with st.spinner("Memproses grafik..."):
            if 'df_shift' in locals() and not df_shift.empty:
                fig_12j = create_bmkg_report(df_shift, f"Prediksi Kecepatan Arus Ketapang-Gilimanuk 12 Jam Kedepan (Mode: {METODE_STATISTIK})")
                with io.BytesIO() as buf:
                    fig_12j.savefig(buf, format="png", dpi=150)
                    st.download_button("Klik Untuk Menyimpan (12 Jam)", data=buf.getvalue(), file_name=f"Grafik_Arus_12J_{METODE_STATISTIK}.png", mime="image/png", use_container_width=True)
                plt.close(fig_12j)
            else: st.error("Data 12 Jam belum tersedia.")

with col_exp2:
    if st.button("Unduh Grafik 24 Jam", use_container_width=True):
        with st.spinner("Memproses grafik..."):
            if 'df_trend_stat' in locals() and not df_trend_stat.empty:
                fig_24j = create_bmkg_report(df_trend_stat, f"Prediksi Kecepatan Arus Ketapang-Gilimanuk 24 Jam Kedepan (Mode: {METODE_STATISTIK})")
                with io.BytesIO() as buf:
                    fig_24j.savefig(buf, format="png", dpi=150)
                    st.download_button("Klik Untuk Menyimpan (24 Jam)", data=buf.getvalue(), file_name=f"Grafik_Arus_24J_{METODE_STATISTIK}.png", mime="image/png", use_container_width=True)
                plt.close(fig_24j)
            else: st.error("Data 24 Jam belum tersedia.")

st.divider()

# ==========================================
# 7. MODEL VERIFICATION
# ==========================================
with st.expander("MODEL VERIFICATION", expanded=False):
    st.markdown("### Akurasi & Validasi Model")
    with st.spinner("Merekap riwayat seluruh baserun prediksi..."):
        
        path_verif_csv = BASE_DIR / "Hasil_Validasi" / "Validasi_Real_Max_12H.csv" 
        df_real_hourly, df_real_agg = pd.DataFrame(), pd.DataFrame()
        
        if path_verif_csv.exists():
            df_real_raw = pd.read_csv(path_verif_csv)
            if 'Waktu_UTC' in df_real_raw.columns:
                df_real_raw['waktu_utc'] = pd.to_datetime(df_real_raw['Waktu_UTC'])
                df_real_raw['Tanggal'] = df_real_raw['waktu_utc'].dt.strftime('%Y-%m-%d')
                df_real_raw['Shift'] = df_real_raw['waktu_utc'].dt.hour.apply(lambda h: "Shift Pagi" if h < 12 else "Shift Malam")
                if satuan_teks == "Knot": df_real_raw['Real_Max_Speed_ms'] *= FAKTOR_KNOT
                
                df_real_hourly = df_real_raw[['waktu_utc', 'Real_Max_Speed_ms']].copy()
                idx_max_real = df_real_raw.groupby(['Tanggal', 'Shift'])['Real_Max_Speed_ms'].idxmax()
                df_real_max = df_real_raw.loc[idx_max_real, ['Tanggal', 'Shift', 'waktu_utc', 'Real_Max_Speed_ms']].copy()
                df_real_max['Waktu_Max_Real'] = (df_real_max['waktu_utc'] + pd.Timedelta(hours=7)).dt.strftime('%H:%M WIB')
                df_real_max.rename(columns={'Real_Max_Speed_ms': 'Real_Max_Speed_Max'}, inplace=True)
                
                df_real_agg = pd.merge(df_real_max, df_real_raw.groupby(['Tanggal', 'Shift']).size().reset_index(name='Jumlah_Jam'), on=['Tanggal', 'Shift'])
                df_real_agg = df_real_agg[df_real_agg['Jumlah_Jam'] >= 6].copy()
                df_real_agg['Kualitas_Data'] = df_real_agg['Jumlah_Jam'].apply(lambda jml: ">75% (Sangat Baik)" if jml >= 9 else "50% - 75% (Cukup)")

        rekap_prediksi = []
        for filepath in list_of_files:
            df_temp = load_data(filepath)
            start_hour_utc, end_hour_utc, shift_nama = get_shift_range_now(df_temp)
            
            mask_shift_local = ((df_temp['datetime'].dt.hour >= start_hour_utc) & (df_temp['datetime'].dt.hour < end_hour_utc) & (df_temp['lat'] >= AREA_LAT_MIN) & (df_temp['lat'] <= AREA_LAT_MAX) & (df_temp['lon'] >= AREA_LON_MIN) & (df_temp['lon'] <= AREA_LON_MAX))
            df_temp_shift = df_temp.loc[mask_shift_local]
            if df_temp_shift.empty: continue
                
            # PENGGUNAAN FUNGSI SENTRAL: apply(ambil_index_target)
            idx_target_local = df_temp_shift.groupby('datetime_wib').apply(ambil_index_target)
            df_stat_shift = df_temp_shift.loc[idx_target_local].copy()
            
            tanggal_str = df_temp_shift['datetime'].dt.strftime('%Y-%m-%d').mode()[0]
            df_stat_shift['pred_speed'] = df_stat_shift['predicted_v_max'] * faktor_konversi
            
            baris_max_pred = df_stat_shift.loc[df_stat_shift['pred_speed'].idxmax()]
            shift_mbe, shift_rmse, raw_errors_shift, chart_data_dict = np.nan, np.nan, [], {}
            
            if not df_real_hourly.empty:
                df_chart = pd.merge(df_stat_shift[['datetime', 'datetime_wib', 'pred_speed']], df_real_hourly[['waktu_utc', 'Real_Max_Speed_ms']], left_on='datetime', right_on='waktu_utc', how='left')
                chart_data_dict = {'waktu': df_chart['datetime_wib'].dt.strftime('%H:%M').tolist(), 'prediksi': df_chart['pred_speed'].tolist(), 'observasi': df_chart['Real_Max_Speed_ms'].tolist()}
                
                df_merged = df_chart.dropna(subset=['Real_Max_Speed_ms', 'pred_speed'])
                if not df_merged.empty:
                    errors = df_merged['Real_Max_Speed_ms'] - df_merged['pred_speed']
                    shift_mbe, shift_rmse, raw_errors_shift = errors.mean(), np.sqrt((errors**2).mean()), errors.tolist()
            
            rekap_prediksi.append({
                'Tanggal': tanggal_str, 'Shift': shift_nama, f'Prediksi Arus ({METODE_STATISTIK} - {satuan_teks})': baris_max_pred['pred_speed'],
                f'MBE ({satuan_teks})': shift_mbe, f'RMSE ({satuan_teks})': shift_rmse, 'Baserun': os.path.basename(filepath),
                'raw_errors': raw_errors_shift, 'chart_data': chart_data_dict
            })
            
        df_rekap_pred = pd.DataFrame(rekap_prediksi)

        if not df_rekap_pred.empty and not df_real_agg.empty:
            df_verif = pd.merge(df_rekap_pred, df_real_agg[['Tanggal', 'Shift', 'Real_Max_Speed_Max', 'Waktu_Max_Real', 'Kualitas_Data']], on=['Tanggal', 'Shift'], how='left')
            mask_kosong = df_verif['Real_Max_Speed_Max'].isna()
            df_verif.loc[mask_kosong, [f'MBE ({satuan_teks})', f'RMSE ({satuan_teks})', 'Waktu_Max_Real']] = np.nan
        else:
            df_verif = df_rekap_pred.copy() if not df_rekap_pred.empty else pd.DataFrame()
            if not df_verif.empty: df_verif['Real_Max_Speed_Max'], df_verif['Waktu_Max_Real'], df_verif['Kualitas_Data'] = np.nan, np.nan, "Tidak Ada Data CSV"

        if not df_verif.empty:
            df_verif.rename(columns={'Real_Max_Speed_Max': f'Observasi ({satuan_teks})', 'Waktu_Max_Real': 'Waktu Puncak', 'Kualitas_Data': 'Ketersediaan Data'}, inplace=True)
            df_verif['Ketersediaan Data'], df_verif['Waktu Puncak'] = df_verif['Ketersediaan Data'].fillna("Data Tidak/Belum Tersedia"), df_verif['Waktu Puncak'].fillna("-")
            df_verif = df_verif.sort_values(by=['Tanggal', 'Baserun'], ascending=[False, False])

            valid_data = df_verif.dropna(subset=[f'RMSE ({satuan_teks})', f'Observasi ({satuan_teks})'])
            if len(valid_data) > 0:
                all_errors = np.array([err for err_list in valid_data['raw_errors'] if isinstance(err_list, list) for err in err_list])
                mbe_score, rmse_score = (all_errors.mean(), np.sqrt((all_errors**2).mean())) if len(all_errors) > 0 else (0.0, 0.0)
                delta_teks_rmse, color_rmse = ("Akurasi Baik", "normal") if rmse_score <= 0.4 else ("Waspada Error", "inverse")
                if mbe_score > 0.1: delta_teks_mbe, color_mbe = "Cenderung Underestimate", "off"
                elif mbe_score < -0.1: delta_teks_mbe, color_mbe = "Cenderung Overestimate", "off"
                else: delta_teks_mbe, color_mbe = "Sangat Presisi", "normal"
            else:
                mbe_score, rmse_score, delta_teks_rmse, delta_teks_mbe, color_rmse, color_mbe = 0.0, 0.0, "Menunggu Observasi", "Menunggu Observasi", "off", "off"

            df_verif[f'Prediksi Arus ({METODE_STATISTIK} - {satuan_teks})'] = df_verif[f'Prediksi Arus ({METODE_STATISTIK} - {satuan_teks})'].map('{:.2f}'.format)
            for col in [f'Observasi ({satuan_teks})', f'MBE ({satuan_teks})', f'RMSE ({satuan_teks})']:
                df_verif[col] = df_verif[col].apply(lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '-')

            df_verif_tampil = df_verif[['Tanggal', 'Shift', f'Prediksi Arus ({METODE_STATISTIK} - {satuan_teks})', f'Observasi ({satuan_teks})', 'Waktu Puncak', f'MBE ({satuan_teks})', f'RMSE ({satuan_teks})', 'Ketersediaan Data']]

            @st.dialog("📈 Grafik Perbandingan Shift", width="large")
            def tampilkan_popup_grafik(tanggal, shift, chart_data):
                if not chart_data or 'waktu' not in chart_data: return st.warning("Data observasi grafik tidak tersedia untuk shift ini.")
                fig_popup = go.Figure()
                fig_popup.add_trace(go.Scatter(x=chart_data['waktu'], y=chart_data['prediksi'], mode='lines+markers', name='Prediksi Model', line=dict(color='blue', width=2)))
                fig_popup.add_trace(go.Scatter(x=chart_data['waktu'], y=chart_data['observasi'], mode='lines+markers', name='Observasi HF Radar', line=dict(color='red', width=2)))
                fig_popup.update_layout(title=f"Tanggal: {tanggal} | Shift: {shift} (Mode: {METODE_STATISTIK})", xaxis_title="Waktu (WIB)", yaxis_title=f"Kecepatan Arus ({satuan_teks})", hovermode="x unified", margin=dict(l=0, r=0, t=60, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
                st.plotly_chart(fig_popup, use_container_width=True)

            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1: st.metric("Shift Available", f"{len(valid_data)}", f"Dari total {len(df_verif)}", delta_color="normal")
            with col_v2: st.metric("MBE", f"{mbe_score:.2f} {satuan_teks}", delta_teks_mbe, delta_color=color_mbe)
            with col_v3: st.metric("RMSE", f"{rmse_score:.2f} {satuan_teks}", delta_teks_rmse, delta_color=color_rmse)

            def warnai_ketersediaan(val):
                if isinstance(val, str):
                    if 'Sangat Baik' in val: return 'color: #28a745;'
                    elif 'Cukup' in val: return 'color: #ffc107;'
                    elif 'Menunggu' in val or 'Tidak' in val: return 'color: #dc3545;'
                return ''

            st.markdown("**Tabel Riwayat Validasi**")
            st.dataframe(df_verif_tampil.style.map(warnai_ketersediaan, subset=['Ketersediaan Data']), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            col_pilih1, col_pilih2, empty = st.columns([2, 1, 1])
            with col_pilih1:
                opsi_pilihan = df_verif['Tanggal'].astype(str) + " - Shift " + df_verif['Shift'].astype(str)
                pilihan_shift = st.selectbox("Pilih jadwal grafik perbandingan:", options=opsi_pilihan.tolist(), label_visibility="collapsed")
            with col_pilih2:
                if st.button("Lihat Grafik", use_container_width=True):
                    idx = opsi_pilihan.tolist().index(pilihan_shift)
                    tampilkan_popup_grafik(df_verif.iloc[idx]['Tanggal'], df_verif.iloc[idx]['Shift'], df_verif.iloc[idx]['chart_data'])
        else:
            st.warning("Belum ada data rekap prediksi yang dapat ditampilkan.")
