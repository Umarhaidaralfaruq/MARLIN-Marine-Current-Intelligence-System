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
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap &copy; CARTO'
        }).addTo(map);

        var imageUrl = 'data:image/png;base64,__IMAGE_BASE64__';
        var imageBounds = [[__LAT_MIN__, __LON_MIN__], [__LAT_MAX__, __LON_MAX__]];
        L.imageOverlay(imageUrl, imageBounds, {opacity: 0.75}).addTo(map);

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

        var velocityLayer = L.velocityLayer({
            displayValues: true,
            displayOptions: { velocityType: 'Kecepatan Arus', displayPosition: 'bottomleft', displayEmptyString: 'Tidak ada data' },
            data: __DATA_JSON__,
            maxVelocity: __MAX_VELOCITY__,
            velocityScale: 0.05,           
            particleAge: 200,
            particleMultiplier: 0.002,     
            colorScale: ['#ffffff']        
        });
        velocityLayer.addTo(map);

        var legend = L.control({position: 'bottomright'});
        legend.onAdd = function (map) {
            var div = L.DomUtil.create('div', 'info legend');
            div.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
            div.style.padding = '10px 15px';
            div.style.borderRadius = '8px';
            div.style.border = '1px solid #ccc';
            div.style.boxShadow = '2px 2px 5px rgba(0,0,0,0.2)';
            div.style.fontFamily = 'Arial, sans-serif';
            div.innerHTML = 
                '<div style="margin-bottom: 8px; font-weight: bold; font-size: 12px; color: black; text-align: center;">Kecepatan Arus (__SATUAN__)</div>' +
                '<div style="background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ffa500, #ff0000, #4b0082); ' +
                'width: 250px; height: 15px; border-radius: 10px; box-shadow: inset 0 0 2px rgba(0,0,0,0.5);"></div>' +
                '<div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: 5px; color: black; font-weight: bold; padding: 0 5px;">' +
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
    df_hour = df_hour.copy()
    df_hour['lon'] = df_hour['lon'].round(4)
    df_hour['lat'] = df_hour['lat'].round(4)
    lons = np.sort(df_hour['lon'].unique())
    lats = np.sort(df_hour['lat'].unique())[::-1] 
    nx, ny = len(lons), len(lats)
    lo1, lo2 = float(lons[0]), float(lons[-1])
    la1, la2 = float(lats[0]), float(lats[-1])
    dx = (lo2 - lo1) / (nx - 1) if nx > 1 else 0
    dy = (la1 - la2) / (ny - 1) if ny > 1 else 0
    data_dict = df_hour.set_index(['lat', 'lon'])[['predicted_u', 'predicted_v']].to_dict('index')
    u_data, v_data = [], []
    for lat in lats:
        for lon in lons:
            if (lat, lon) in data_dict and not pd.isna(data_dict[(lat, lon)]['predicted_u']):
                u_data.append(float(data_dict[(lat, lon)]['predicted_u']))
                v_data.append(float(data_dict[(lat, lon)]['predicted_v']))
            else:
                u_data.append(0.0)
                v_data.append(0.0)
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

LAT_MIN, LAT_MAX = -8.209, -8.160084
LON_MIN, LON_MAX = 114.387089, 114.435215

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

# Fungsi Helper
def derajat_ke_arah_lengkap(deg):
    if pd.isna(deg): return ""
    arah = ['Utara', 'Timur Laut', 'Timur', 'Tenggara', 'Selatan', 'Barat Daya', 'Barat', 'Barat Laut']
    return arah[int((deg + 22.5) % 360 // 45)]

def derajat_ke_panah(deg):
    if pd.isna(deg): return ""
    panah = ['↑ U', '↗ TL', '→ T', '↘ TG', '↓ S', '↙ BD', '← B', '↖ BL']
    return panah[int((deg + 22.5) % 360 // 45)]

def get_p95_idx(group):
    p95 = group['predicted_v_max'].quantile(0.95)
    if pd.isna(p95): return group.index[0]
    return (group['predicted_v_max'] - p95).abs().idxmin()

list_of_files = get_available_baseruns()
if not list_of_files:
    st.error(f"❌ Tidak ada file prediksi ditemukan di folder: {FOLDER_OUTPUT}")
    st.stop()
file_options = {os.path.basename(f): f for f in list_of_files}

# ==========================================
# 1. HEADER & SYSTEM STATUS (COMPACT)
# ==========================================
logo_base64 = get_base64_logo(logo_seacast)

# Kita taruh header dan status bersebelahan
col_head1, col_head2 = st.columns([3, 1])

with col_head1:
    if logo_base64:
        # Header dibuat lebih compact, margin diperkecil
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
# PERSIAPAN DATA UNTUK SUMMARY (BACKGROUND PROCESS)
# ==========================================
# Default selections from query params or session state could go here, 
# for now we use the first file for calculations
selected_filename = st.session_state.get('sel_file', list(file_options.keys())[0])
satuan_pilihan = st.session_state.get('sel_unit', 'm/s')

df_spatial = load_data(file_options[selected_filename])
faktor_konversi = 1.94384 if satuan_pilihan == "Knot" else 1.0
satuan_teks = satuan_pilihan
batas_waspada = 1.2 * faktor_konversi 

df_spatial['predicted_v_max'] = df_spatial['predicted_v_max'] * faktor_konversi
df_spatial['predicted_u'] = df_spatial['predicted_u'] * faktor_konversi
df_spatial['predicted_v'] = df_spatial['predicted_v'] * faktor_konversi

df_filtered_area = df_spatial[
    (df_spatial['lat'] >= LAT_MIN) & (df_spatial['lat'] <= LAT_MAX) &
    (df_spatial['lon'] >= LON_MIN) & (df_spatial['lon'] <= LON_MAX)
]
idx_p95_trend = df_filtered_area.groupby('datetime_wib').apply(get_p95_idx)
df_trend_p95 = df_filtered_area.loc[idx_p95_trend].reset_index(drop=True)

# Kalkulasi KPI Operational
baris_max = df_trend_p95.loc[df_trend_p95['predicted_v_max'].idxmax()]
max_speed_global = baris_max['predicted_v_max']
waktu_ekstrem = baris_max['datetime_wib'].strftime('%H:%M WIB')
arah_dominan_global = derajat_ke_arah_lengkap(baris_max['arah_derajat'])
status_operasi = "⚠ WASPADA" if max_speed_global > batas_waspada else "🟢 NORMAL"

# Hitung Status Data
waktu_baserun_wib = df_spatial['datetime_wib'].min() - pd.Timedelta(hours=1)
waktu_sekarang_wib = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=7)
selisih_jam = (waktu_sekarang_wib - waktu_baserun_wib).total_seconds() / 3600

with col_head2:
    status_icon = "🟢" if selisih_jam <= 24 else "🔴"
    status_teks_data = "DATA TERUPDATE" if selisih_jam <= 24 else "TIDAK UPDATE"
    teks_warna = "#28a745" if selisih_jam <= 24 else "#dc3545"
    st.markdown(f"<div style='text-align: right; margin-top: 20px; font-weight: bold; color: {teks_warna};'>{status_icon} {status_teks_data}</div>", unsafe_allow_html=True)

st.divider()

# ==========================================
# 2. OPERATIONAL SUMMARY (P0)
# ==========================================
st.markdown("### OPERATIONAL SUMMARY")
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

col_kpi1.metric("Max Speed 24H", f"{max_speed_global:.2f} {satuan_teks}")
col_kpi2.metric("Peak Time", waktu_ekstrem)
col_kpi3.metric("Dominant Direction", arah_dominan_global)
# Mengakali warna metric bawaan Streamlit
col_kpi4.metric("Status", status_operasi, delta="Di atas Threshold" if max_speed_global > batas_waspada else "Aman", delta_color="inverse" if max_speed_global > batas_waspada else "normal")

st.divider()

# ==========================================
# 3. DATA CONTROL (P1)
# ==========================================
# Dibuat satu baris ringkas
st.markdown("**Data Control**")
col_dc1, col_dc2, col_dc3 = st.columns([2, 1, 1])

with col_dc1:
    selected_filename = st.selectbox("Prediction File", options=list(file_options.keys()), label_visibility="collapsed")
with col_dc2:
    if st.button("↻ Refresh Data", use_container_width=True):
        load_data.clear() 
        st.rerun()
with col_dc3:
    satuan_pilihan = st.radio("Unit:", ["m/s", "Knot"], horizontal=True, label_visibility="collapsed")

# Save state for reload
if st.session_state.get('sel_file') != selected_filename or st.session_state.get('sel_unit') != satuan_pilihan:
    st.session_state['sel_file'] = selected_filename
    st.session_state['sel_unit'] = satuan_pilihan
    st.rerun()

st.divider()

# ==========================================
# 4. PEMANTAUAN UTAMA (P0 - Map & Prakiraan)
# ==========================================
st.markdown("### PEMANTAUAN UTAMA")

# 1. CSS Trik: Sembunyikan angka bawaan di atas slider Streamlit biar nggak numpuk
st.markdown("""
<style>
    div[data-testid="stThumbValue"] { display: none; }
    div[data-testid="stTickBar"] { display: none; }
</style>
""", unsafe_allow_html=True)

col_map, col_chart = st.columns([1.3, 1])

# --- MAP SECTION (KIRI) ---
with col_map:
    waktu_unik = sorted(df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M').unique())
    waktu_sekarang_str = waktu_sekarang_wib.floor('h').strftime('%Y-%m-%d %H:%M')
    waktu_default = waktu_sekarang_str if waktu_sekarang_str in waktu_unik else waktu_unik[-1]

    st.markdown("<div style='margin-bottom: -15px; font-weight: bold;'>PETA SPASIAL ARUS</div>", unsafe_allow_html=True)
    
    selected_time_str = st.select_slider("Waktu Prediksi", options=waktu_unik, value=waktu_default, label_visibility="collapsed")
    
    bulan_indo = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MEI', 6: 'JUN', 7: 'JUL', 8: 'AGU', 9: 'SEP', 10: 'OKT', 11: 'NOV', 12: 'DES'}
    dt_obj = datetime.strptime(selected_time_str, '%Y-%m-%d %H:%M')
    formatted_time = f"{dt_obj.day:02d} {bulan_indo[dt_obj.month]} {dt_obj.year} • {dt_obj.strftime('%H:%M')} WIB"
    
    # Background dibikin semi-transparan agar masuk ke Light/Dark mode
    st.markdown(f"<div style='text-align: center; background-color: rgba(128, 128, 128, 0.1); padding: 4px; border-radius: 4px; font-weight: bold; font-size: 14px; margin-bottom: 5px;'>⏱️ {formatted_time}</div>", unsafe_allow_html=True)
    
    df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == selected_time_str]
    
    # KEMBALIKAN KE DINAMIS: Ambil batas asli dari data jam ini agar plot penuh
    lon_min_data, lon_max_data = float(df_hour['lon'].min()), float(df_hour['lon'].max())
    lat_min_data, lat_max_data = float(df_hour['lat'].min()), float(df_hour['lat'].max())

    @st.cache_data(show_spinner=False, ttl=3600)
    def generate_map_background(df_hour_json, lat_min, lat_max, lon_min, lon_max, vmax_plot):
        df_hour = pd.DataFrame(df_hour_json)
        triang = tri.Triangulation(df_hour['lon'], df_hour['lat'])
        batas_resolusi = 0.010 
        x_tri, y_tri = triang.x[triang.triangles], triang.y[triang.triangles]
        sisi_a = np.sqrt((x_tri[:, 0] - x_tri[:, 1])**2 + (y_tri[:, 0] - y_tri[:, 1])**2)
        sisi_b = np.sqrt((x_tri[:, 1] - x_tri[:, 2])**2 + (y_tri[:, 1] - y_tri[:, 2])**2)
        sisi_c = np.sqrt((x_tri[:, 2] - x_tri[:, 0])**2 + (y_tri[:, 2] - y_tri[:, 0])**2)
        triang.set_mask(np.any([sisi_a > batas_resolusi, sisi_b > batas_resolusi, sisi_c > batas_resolusi], axis=0))
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ['blue', 'green', 'yellow', 'orange', 'red'])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        ax.set_xlim(lon_min, lon_max) 
        ax.set_ylim(lat_min, lat_max)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        ax.tricontourf(triang, df_hour['predicted_v_max'], levels=np.linspace(0.0, vmax_plot, 1000), cmap=custom_cmap, vmin=0.0, vmax=vmax_plot, extend='max', alpha=0.9)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        buf.seek(0)
        plt.close(fig)
        img = Image.open(buf)
        r, g, b, a = img.split()
        a_blur = a.filter(ImageFilter.GaussianBlur(radius=10))
        a_final = a_blur.point(lambda p: 255 if p > 120 else 0).filter(ImageFilter.GaussianBlur(radius=1))
        img_rounded = Image.merge('RGBA', (r, g, b, a_final))
        buf_final = io.BytesIO()
        img_rounded.save(buf_final, format='PNG')
        return base64.b64encode(buf_final.getvalue()).decode('utf-8')

    df_hour_dict = df_hour[['lat', 'lon', 'predicted_v_max']].to_dict('list')
    vmax_dinamis = 3.0 * faktor_konversi
    
    # Passing batas data dinamis ke generator gambar
    image_base64 = generate_map_background(df_hour_dict, lat_min_data, lat_max_data, lon_min_data, lon_max_data, vmax_dinamis)

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

    lat_tengah, lon_tengah = (lat_min_data + lat_max_data) / 2, (lon_min_data + lon_max_data) / 2
    
    html_siap_render = HTML_TEMPLATE.replace("__DATA_JSON__", konversi_ke_velocity_json(df_hour)).replace("__LABELS_JSON__", json.dumps(labels_list)).replace("__IMAGE_BASE64__", image_base64).replace("__LAT_MIN__", str(lat_min_data)).replace("__LON_MIN__", str(lon_min_data)).replace("__LAT_MAX__", str(lat_max_data)).replace("__LON_MAX__", str(lon_max_data)).replace("__LAT_CENTER__", str(lat_tengah)).replace("__LON_CENTER__", str(lon_tengah)).replace("__MAX_VELOCITY__", str(8.0 * faktor_konversi)).replace("__SATUAN__", satuan_teks).replace("__LEG_1__", f"{round(1.0 * faktor_konversi)}").replace("__LEG_2__", f"{round(2.0 * faktor_konversi)}").replace("__LEG_3__", f"{round(3.0 * faktor_konversi)}").replace("__LEG_4__", f"{round(4.0 * faktor_konversi)}")
    
    components.html(html_siap_render, height=480)

# --- PRAKIRAAN SECTION (KANAN) ---
with col_chart:
    def get_shift_range_now(df_prediksi_24jam):
        last_hour = df_prediksi_24jam['datetime'].max().hour 
        return (12, 24, "Shift Malam") if last_hour < 12 else (0, 12, "Shift Pagi")

    start_hour_utc, end_hour_utc, shift_name = get_shift_range_now(df_spatial)
    mask_shift = (df_spatial['datetime'].dt.hour >= start_hour_utc) & (df_spatial['datetime'].dt.hour < end_hour_utc)
    df_shift_raw = df_spatial.loc[mask_shift & (df_spatial['lat'] >= LAT_MIN) & (df_spatial['lat'] <= LAT_MAX) & (df_spatial['lon'] >= LON_MIN) & (df_spatial['lon'] <= LON_MAX)]
    
    st.markdown(f"<div style='margin-bottom: 5px; font-weight: bold;'>PRAKIRAAN KECEPATAN & ARAH — 12 JAM</div>", unsafe_allow_html=True)

    if not df_shift_raw.empty:
        idx_p95_shift = df_shift_raw.groupby('datetime_wib').apply(get_p95_idx)
        df_shift = df_shift_raw.loc[idx_p95_shift].reset_index(drop=True)
        df_shift['arah_panah'] = df_shift['arah_derajat'].apply(derajat_ke_panah)
        
        baris_max_shift = df_shift.loc[df_shift['predicted_v_max'].idxmax()]
        max_speed_shift = baris_max_shift['predicted_v_max']
        waktu_max_shift = baris_max_shift['datetime_wib'].strftime('%H:%M WIB')
        
        arah_mode = df_shift['arah_derajat'].apply(derajat_ke_arah_lengkap).mode()
        arah_dominan = arah_mode[0] if not arah_mode.empty else "Variabel"
        
        # PERBAIKAN: Threshold baru dan Desain Alert transparan (hanya border yang berwarna)
        batas_waspada = 1.2 * faktor_konversi
        batas_awas = 2.0 * faktor_konversi
        
        if max_speed_shift >= batas_awas:
            border_color, icon, status_text = "#dc3545", "🔴", "AWAS"
        elif max_speed_shift >= batas_waspada:
            border_color, icon, status_text = "#ffc107", "🟡", "WASPADA"
        else:
            border_color, icon, status_text = "#28a745", "🟢", "NORMAL"
            
        st.markdown(f"""
        <div style="background-color: transparent; border: 1px solid rgba(128,128,128,0.2); border-left: 5px solid {border_color}; padding: 12px; border-radius: 6px; margin-bottom: 15px;">
            <div style="font-weight: bold; font-size: 15px; margin-bottom: 4px; color: {border_color};">{icon} {status_text}</div>
            <div style="font-size: 14px; opacity: 0.9;">
                Puncak: <strong>{max_speed_shift:.2f} {satuan_teks}</strong> ({waktu_max_shift}) <br>
                Arah Dominan: <strong>{arah_dominan}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig_shift = go.Figure()
        
        fig_shift.add_trace(go.Scatter(
            x=df_shift['datetime_wib'], y=df_shift['predicted_v_max'],
            mode='lines+markers', name='Prediksi Arus',
            line=dict(color='#0d6efd', width=3), 
            marker=dict(size=8, color='#0d6efd', line=dict(color='white', width=1)),
            customdata=df_shift['arah_panah'],
            hovertemplate="<b>%{x}</b><br>Kecepatan: %{y:.2f} " + satuan_teks + "<br>Arah: %{customdata}<extra></extra>"
        ))
        
        # PERBAIKAN: Update garis Waspada dan Awas
        fig_shift.add_hline(y=batas_waspada, line_dash="dash", line_color="#ffc107", 
                            annotation_text=f"Waspada > {batas_waspada:.2f}", 
                            annotation_position="top left",
                            annotation_font_color="#ffc107")
                            
        fig_shift.add_hline(y=batas_awas, line_dash="solid", line_color="#dc3545", 
                            annotation_text=f"Awas > {batas_awas:.2f}", 
                            annotation_position="top left",
                            annotation_font_color="#dc3545")

        fig_shift.update_layout(
            margin=dict(l=0, r=0, t=10, b=20), 
            xaxis_title=None,
            yaxis_title=f"Kecepatan ({satuan_teks})",
            height=340, 
            showlegend=False,
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        # Menambahkan grid samar biar nyambung sama dark mode
        fig_shift.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig_shift.update_xaxes(showgrid=False)
        
        st.plotly_chart(fig_shift, use_container_width=True)
    else:
        st.info("Data prediksi untuk shift ini belum tersedia.")

st.divider()

# ==========================================
# 5. PORT CONDITIONS (P1)
# ==========================================
st.markdown("### PORT CONDITIONS")
col_ket, col_gil = st.columns(2)

def derajat_ke_utara_selatan(deg):
    if pd.isna(deg): return ""
    return "Utara" if deg >= 270 or deg < 90 else "Selatan"

def buat_tabel_area(df_raw, lat_min, lat_max, lon_min, lon_max):
    df_area = df_raw[(df_raw['lat'] >= lat_min) & (df_raw['lat'] <= lat_max) & (df_raw['lon'] >= lon_min) & (df_raw['lon'] <= lon_max)]
    if df_area.empty: return pd.DataFrame()
    idx_p95_area = df_area.groupby('datetime_wib').apply(get_p95_idx)
    df_trend_area = df_area.loc[idx_p95_area].reset_index(drop=True)
    return pd.DataFrame({
        'Waktu (WIB)': df_trend_area['datetime_wib'].dt.strftime('%H:%M'),
        f'Speed ({satuan_teks})': df_trend_area['predicted_v_max'], 
        'Direction': df_trend_area['arah_derajat'].apply(derajat_ke_utara_selatan)
    })

def style_kecepatan(val):
    return 'color: #dc3545; font-weight: bold;' if val > batas_waspada else ''

df_shift_full = df_spatial.loc[mask_shift] if 'mask_shift' in locals() else pd.DataFrame()

with col_ket:
    st.markdown("**Ketapang**")
    tabel_ketapang = buat_tabel_area(df_shift_full, -8.209161, -8.151693, 114.38673, 114.414975) if not df_shift_full.empty else pd.DataFrame()
    if not tabel_ketapang.empty:
        styled_ket = tabel_ketapang.style.format({f'Speed ({satuan_teks})': "{:.2f}"}).map(style_kecepatan, subset=[f'Speed ({satuan_teks})'])
        st.dataframe(styled_ket, hide_index=True, use_container_width=True)
    else:
        st.caption("No data")

with col_gil:
    st.markdown("**Gilimanuk**")
    tabel_gilimanuk = buat_tabel_area(df_shift_full, -8.21003, -8.16198, 114.41354, 114.43416) if not df_shift_full.empty else pd.DataFrame()
    if not tabel_gilimanuk.empty:
        styled_gil = tabel_gilimanuk.style.format({f'Speed ({satuan_teks})': "{:.2f}"}).map(style_kecepatan, subset=[f'Speed ({satuan_teks})'])
        st.dataframe(styled_gil, hide_index=True, use_container_width=True)
    else:
        st.caption("No data")

st.divider()

# ==========================================
# 6. EXPORT (P3)
# ==========================================
st.markdown("### EXPORT")
# Compact view for Export
col_exp1, col_exp2, col_exp_empty = st.columns([1, 1, 2])

with col_exp1:
    # Dummy button logic based on original, kept compact
    if st.button("↓ Grafik 12 Jam", use_container_width=True):
        st.toast("Generating 12H Chart...")
with col_exp2:
    if st.button("↓ Grafik 24 Jam", use_container_width=True):
        st.toast("Generating 24H Chart...")

st.divider()

# ==========================================
# 7. MODEL VERIFICATION (P2)
# ==========================================
with st.expander("MODEL VERIFICATION", expanded=False):
    st.markdown("### Akurasi & Validasi Model")
    with st.spinner("Merekap riwayat seluruh baserun prediksi..."):
        
        # ---------------------------------------------------------
        # LANGKAH 1: PRE-LOAD DATA REAL
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
                
                if satuan_teks == "Knot":
                    df_real_raw['Real_Max_Speed_ms'] = df_real_raw['Real_Max_Speed_ms'] * 1.94384
                
                df_real_hourly = df_real_raw[['waktu_utc', 'Real_Max_Speed_ms']].copy()
                
                idx_max_real = df_real_raw.groupby(['Tanggal', 'Shift'])['Real_Max_Speed_ms'].idxmax()
                df_real_max = df_real_raw.loc[idx_max_real, ['Tanggal', 'Shift', 'waktu_utc', 'Real_Max_Speed_ms']].copy()
                
                df_real_max['Waktu_Max_Real'] = (df_real_max['waktu_utc'] + pd.Timedelta(hours=7)).dt.strftime('%H:%M WIB')
                df_real_max.rename(columns={'Real_Max_Speed_ms': 'Real_Max_Speed_Max'}, inplace=True)
                
                df_real_count = df_real_raw.groupby(['Tanggal', 'Shift']).size().reset_index(name='Jumlah_Jam')
                df_real_agg = pd.merge(df_real_max, df_real_count, on=['Tanggal', 'Shift'])
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
            
            mask_shift_local = (
                (df_temp['datetime'].dt.hour >= start_hour_utc) & 
                (df_temp['datetime'].dt.hour < end_hour_utc) &
                (df_temp['lat'] >= LAT_MIN) & (df_temp['lat'] <= LAT_MAX) &
                (df_temp['lon'] >= LON_MIN) & (df_temp['lon'] <= LON_MAX)
            )
            df_temp_shift = df_temp.loc[mask_shift_local]
            
            if df_temp_shift.empty:
                continue
                
            idx_p95 = df_temp_shift.groupby('datetime_wib').apply(get_p95_idx_local)
            df_p95_shift = df_temp_shift.loc[idx_p95].copy()
            
            tanggal_str = df_temp_shift['datetime'].dt.strftime('%Y-%m-%d').mode()[0]
            
            df_p95_shift['pred_speed'] = df_p95_shift['predicted_v_max'] * faktor_konversi
            baris_max_pred = df_p95_shift.loc[df_p95_shift['pred_speed'].idxmax()]
            
            max_pred_shift = baris_max_pred['pred_speed']
            waktu_max_pred_str = baris_max_pred['datetime_wib'].strftime('%H:%M WIB')
            
            shift_mbe = np.nan
            shift_rmse = np.nan
            raw_errors_shift = [] 
            chart_data_dict = {} 
            
            if not df_real_hourly.empty:
                df_chart = pd.merge(
                    df_p95_shift[['datetime', 'datetime_wib', 'pred_speed']], 
                    df_real_hourly[['waktu_utc', 'Real_Max_Speed_ms']], 
                    left_on='datetime', 
                    right_on='waktu_utc', 
                    how='left'
                )
                
                df_chart['jam_wib'] = df_chart['datetime_wib'].dt.strftime('%H:%M')
                chart_data_dict = {
                    'waktu': df_chart['jam_wib'].tolist(),
                    'prediksi': df_chart['pred_speed'].tolist(),
                    'observasi': df_chart['Real_Max_Speed_ms'].tolist()
                }
                
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
                'chart_data': chart_data_dict
            })
            
        df_rekap_pred = pd.DataFrame(rekap_prediksi)

        # ---------------------------------------------------------
        # LANGKAH 3: GABUNGKAN & HITUNG METRIK GLOBAL
        # ---------------------------------------------------------
        if not df_rekap_pred.empty and not df_real_agg.empty:
            df_verif = pd.merge(
                df_rekap_pred, 
                df_real_agg[['Tanggal', 'Shift', 'Real_Max_Speed_Max', 'Waktu_Max_Real', 'Kualitas_Data']], 
                on=['Tanggal', 'Shift'], 
                how='left' 
            )
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
                'Real_Max_Speed_Max': f'Observasi ({satuan_teks})',
                'Waktu_Max_Real': 'Waktu Puncak',
                'Kualitas_Data': 'Ketersediaan Data'
            }, inplace=True)
            
            df_verif['Ketersediaan Data'] = df_verif['Ketersediaan Data'].fillna("Data Tidak/Belum Tersedia")
            df_verif['Waktu Puncak'] = df_verif['Waktu Puncak'].fillna("-")
            df_verif = df_verif.sort_values(by=['Tanggal', 'Baserun'], ascending=[False, False])

            valid_data = df_verif.dropna(subset=[f'RMSE ({satuan_teks})', f'Observasi ({satuan_teks})'])
            
            if len(valid_data) > 0:
                all_errors = []
                for err_list in valid_data['raw_errors']:
                    if isinstance(err_list, list) and len(err_list) > 0:
                        all_errors.extend(err_list)
                        
                all_errors = np.array(all_errors)
                mbe_score = all_errors.mean() if len(all_errors) > 0 else 0.0
                rmse_score = np.sqrt((all_errors**2).mean()) if len(all_errors) > 0 else 0.0
                
                delta_teks_rmse = "Akurasi Baik" if rmse_score <= 0.4 else "Waspada Error"
                color_rmse = "normal" if rmse_score <= 0.4 else "inverse"
                
                if mbe_score > 0.1: 
                    delta_teks_mbe, color_mbe = "Cenderung Underestimate", "off"
                elif mbe_score < -0.1: 
                    delta_teks_mbe, color_mbe = "Cenderung Overestimate", "off"
                else:
                    delta_teks_mbe, color_mbe = "Sangat Presisi", "normal"
            else:
                mbe_score, rmse_score = 0.0, 0.0
                delta_teks_rmse = delta_teks_mbe = "Menunggu Observasi"
                color_rmse = color_mbe = "off"

            # Format angka untuk tabel
            df_verif[f'Prediksi Arus Maks ({satuan_teks})'] = df_verif[f'Prediksi Arus Maks ({satuan_teks})'].map('{:.2f}'.format)
            df_verif[f'Observasi ({satuan_teks})'] = df_verif[f'Observasi ({satuan_teks})'].apply(lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '-')
            df_verif[f'MBE ({satuan_teks})'] = df_verif[f'MBE ({satuan_teks})'].apply(lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '-')
            df_verif[f'RMSE ({satuan_teks})'] = df_verif[f'RMSE ({satuan_teks})'].apply(lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '-')

            kolom_fix = ['Tanggal', 'Shift', f'Prediksi Arus Maks ({satuan_teks})', f'Observasi ({satuan_teks})', 'Waktu Puncak', f'MBE ({satuan_teks})', f'RMSE ({satuan_teks})', 'Ketersediaan Data']
            df_verif_tampil = df_verif[kolom_fix]

            # FUNGSI POP-UP GRAFIK
            @st.dialog("📈 Grafik Perbandingan Shift", width="large")
            def tampilkan_popup_grafik(tanggal, shift, chart_data):
                if not chart_data or 'waktu' not in chart_data:
                    st.warning("Data observasi grafik tidak tersedia untuk shift ini.")
                    return
                    
                fig_popup = go.Figure()
                fig_popup.add_trace(go.Scatter(x=chart_data['waktu'], y=chart_data['prediksi'], mode='lines+markers', name='Prediksi Model', line=dict(color='blue', width=2)))
                fig_popup.add_trace(go.Scatter(x=chart_data['waktu'], y=chart_data['observasi'], mode='lines+markers', name='Observasi HF Radar', line=dict(color='red', width=2)))
                fig_popup.update_layout(
                    title=f"Tanggal: {tanggal} | Shift: {shift}",
                    xaxis_title="Waktu (WIB)", yaxis_title=f"Kecepatan Arus ({satuan_teks})",
                    hovermode="x unified", margin=dict(l=0, r=0, t=60, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5) 
                )
                st.plotly_chart(fig_popup, use_container_width=True)

            # ---------------------------------------------------------
            # LANGKAH 4: RENDER UI
            # ---------------------------------------------------------
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
            styled_df = df_verif_tampil.style.map(warnai_ketersediaan, subset=['Ketersediaan Data'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            col_pilih1, col_pilih2, empty = st.columns([2, 1, 1])
            with col_pilih1:
                opsi_pilihan = df_verif['Tanggal'].astype(str) + " - Shift " + df_verif['Shift'].astype(str)
                pilihan_shift = st.selectbox("Pilih jadwal grafik perbandingan:", options=opsi_pilihan.tolist(), label_visibility="collapsed")
            with col_pilih2:
                if st.button("Lihat Grafik", use_container_width=True):
                    idx = opsi_pilihan.tolist().index(pilihan_shift)
                    baris = df_verif.iloc[idx]
                    tampilkan_popup_grafik(baris['Tanggal'], baris['Shift'], baris['chart_data'])
                    
        else:
            st.warning("Belum ada data rekap prediksi yang dapat ditampilkan.")
