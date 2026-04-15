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
from branca.element import Template, MacroElement
from streamlit_folium import st_folium
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime

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
st.set_page_config(page_title="Dashboard Arus Selat Bali", page_icon="🌊", layout="wide")

st.title("🌊 Dashboard Prediksi Arus Laut Selat Bali")
st.markdown("Prediksi spasial dan deret waktu 24 jam ke depan berbasis Machine Learning (LSTM).")

# Koordinat Anchor (Sesuai script Anda)
ANCHOR_LAT = -8.20880659
ANCHOR_LON = 114.4111511 

# Sesuaikan dengan lokasi folder penyimpanan CSV dari config Anda
# Ubah path ini jika folder output Anda berbeda
FOLDER_OUTPUT = r"D:\Haidar\Prediksi Arus\03Hasil Prediksi\Arsip_Prediksi"

# ==========================================
# 2. FUNGSI PENCARI & PEMUAT DATA (SISTEM BASERUN)
# ==========================================
def get_available_baseruns():
    # Mencari semua file prediksi spasial
    search_pattern = os.path.join(FOLDER_OUTPUT, "prediksi_spasial*.csv")
    list_of_files = glob.glob(search_pattern)
    
    # Urutkan file berdasarkan waktu pembuatan, yang terbaru di atas
    list_of_files.sort(key=os.path.getctime, reverse=True)
    return list_of_files

@st.cache_data(ttl=3600) 
def load_data(file_path):
    df = pd.read_csv(file_path)
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
# LANGKAH A: Tampilkan Dropdown di Kolom 2 (Tengah)
# ---------------------------------------------------------
selected_filename = col_m2.selectbox(
    "🗂️ Pilih Baserun Data Prediksi:",
    options=list(file_options.keys())
)

# Tombol refresh mungil di bawah dropdown (di Kolom 2)
if col_m2.button("🔄 Perbarui Daftar File", use_container_width=True):
    load_data.clear() 
    st.rerun()

# ---------------------------------------------------------
# LANGKAH B: Memuat data sesuai pilihan dari Kolom 2
# ---------------------------------------------------------
selected_filepath = file_options[selected_filename]
df_spatial = load_data(selected_filepath)

waktu_unik = sorted(df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M').unique())

# ---------------------------------------------------------
# LANGKAH C: Menghitung & Menampilkan Metrik di Kolom 1 dan 3
# ---------------------------------------------------------
df_filtered_area = df_spatial[
    (df_spatial['lat'] >= LAT_MIN) & (df_spatial['lat'] <= LAT_MAX) &
    (df_spatial['lon'] >= LON_MIN) & (df_spatial['lon'] <= LON_MAX)
]

baris_max = df_filtered_area.loc[df_filtered_area['predicted_v_max'].idxmax()]
max_speed_global = baris_max['predicted_v_max']
waktu_ekstrem = baris_max['datetime_wib'].strftime('%H:%M WIB')

status_teks = "Waspada" if max_speed_global > 1.2 else "Aman"
keterangan_delta = f"Pukul {waktu_ekstrem} ({status_teks})"

# Menaruh Kartu Kecepatan di Kolom 1 (Kiri)
col_m1.metric(
    label="🌪️ Kecepatan Maksimum (24J ke Depan)", 
    value=f"{max_speed_global:.2f} m/s", 
    delta=keterangan_delta,
    delta_color="inverse" if max_speed_global > 1.2 else "normal"
)

# Menaruh Total Waktu Prediksi di Kolom 3 (Kanan)
col_m3.metric(label="⏱️ Total Waktu Prediksi", value=f"{len(waktu_unik)} Jam")

st.divider()

# ==========================================
# 5. LAYOUT UTAMA (PETA & GRAFIK)
# ==========================================
col1, col2 = st.columns([1.1, 1]) 

with col1:
    st.subheader("🗺️ Peta Prediksi Kecepatan Arus Penyeberangan Ketapang - Gilimanuk")
    
    # Pilih jam untuk ditampilkan di peta
    selected_time_str = st.select_slider(
        "⏱️ Geser untuk melihat perubahan arus per jam:", 
        options=waktu_unik,
        value=waktu_unik[0] # Set default penunjuk di jam paling awal
    )
    
    # Filter data sesuai jam yang dipilih
    df_hour = df_spatial[df_spatial['datetime_wib'].dt.strftime('%Y-%m-%d %H:%M') == selected_time_str]
    
    # Inisialisasi Peta Folium
    m = folium.Map(location=[-8.17, 114.41], zoom_start=12, tiles="CartoDB positron")
    
    # Titik Anchor
    folium.Marker(
        location=[ANCHOR_LAT, ANCHOR_LON],
        popup="<b>Titik Referensi (Anchor)</b>",
        icon=folium.Icon(color="green", icon="anchor", prefix='fa')
    ).add_to(m)
    
    # ==========================================
    # Membuat Colormap Kecepatan (Tanpa merendernya ke peta)
    # ==========================================
    v_min, v_max = df_spatial['predicted_v_max'].min(), df_spatial['predicted_v_max'].max()
    colormap = branca.colormap.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=v_min, vmax=v_max)
    
    # Plot titik grid spasial
    for idx, row in df_hour.reset_index(drop=True).iterrows():
        # Dapatkan warna dari colormap berdasarkan kecepatan
        speed_color = colormap(row['predicted_v_max'])
        
        # Ukuran lingkaran berdasarkan kecepatan (makin kencang makin besar)
        radius_size = max(3, row['predicted_v_max'] * 5)
        
        current_direction = "Utara" if row['predicted_v'] >= 0 else "Selatan"
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius_size,
            color=speed_color, 
            fill=True,
            fill_color=speed_color, 
            fill_opacity=0.8,
            tooltip=f"Kecepatan Arus: <b>{row['predicted_v_max']:.2f} m/s</b><br>Arah Arus: <b>{current_direction}</b>"
        ).add_to(m)
        
        # Gambarlah panah animasi di setiap titik ke-N agar peta tidak terlalu padat
        arrow_step = 1 
        if idx % arrow_step == 0:
            dash_val = max(1, 10 / max(0.1, row['predicted_v_max'])) 
            
            arrow_len = 0.003
            if row['predicted_v'] > 0: # Utara
                locations = [[row['lat'], row['lon']], [row['lat'] + arrow_len, row['lon']]]
            else: # Selatan
                locations = [[row['lat'], row['lon']], [row['lat'] - arrow_len, row['lon']]]
                
            folium.plugins.AntPath(locations, delay=1000, dash_array=[dash_val, dash_val], color='gray', weight=2, opacity=0.6).add_to(m)
    
    # ==========================================
    # LEGENDA KUSTOM DI KIRI BAWAH
    # ==========================================
    # Menghitung nilai tengah untuk tampilan legenda
    v_mid = (v_min + v_max) / 2
    
    legend_html = f'''
    {{% macro html(this, kwargs) %}}
    <div style="
        position: absolute; 
        bottom: 30px; left: 30px; 
        width: 250px; 
        background-color: rgba(255, 255, 255, 0.9); 
        z-index: 9999; 
        border-radius: 8px; 
        padding: 10px; 
        border: 1px solid grey;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        ">
        <div style="margin-bottom: 5px; font-weight: bold; font-size: 13px; color: black;">Kecepatan Arus (m/s)</div>
        <div style="background: linear-gradient(to right, blue, green, yellow, orange, red); 
                    width: 100%; height: 15px; border: 1px solid #333; border-radius: 3px;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: 5px; color: black; font-weight: bold;">
            <span>0.0</span>
            <span>1.2</span>
            <span>2.0</span>
            <span>>2.5</span>
        </div>
    </div>
    {{% endmacro %}}
    '''
    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)
        
    st_folium(m, use_container_width=True, height=650)
    
    st.caption("Warna titik menunjukkan Kecepatan Arus (m/s) sesuai Colormap. Panah abu-abu menunjukkan Arah Vektor Arus dengan animasi kecepatan.")

with col2:
    st.subheader("📈 Kecepatan Arus Maksimum 24 Jam Ke depan")
    
    # Mengelompokkan data per jam untuk mencari nilai tertinggi di seluruh grid
    df_trend = df_filtered_area.groupby('datetime_wib')['predicted_v_max'].max().reset_index()
    
    # Plotly Line Chart Utama (24 Jam)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_trend['datetime_wib'], 
        y=df_trend['predicted_v_max'],
        mode='lines+markers',
        name='Max Speed Grid',
        line=dict(color='black', width=2),
        marker=dict(size=8, color='crimson')
    ))
    
    # Garis Batas Waspada
    fig.add_hline(y=1.2, line_dash="dash", line_color="orange", annotation_text="Batas Waspada (1.2 m/s)")
    
    fig.update_layout(
        xaxis_title="Waktu (WIB)",
        yaxis_title="Kecepatan Maksimum (m/s)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # BARU: FOKUS OPERASIONAL SHIFT JAGA (LOGIKA UTC)
    # ==========================================
    # Fungsi penentu shift menggunakan waktu UTC asli Anda
    def get_shift_range_now(df_prediksi_24jam):
        last_time = df_prediksi_24jam['datetime'].max() # Kembali menggunakan UTC
        last_hour = last_time.hour 
        if last_hour < 12: 
            return 12, 24, "Shift Malam" # UTC 12-24 setara WIB 19:00 - 07:00
        else: 
            return 0, 12, "Shift Pagi"   # UTC 00-12 setara WIB 07:00 - 19:00

    # Dapatkan rentang jam UTC dan nama shift dari data asli (df_spatial)
    start_hour_utc, end_hour_utc, shift_name = get_shift_range_now(df_spatial)
    
    # Filter data HANYA untuk jam shift tersebut MENGGUNAKAN JAM UTC
    mask_shift = (df_spatial['datetime'].dt.hour >= start_hour_utc) & (df_spatial['datetime'].dt.hour < end_hour_utc)
    df_shift_raw = df_spatial.loc[mask_shift & 
                                  (df_spatial['lat'] >= LAT_MIN) & (df_spatial['lat'] <= LAT_MAX) &
                                  (df_spatial['lon'] >= LON_MIN) & (df_spatial['lon'] <= LON_MAX)]

    st.divider() # Garis pembatas estetik
    st.subheader(f"🎯 Kecepatan Arus Maksimum 12 Jam Ke depan ({shift_name})")

    if not df_shift_raw.empty:
        # Kelompokkan data shift yang sudah difilter untuk mencari max per jamnya
        df_shift = df_shift_raw.groupby('datetime_wib')['predicted_v_max'].max().reset_index()

        # 1. Cari baris dengan kecepatan maksimal KHUSUS di shift ini
        baris_max_shift = df_shift.loc[df_shift['predicted_v_max'].idxmax()]
        max_speed_shift = baris_max_shift['predicted_v_max']
        waktu_max_shift = baris_max_shift['datetime_wib'].strftime('%H:%M WIB')
        
        # 2. Peringatan Kecepatan (Warna Dinamis)
        if max_speed_shift > 1.2:
            st.error(f"🚨 **PERINGATAN DINI**  \nKecepatan Arus Maksimal **{max_speed_shift:.2f} m/s** diprediksi terjadi pada pukul **{waktu_max_shift}**.")
        else:
            st.success(f"✅ **ARUS AMAN**  \nKecepatan Arus Maksimal hanya **{max_speed_shift:.2f} m/s** pada pukul **{waktu_max_shift}**.")
            
        # 3. Grafik Zoom Khusus Shift (Tetap ditampilkan dalam WIB agar ramah pengguna)
        fig_shift = go.Figure()
        fig_shift.add_trace(go.Scatter(
            x=df_shift['datetime_wib'], 
            y=df_shift['predicted_v_max'],
            mode='lines+markers',
            name=f'Arus {shift_name}',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='crimson', line=dict(color='black', width=1))
        ))
        
        fig_shift.add_hline(y=1.2, line_dash="dash", line_color="orange", annotation_text="Batas Waspada (1.2 m/s)")
        
        # Menambahkan titik penanda (anotasi) pada jam puncak di grafik
        fig_shift.add_annotation(
            x=baris_max_shift['datetime_wib'],
            y=max_speed_shift,
            text=f"Maks: {max_speed_shift:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowcolor="red",
            ax=0,
            ay=-30,
            font=dict(color="red", weight="bold")
        )
        
        fig_shift.update_layout(
            xaxis_title="Jam Operasional (WIB)",
            yaxis_title="Kecepatan (m/s)",
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

# Fungsi ini berisi 100% setingan grafik final Anda tanpa ada yang diubah
def create_bmkg_report(df_source, title_text):
    df_plot = df_source.copy()
    if 'datetime_utc' not in df_plot.columns:
        df_plot['datetime_utc'] = df_plot['datetime_wib'] - pd.Timedelta(hours=7)
        
    # KUNCI SEJAJAR & ADA MARGIN:
    x_min_asli = df_plot['datetime_utc'].min()
    x_max_asli = df_plot['datetime_utc'].max()
    x_min_buffer = x_min_asli - pd.Timedelta(hours=1)
    x_max_buffer = x_max_asli + pd.Timedelta(hours=1)
    extended_ticks_utc = pd.date_range(start=x_min_buffer, end=x_max_buffer, freq='H')

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
    ax0.set_ylabel("Kecepatan Arus (m/s)", fontsize=13, weight='bold')
    y_max = max(2.0, df_plot['predicted_v_max'].max() + 0.5)
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

    # 6. Menambahkan Label Angka di Setiap Titik 
    for idx, row in df_plot.iterrows():
        label_text = f"{row['predicted_v_max']:.2f}"
        ax0.annotate(label_text, (row['datetime_utc'], row['predicted_v_max']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, weight='bold')

    # 7. Metadata Header BMKG
    header_text_left = "Kantor Layanan Informasi Meteorologi\nMaritim Ketapang, BMKG\nBMKG"
    fig_bmkg.text(0.11, 0.95, header_text_left, fontsize=12, weight='bold', color='black', ha='left', va='top', linespacing=1.4)
    
    issue_time_wib = datetime.now() 
    issue_date_str = issue_time_wib.strftime('%d %B %Y')
    issue_time_str = issue_time_wib.strftime('%H:%M WIB')
    fig_bmkg.text(0.95, 0.95, f"Dikeluarkan pada {issue_time_str}, {issue_date_str}", fontsize=12, weight='bold', color='black', ha='right', va='top')

    # 8. Logo BMKG 
    LOGO_PATH = "logo_bmkg.png"
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