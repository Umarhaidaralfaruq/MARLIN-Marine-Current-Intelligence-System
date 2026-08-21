# =============================================================================
# FILE: 01_data_collector.py
# FUNGSI: Mengunduh HF Radar & Web Scraping AWS secara Otomatis
# =============================================================================

import os
import sys
import csv
import json
import time
import requests
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from config import (
    DATA_DIR, PATHS,
    AWS_STATION_ID, AWS_LOGIN_URL, AWS_USERNAME, AWS_PASSWORD, AWS_START_DEFAULT, AWS_DATA_URL, HFRADAR_URL
)

# Definisikan nama file output spesifik untuk modul ini
NC_LATEST_FILE = PATHS["nc_file_arus"]


# =============================================================================
# 1. UNDUH & EKSTRAK HF RADAR (ANCHOR POINT)
# =============================================================================
def proses_hf_radar():
    print("\n" + "="*50)
    print("🌊 MULAI PROSES AKUISISI HF RADAR (DOWNLOAD & CONCAT)")
    print("="*50)

    url_hf = (HFRADAR_URL)
    all_files = []

    # ---------------------------------------------------------------------------
    # 1. MENGUNDUH FILE TERBARU DARI BMKG
    # ---------------------------------------------------------------------------
    try:
        print("📥 1. Mengunduh file terbaru dari BMKG...")
        response = requests.get(url_hf)
        response.raise_for_status()
        
        with open(NC_LATEST_FILE, "wb") as file:
            file.write(response.content)
            
        # Penanganan nama file jika menggunakan pathlib atau string biasa
        nama_file = NC_LATEST_FILE.name if hasattr(NC_LATEST_FILE, 'name') else os.path.basename(NC_LATEST_FILE)
        print(f"✅ File berhasil disimpan di: {nama_file}")
        
        # Tambahkan file yang baru diunduh ke antrean proses
        all_files.append(str(NC_LATEST_FILE))
        
    except Exception as error:
        print(f"❌ Gagal mengunduh file terbaru (Melanjutkan dengan file lokal): {error}")

# =============================================================================
# 2. SCRAPING AWS & UPDATE DATABASE (CORRECTED)
# =============================================================================
def proses_aws():
    print("\n" + "="*50)
    print("🌦️ MULAI PROSES SCRAPING AWS (PASUT & ANGIN)")
    print("="*50)

    db_path = PATHS["aws_database"]
    
    if db_path.exists():
        df_old = pd.read_excel(db_path)
        df_old["datetime"] = pd.to_datetime(df_old["datetime"], dayfirst=True)
        START_TIME = df_old["datetime"].max() + timedelta(minutes=1)
        print(f"📂 Database AWS ditemukan. Melanjutkan penarikan data dari: {START_TIME}")
    else:
        df_old = None
        START_TIME = AWS_START_DEFAULT
        print(f"⚠️ Database AWS belum ada. Mulai penarikan awal dari: {START_TIME}")

    END_TIME = datetime.now()
    print(f"🎯 Target unduh sampai dengan: {END_TIME}")

    # Login AWS (Selenium)
    print("\n🤖 Login AWS menggunakan Webdriver...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--log-level=3") 
    # Tambahkan User-Agent di Selenium untuk berjaga-jaga
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(AWS_LOGIN_URL)
    time.sleep(2)

    try:
        driver.find_element(By.NAME,"username").send_keys(AWS_USERNAME)
        driver.find_element(By.NAME,"password").send_keys(AWS_PASSWORD)
        driver.find_element(By.XPATH,"//button[@type='submit']").click()
        time.sleep(3) # Tunggu loading selesai
        cookies = driver.get_cookies()
        driver.quit()
        print("✅ Login AWS Berhasil! Cookie didapatkan.")
    except Exception as e:
        driver.quit()
        print(f"❌ Login AWS Gagal! Periksa kredensial di config.py. Error: {e}")
        return

    # Session Requests
    session = requests.Session()
    # PENTING: Tambahkan Header User-Agent agar tidak di-block oleh firewall server AWS
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01"
    })
    
    for cookie in cookies:
        session.cookies.set(cookie["name"], cookie["value"])

    print("\n📥 Mulai mengunduh data chunking (per 7 hari)...")
    delta = timedelta(days=7)
    current = START_TIME
    df_download = []

    while current <= END_TIME:
        end_chunk = min(current + delta, END_TIME)
        
        # PERBAIKAN FORMAT WAKTU: Ubah '/' menjadi spasi. 
        # (Jika server API Anda SPESIFIK butuh '/', kembalikan ke '/' tapi biasanya ini biang keroknya)
        start_str = current.strftime("%Y-%m-%d/%H:%M:%S") 
        end_str = end_chunk.strftime("%Y-%m-%d/%H:%M:%S")
        
        url = f"{AWS_DATA_URL}/{AWS_STATION_ID}/{start_str}/{end_str}"
        print(f"   -> Request: {start_str} s/d {end_str}")
        print(f"      URL: {url}") # Print URL untuk dicek manual jika gagal

        try:
            r = session.get(url, timeout=30)
            
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:
                    # Menangkap error jika responsnya HTML (misal: dilempar balik ke halaman login)
                    print(f"      ⚠️ Gagal parse JSON! Respons server kemungkinan HTML (Sesi tidak valid/Cookie ditolak).")
                    print(f"      Preview Respons: {r.text[:150]}")
                    current += delta
                    continue
                
                if isinstance(data, dict) and "aaData" in data:
                    data = data["aaData"]
                    
                if isinstance(data, list) and len(data) > 0:
                    df_download.append(pd.DataFrame(data))
                    print(f"      ✅ Berhasil mengunduh {len(data)} baris data.")
                else:
                    print("      ⚠️ Request sukses, tapi data 'aaData' kosong [] dari server.")
            else:
                print(f"      ⚠️ Error HTTP Status: {r.status_code}. Server menolak request.")
                
        except Exception as e:
            print(f"      ⚠️ Error unduh chunk: {e}")
            
        current += delta

    if len(df_download) == 0:
        print("\n✅ Tidak ada data pasut/angin baru dari server AWS saat ini (atau request ditolak).")
        return

    # Gabung dan bersihkan data
    print("\n🔄 Memproses dan menggabungkan data...")
    df = pd.concat(df_download, ignore_index=True)
    df["waktu"] = pd.to_datetime(df["twaktu"], dayfirst=True).dt.strftime("%d/%m/%Y %H:%M:%S")
    
    kolom_final = ["waktu", "windspeed", "winddir", "temp", "rh", "pressure", "rain", "solrad", "netrad", "watertemp", "waterlevel", "ta_min", "ta_max", "pancilevel", "pancitemp"]
    
    # Pastikan hanya mengambil kolom yang ada dari server, hindari KeyError
    kolom_tersedia = [k for k in kolom_final if k in df.columns]
    df = df.reindex(columns=kolom_tersedia)
    
    df["datetime"] = pd.to_datetime(df["waktu"], dayfirst=True)
    df = df.set_index("datetime").sort_index()

    # QC Waterlevel (-2.0 sampai 2.0)
    if "waterlevel" in df.columns:
        df["waterlevel"] = pd.to_numeric(df["waterlevel"], errors="coerce")
        mask = (df["waterlevel"] < -2.0) | (df["waterlevel"] > 2.0)
        df.loc[mask, "waterlevel"] = np.nan

    if "windspeed" in df.columns:
        df["windspeed"] = pd.to_numeric(df["windspeed"], errors="coerce")
        df = df.dropna(subset=["windspeed", "waterlevel"], how='all') # Hindari hapus jika salah satu masi ada nilainya (opsional)

    # Resample 10 Menit
    # Note: Kita harus memastikan kolom numerik aman untuk di-resample
    agg_dict = {}
    if "windspeed" in df.columns: agg_dict["windspeed"] = "mean"
    if "waterlevel" in df.columns: agg_dict["waterlevel"] = "mean"
    
    if agg_dict:
        df_new = df.resample("10min").agg(agg_dict)
        df_new.index = df_new.index.strftime("%d/%m/%Y %H:%M:%S")
        df_new = df_new.reset_index().rename(columns={"index":"datetime"})

        # Gabung ke Database Lama
        df_final = pd.concat([df_old, df_new], ignore_index=True) if df_old is not None else df_new
        df_final["datetime"] = pd.to_datetime(df_final["datetime"], dayfirst=True)
        df_final = df_final.drop_duplicates(subset="datetime", keep="last").sort_values("datetime")
        df_final["datetime"] = df_final["datetime"].dt.strftime("%d/%m/%Y %H:%M:%S")

        # Simpan kembali ke Excel
        df_final.to_excel(db_path, index=False)
        print(f"\n✅ Database AWS berhasil diperbarui! Total Data: {len(df_final)} baris.")
        print(f"📄 Lokasi: {db_path}")
    else:
        print("\n⚠️ Data berhasil didapat tapi tidak mengandung kolom 'windspeed' atau 'waterlevel'.")


# =============================================================================
# EKSEKUSI UTAMA
# =============================================================================
if __name__ == "__main__":
    print("🚀 MENJALANKAN PIPELINE DATA COLLECTOR...")
    proses_hf_radar()
    print("\n🏁 SEMUA PROSES AKUISISI DATA SELESAI!")

