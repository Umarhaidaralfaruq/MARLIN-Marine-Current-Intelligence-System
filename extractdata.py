# =============================================================================
# FILE: 01_data_collector.py
# FUNGSI: Mengunduh HF Radar & Web Scraping AWS secara Otomatis (Robust Mode)
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
import cloudscraper
from datetime import datetime, timedelta

# Tambahan library untuk menjaga koneksi agar tidak terputus
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Mengembalikan import config bawaan awal Anda
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
    print("🌊 MULAI PROSES AKUISISI HF RADAR (ROBUST DOWNLOAD)")
    print("="*50)

    url_hf = HFRADAR_URL
    all_files = []

    try:
        print("📥 1. Mengunduh file terbaru dari BMKG...")
        
        # Buat instance scraper dengan Auto-Retry agar koneksi lebih stabil
        scraper = cloudscraper.create_scraper()
        retry_strategy = Retry(
            total=5,
            backoff_factor=3,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        scraper.mount("http://", adapter)
        scraper.mount("https://", adapter)
        
        # Header menggunakan keep-alive karena kita butuh stream yang panjang
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Connection': 'keep-alive'
        }

        # --- FASE CEK UKURAN TARGET ---
        try:
            head_req = scraper.head(url_hf, headers=headers, timeout=30)
            total_size = int(head_req.headers.get('content-length', 0))
            if total_size > 0:
                print(f"📊 Total ukuran target dari server: {total_size / (1024*1024):.2f} MB")
            else:
                print("📊 Total ukuran target: Tidak diketahui (Server tidak mengirim Content-Length)")
        except:
            total_size = 0
            print("📊 Total ukuran target: Gagal didapatkan, melanjutkan unduhan buta...")

        # Persiapan Loop Auto-Resume
        downloaded_size = 0
        mode = 'wb'
        max_retries = 10
        attempt = 0

        # Bersihkan file lama jika ada agar tidak tertimpa sebagian
        if os.path.exists(NC_LATEST_FILE):
            os.remove(NC_LATEST_FILE)

        # --- FASE UNDUHAN ROBUST ---
        while attempt < max_retries:
            # Jika terputus, tambahkan header Range untuk menyambung (resume)
            if downloaded_size > 0:
                headers['Range'] = f"bytes={downloaded_size}-"
                mode = 'ab'  # Append byte (melanjutkan tulisan)
                print(f"🔄 Melanjutkan unduhan dari byte ke-{downloaded_size} (Percobaan {attempt+1}/{max_retries})...")

            try:
                response = scraper.get(url_hf, headers=headers, stream=True, timeout=60)
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type and downloaded_size == 0:
                    print("❌ Error: Server mengirim halaman HTML (Kemungkinan blokir/firewall).")
                    return

                # Menulis file dalam chunk sedang (512 KB)
                with open(NC_LATEST_FILE, mode) as file:
                    for chunk in response.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            file.write(chunk)
                            downloaded_size += len(chunk)
                
                # Cek hasil akhir setelah terputus/selesai dari loop iter_content
                final_size = os.path.getsize(NC_LATEST_FILE)
                
                # Evaluasi 1: Jika total_size diketahui dan file masih kurang
                if total_size > 0 and final_size < total_size:
                    print(f"⚠️ Terputus di {final_size / 1024:.2f} KB. Mencoba menyambung ulang...")
                    attempt += 1
                    time.sleep(3)
                    continue
                
                # Evaluasi 2: Jika total_size tidak diketahui, file NetCDF minimal harusnya > 1MB
                elif total_size == 0 and final_size < 1024 * 1024:
                    print(f"⚠️ Ukuran terlalu kecil ({final_size / 1024:.2f} KB) untuk NetCDF. Terputus, mencoba ulang...")
                    attempt += 1
                    time.sleep(3)
                    continue
                    
                # Jika lolos semua evaluasi, unduhan sukses
                nama_file = NC_LATEST_FILE.name if hasattr(NC_LATEST_FILE, 'name') else os.path.basename(NC_LATEST_FILE)
                print(f"✅ Unduhan selesai dan 100% utuh! File disimpan di: {nama_file} ({final_size / (1024*1024):.2f} MB)")
                all_files.append(str(NC_LATEST_FILE))
                break  # Keluar dari while loop
                
            except Exception as e:
                print(f"❌ Koneksi terputus ({e}). Jeda 5 detik sebelum mencoba lagi...")
                attempt += 1
                time.sleep(5)
                
        if attempt >= max_retries:
            print("❌ Gagal mengunduh file secara utuh setelah batas maksimal percobaan. Script dihentikan.")
            
    except Exception as error:
        print(f"❌ Terjadi kesalahan sistem: {error}")

# =============================================================================
# EKSEKUSI UTAMA
# =============================================================================
if __name__ == "__main__":
    print("🚀 MENJALANKAN PIPELINE DATA COLLECTOR...")
    proses_hf_radar()
    print("\n🏁 SEMUA PROSES AKUISISI DATA SELESAI!")
