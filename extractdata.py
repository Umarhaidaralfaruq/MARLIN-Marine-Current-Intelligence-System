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
import cloudscraper
from datetime import datetime, timedelta

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

    url_hf = HFRADAR_URL
    all_files = []

    # ---------------------------------------------------------------------------
    # 1. MENGUNDUH FILE TERBARU DARI BMKG
    # ---------------------------------------------------------------------------
    try:
        print("📥 1. Mengunduh file terbaru dari BMKG...")
        
        # Buat instance scraper
        scraper = cloudscraper.create_scraper()
        
        # Headers tambahan (opsional)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # PENTING: Gunakan scraper.get, BUKAN requests.get
        response = scraper.get(url_hf, headers=headers, stream=True, timeout=120)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        
        # Perbaikan Indentasi
        if 'text/html' in content_type:
            print("❌ Error: Server malah mengirim halaman web/HTML, bukan file data. Kemungkinan terkena blokir atau butuh login.")
        else:
            with open(NC_LATEST_FILE, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            
            # Penanganan nama file jika menggunakan pathlib atau string biasa
            nama_file = NC_LATEST_FILE.name if hasattr(NC_LATEST_FILE, 'name') else os.path.basename(NC_LATEST_FILE)
            print(f"✅ File berhasil disimpan di: {nama_file}")
            
            # Tambahkan file yang baru diunduh ke antrean proses
            all_files.append(str(NC_LATEST_FILE))
            
    except Exception as error:
        print(f"❌ Gagal mengunduh file terbaru (Melanjutkan dengan file lokal): {error}")


# =============================================================================
# EKSEKUSI UTAMA
# =============================================================================
if __name__ == "__main__":
    print("🚀 MENJALANKAN PIPELINE DATA COLLECTOR...")
    proses_hf_radar()
    print("\n🏁 SEMUA PROSES AKUISISI DATA SELESAI!")
