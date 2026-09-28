# =============================================================================
# FILE: 02c_confidence_interval.py
# DESKRIPSI: Modul 2 - Menghitung 95% Confidence Interval Otomatis
# =============================================================================

import os
import pandas as pd
import numpy as np
from config import PATHS

def calculate_confidence_interval():
    print("\n" + "="*50)
    print("📊 CONFIDENCE INTERVAL (95%) - PRINCIPAL COMPONENTS")
    print("="*50)

    # Langsung tembak ke file utama tanpa replace string
    train_path = PATHS['pc_train']
    if not os.path.exists(train_path):
        print(f"❌ File tidak ditemukan: {train_path}")
        return
        
    df = pd.read_csv(train_path)
    print(f"🔍 Kolom terdeteksi: {list(df.columns)}\n")
    
    # Ambil semua kolom yang mengandung kata 'PC'
    pc_columns = [col for col in df.columns if 'PC' in col.upper()]
    
    if not pc_columns:
        print("❌ Tidak ada kolom Principal Component (PC) di dalam file ini.")
        return
        
    for target in pc_columns:
        data = df[target].dropna()
        n = len(data)
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        
        # Standard Error of the Mean (SEM)
        sem = std_val / np.sqrt(n)
        
        # Z-Score untuk 95% CI adalah 1.96
        z_score = 1.96
        margin_of_error = z_score * sem
        
        lower_bound = mean_val - margin_of_error
        upper_bound = mean_val + margin_of_error
        
        print(f"🎯 Target: {target}")
        print(f"   -> Mean           : {mean_val:.4f}")
        print(f"   -> Std Deviation  : {std_val:.4f}")
        print(f"   -> 95% CI (Mean)  : [{lower_bound:.4f}  s/d  {upper_bound:.4f}]")
        
        # Simulasi rentang error prediksi (Prediction Interval)
        pred_margin = z_score * std_val
        print(f"   -> Prediction Interval : Prediksi ± {pred_margin:.4f}\n")

if __name__ == "__main__":
    calculate_confidence_interval()