# =============================================================================
# FILE: 02c_confidence_interval.py
# DESKRIPSI: Modul 2 - Menghitung 95% Confidence Interval dari Data MARLIN
# =============================================================================

import os
import pandas as pd
import numpy as np
import scipy.stats as st
from config import PATHS

def calculate_confidence_interval(comp):
    print("\n" + "="*50)
    print(f"📊 CONFIDENCE INTERVAL (95%) - KOMPONEN {comp.upper()}")
    print("="*50)

    train_path = str(PATHS['pc_train']).replace('.npy', f'_{comp}.csv')
    if not os.path.exists(train_path):
        print(f"❌ File tidak ditemukan: {train_path}")
        return
        
    df = pd.read_csv(train_path)
    
    for target in ['PC1', 'PC2']:
        if target not in df.columns:
            continue
            
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
        
        # --- SIMULASI CI PREDIKSI (PREDICTION INTERVAL) ---
        # Jika model lu menebak nilai 0.0, seberapa lebar rentang error alaminya?
        pred_margin = z_score * std_val
        print(f"   -> Rentang Tebakan Model (Pred Interval) : Prediksi ± {pred_margin:.4f}\n")

if __name__ == "__main__":
    for c in ['u', 'v']:
        calculate_confidence_interval(c)