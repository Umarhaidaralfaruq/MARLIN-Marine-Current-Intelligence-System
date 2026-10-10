# =============================================================================
# FILE: 02b_eda_correlation.py
# DESKRIPSI: Modul 2 - Multivariate Analysis (Korelasi Fitur vs Target) per Komponen
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import PATHS
from feature_engineering import build_features, TARGET_COLS

def analyze_correlation_for_component(comp):
    print(f"\n" + "="*50)
    print(f"🔗 ANALISIS KORELASI KOMPONEN: {comp.upper()}")
    print("="*50)

    # 1. BUKA DATA TRAINING
    train_path = str(PATHS['pc_train']).replace('.csv', f'_{comp}.csv')
    
    if not os.path.exists(train_path):
        print(f"❌ File tidak ditemukan: {train_path}")
        return
        
    df_train = pd.read_csv(train_path, parse_dates=['datetime'])

    print("⚙️ Merakit fitur menggunakan build_features()...")
    df_feats = build_features(df_train)
    
    cols_to_drop = ['datetime', 'Segment_ID']
    df_corr_input = df_feats.drop(columns=[c for c in cols_to_drop if c in df_feats.columns])

    print("🧮 Menghitung Matriks Korelasi (Pearson)...")
    corr_matrix = df_corr_input.corr(method='pearson')

    # KOREKSI: Ubah target menjadi MEOF_PC1 dan MEOF_PC2
    for target in ['MEOF_PC1', 'MEOF_PC2']:
        if target in corr_matrix.columns:
            print(f"\n🏆 TOP FITUR PREDIKTOR UNTUK {target} ({comp.upper()}):")
            target_corr = corr_matrix[target].drop(target)
            sorted_corr = target_corr.abs().sort_values(ascending=False)
            
            for feat, abs_val in sorted_corr.head(10).items():
                real_val = target_corr[feat]
                print(f"   -> {feat:<30} : {real_val:+.4f}")
        else:
            print(f"⚠️ Target {target} tidak ditemukan.")

    # Visualisasi Heatmap (Sama seperti sebelumnya)
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .7}, annot=False) 
    plt.title(f"Matriks Korelasi Fitur MARLIN - {comp.upper()}", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"heatmap_korelasi_{comp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Heatmap korelasi {comp.upper()} tersimpan di: {save_path}")

def run_all_correlations():
    # KOREKSI: Hanya run untuk 'meof'
    for comp in ['meof']:
        analyze_correlation_for_component(comp)

if __name__ == "__main__":
    run_all_correlations()