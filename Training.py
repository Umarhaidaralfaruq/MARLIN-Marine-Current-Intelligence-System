# =============================================================================
# FILE: 03_train_lstm.py (DENGAN FEATURE TOGGLES)
# DESKRIPSI: Training LSTM dengan Dynamic Feature Panel & Segment Bounding
# =============================================================================

import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import PATHS, MODEL_CONFIG, FEATURE_TOGGLES
from feature_engineering import build_features, TARGET_COLS

# Sesuaikan dengan nama kolom angin di CSV lu!
WIND_COLS = ['windspeed'] 

# Parameter Arsitektur
LOOKBACK = MODEL_CONFIG['input_window']    
HORIZON = MODEL_CONFIG['output_window']    
N_PC_FEATURES = MODEL_CONFIG['n_modes']    
MIN_SEQ_LEN = MODEL_CONFIG['min_seq_len']  
# -----------------------------------------------------------------------------

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_sequences_by_segment(df, lookback, horizon, min_seq_len, input_cols, target_cols):
    """
    Membuat Sliding Window secara dinamis berdasarkan kolom input yang aktif.
    """
    X, Y = [], []
    
    for seg_id, group in df.groupby('Segment_ID'):
        if len(group) < min_seq_len:
            continue
            
        features_input = group[input_cols].values
        target_pc = group[target_cols].values
        
        for i in range(len(group) - lookback - horizon + 1):
            X.append(features_input[i : i + lookback])
            Y.append(target_pc[i + lookback : i + lookback + horizon])
            
    return np.array(X), np.array(Y)

def train_component(comp):
    print("\n" + "="*50)
    print(f"🤖 TRAINING MODEL KOMPONEN: {comp.upper()}")
    print("="*50)
    
    # 1. LOAD DATA CSV
    train_path = str(PATHS['pc_train']).replace('.npy', f'_{comp}.csv')
    test_path = str(PATHS['pc_test']).replace('.npy', f'_{comp}.csv')
    
    df_train = pd.read_csv(train_path, parse_dates=['datetime'])
    df_test = pd.read_csv(test_path, parse_dates=['datetime'])

    # 2. FEATURE ENGINEERING (DINAMIS BERDASARKAN TOGGLES)
    df_train = build_features(df_train)
    df_test = build_features(df_test)
    
    ALL_GENERATED_COLS = [col for col in df_train.columns if col not in ['datetime', 'Segment_ID']]
    INPUT_COLS = [col for col in ALL_GENERATED_COLS if col not in TARGET_COLS]

    # 3. SCALING DINAMIS (MinMax -1 to 1)
    # Trik: Target PC diletakkan paling awal di list SCALE_COLS agar gampang di-inverse
    SCALE_COLS = TARGET_COLS + [col for col in INPUT_COLS if col not in TARGET_COLS]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_train[SCALE_COLS] = scaler.fit_transform(df_train[SCALE_COLS])
    df_test[SCALE_COLS] = scaler.transform(df_test[SCALE_COLS])
    
    scaler_path = str(PATHS['scaler_path']).replace('.joblib', f'_{comp}.joblib')
    joblib.dump(scaler, scaler_path)

    # 4. MEMBUAT SEQUENCE
    print(f"✂️ Memotong Sequence (Input: {len(INPUT_COLS)} Fitur)...")
    X_train, Y_train = create_sequences_by_segment(df_train, LOOKBACK, HORIZON, MIN_SEQ_LEN, INPUT_COLS, TARGET_COLS)
    X_test, Y_test = create_sequences_by_segment(df_test, LOOKBACK, HORIZON, MIN_SEQ_LEN, INPUT_COLS, TARGET_COLS)
    
    print(f"   -> Tensor Train : {X_train.shape}")
    print(f"   -> Tensor Test  : {X_test.shape}")
    
    if len(X_train) == 0:
        print("❌ Gagal: Tidak ada sequence. Cek filter min_seq_len!")
        return

    N_INPUT_FEATURES = X_train.shape[2]

    # =========================================================================
    # 5. BUILD MODEL (ADAPTASI PARAMETER 91%)
    # =========================================================================
    model = Sequential([
        # Otak utama dibesarkan sedikit agar leluasa menampung 41 fitur x 48 jam
        LSTM(MODEL_CONFIG['lstm_units'], activation='tanh', input_shape=(LOOKBACK, N_INPUT_FEATURES), return_sequences=False),
        
        Dropout(MODEL_CONFIG['dropout_rate']),
        
        # 💡 INI KUNCI ARSITEKTUR: Layer Perantara (Decoder)
        # Jangan biarkan LSTM langsung muntah ke Output. Kasih ruang buat dia mikir.
        Dense(MODEL_CONFIG['dense_units'], activation='relu'),
        
        # Layer Output (120 titik)
        Dense(HORIZON * N_PC_FEATURES), 
        
        # Merapikan output menjadi bentuk (24, 5)
        Reshape((HORIZON, N_PC_FEATURES))
    ])

    # Menggunakan MAE sebagai Loss Function sesuai request
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']), 
        loss=MODEL_CONFIG['loss_function']
    )

    # =========================================================================
    # 6. TRAINING
    # =========================================================================
    print(f"🚀 Memulai Training Komponen {comp.upper()}...")
    model_path = str(PATHS['model_path']).replace('.h5', f'_{comp}.h5')
    
    # Callbacks yang ditarik dari config
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=MODEL_CONFIG['early_stop_patience'], 
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=MODEL_CONFIG['reduce_lr_factor'], 
        patience=MODEL_CONFIG['reduce_lr_patience'], 
        min_lr=MODEL_CONFIG['reduce_lr_min'], 
        verbose=1
    )
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)

    print("⏳ Sedang melatih (Training) model... (Ini mungkin memakan waktu)")
    model.fit(
        X_train, Y_train, 
        validation_data=(X_test, Y_test),
        epochs=MODEL_CONFIG['epochs'],            
        batch_size=MODEL_CONFIG['batch_size'], 
        shuffle=True,          
        callbacks=[early_stopping, reduce_lr, checkpoint], 
        verbose=1
    )

    # 7. EVALUATION DENGAN INVERSE SCALING DINAMIS
    print(f"\n📊 Mengevaluasi Model Komponen {comp.upper()}...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Fungsi inverse yang aman berapapun fitur yang aktif
    def inverse_scale_dynamic(y_data, scaler_obj, total_scale_cols, n_target_cols):
        n_samples, out_len, n_pc = y_data.shape
        # Bikin matriks 0 sebesar total kolom yang di-scale
        dummy = np.zeros((n_samples * out_len, total_scale_cols))
        # Masukkan tebakan AI ke kolom-kolom terdepan (karena Target kita taruh depan tadi)
        dummy[:, :n_target_cols] = y_data.reshape(-1, n_pc)
        unscaled = scaler_obj.inverse_transform(dummy)
        # Ambil kembali hasil inverse-nya
        return unscaled[:, :n_target_cols].reshape(n_samples, out_len, n_target_cols)

    TOTAL_SCALED_COLS = len(SCALE_COLS)
    y_pred_unscaled = inverse_scale_dynamic(y_pred_scaled, scaler, TOTAL_SCALED_COLS, N_PC_FEATURES)
    y_true_unscaled = inverse_scale_dynamic(Y_test, scaler, TOTAL_SCALED_COLS, N_PC_FEATURES)

    print("\n" + "-"*40)
    print(f"🏆 DETAIL R² PER MODE ({comp.upper()})")
    print("-"*40)
    for i in range(N_PC_FEATURES):
        r2_pc = r2_score(y_true_unscaled[:, :, i].reshape(-1), y_pred_unscaled[:, :, i].reshape(-1))
        print(f"   -> Mode {i+1} R² : {r2_pc:.4f}")

def run_all_trainings():
    set_seed(42)
    os.makedirs(os.path.dirname(PATHS['model_path']), exist_ok=True)
    
    for comp in ['u', 'v']:
        train_component(comp)

if __name__ == "__main__":
    run_all_trainings()