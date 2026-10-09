# =============================================================================
# FILE: 03_train_lstm.py (MODULAR EOF / MEOF)
# DESKRIPSI: Training LSTM Dinamis untuk mode Independent (U/V) atau MEOF
# =============================================================================

import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import PATHS, MODEL_CONFIG, FEATURE_TOGGLES
# 🚀 PERHATIKAN: Kita HANYA mengimpor fungsinya, tidak mengimpor TARGET_COLS
from feature_engineering import build_features

LOOKBACK = MODEL_CONFIG['input_window']    
HORIZON = MODEL_CONFIG['output_window']    
MIN_SEQ_LEN = MODEL_CONFIG['min_seq_len']  
SPATIAL_MODE = MODEL_CONFIG.get('spatial_mode', 'EOF').upper()

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_sequences_by_segment(df, lookback, horizon, min_seq_len, input_cols, target_cols):
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
    print("\n" + "="*60)
    print(f"🤖 TRAINING MODEL KOMPONEN: {comp.upper()} (Mode: {SPATIAL_MODE})")
    print("="*60)
    
    train_path = str(PATHS['pc_train']).replace('.csv', f'_{comp}.csv')
    test_path = str(PATHS['pc_test']).replace('.csv', f'_{comp}.csv')
    
    df_train = pd.read_csv(train_path, parse_dates=['datetime'])
    df_test = pd.read_csv(test_path, parse_dates=['datetime'])

    # 1. FEATURE ENGINEERING
    df_train = build_features(df_train)
    df_test = build_features(df_test)
    
    # 2. 🎯 DETEKSI TARGET DINAMIS (Menggantikan N_PC_FEATURES statis)
    TARGET_COLS = [col for col in df_train.columns if 'PC' in col and not col.startswith('rolling_') and not col.startswith('jarak_')]
    
    ALL_GENERATED_COLS = [col for col in df_train.columns if col not in ['datetime', 'Segment_ID']]
    INPUT_COLS = ALL_GENERATED_COLS.copy()

    # 3. SCALING DINAMIS
    SCALE_COLS = TARGET_COLS + [col for col in INPUT_COLS if col not in TARGET_COLS]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_train[SCALE_COLS] = scaler.fit_transform(df_train[SCALE_COLS])
    df_test[SCALE_COLS] = scaler.transform(df_test[SCALE_COLS])
    
    scaler_path = str(PATHS['scaler_path']).replace('.joblib', f'_{comp}.joblib')
    joblib.dump(scaler, scaler_path)

    # 4. SEQUENCE CUTTING
    X_full_train, Y_full_train = create_sequences_by_segment(df_train, LOOKBACK, HORIZON, MIN_SEQ_LEN, INPUT_COLS, TARGET_COLS)
    X_test, Y_test = create_sequences_by_segment(df_test, LOOKBACK, HORIZON, MIN_SEQ_LEN, INPUT_COLS, TARGET_COLS)
    
    if len(X_full_train) == 0:
        print("❌ Gagal: Tidak ada sequence pada data latih.")
        return

    VAL_RATIO = 0.20
    split_idx = int(len(X_full_train) * (1 - VAL_RATIO))
    
    X_train, Y_train = X_full_train[:split_idx], Y_full_train[:split_idx]
    X_val, Y_val     = X_full_train[split_idx:], Y_full_train[split_idx:]
    
    print(f"   -> Tensor Train : {X_train.shape}")
    print(f"   -> Tensor Val   : {X_val.shape}")
    
    # 5. MENGHITUNG DIMENSI MODEL SECARA OTOMATIS
    N_INPUT_FEATURES = X_train.shape[2]
    N_TARGET_FEATURES = len(TARGET_COLS)

    # 6. BUILD MODEL
    model = Sequential([
        LSTM(MODEL_CONFIG['lstm_units'], activation='tanh', input_shape=(LOOKBACK, N_INPUT_FEATURES), return_sequences=False),
        Dropout(MODEL_CONFIG['dropout_rate']),
        Dense(MODEL_CONFIG['dense_units'], activation='relu'),
        # 🚀 PERHATIKAN: Sekarang layer terakhir diukur secara presisi
        Dense(HORIZON * N_TARGET_FEATURES, activation='tanh'), 
        Reshape((HORIZON, N_TARGET_FEATURES))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']), 
        loss=MODEL_CONFIG['loss_function']
    )

    # 7. TRAINING
    model_path = str(PATHS['model_path']).replace('.h5', f'_{comp}.h5')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=MODEL_CONFIG['early_stop_patience'], restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=MODEL_CONFIG['reduce_lr_factor'], patience=MODEL_CONFIG['reduce_lr_patience'], min_lr=MODEL_CONFIG['reduce_lr_min'], verbose=1)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)

    model.fit(
        X_train, Y_train, 
        validation_data=(X_val, Y_val),
        epochs=MODEL_CONFIG['epochs'],            
        batch_size=MODEL_CONFIG['batch_size'], 
        shuffle=True,          
        callbacks=[early_stopping, reduce_lr, checkpoint], 
        verbose=1
    )

    print(f"\n📊 Evaluasi Awal Komponen {comp.upper()}...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    def inv_scale(y_data):
        n_samples, out_len, n_pc = y_data.shape
        dummy = np.zeros((n_samples * out_len, len(SCALE_COLS)))
        dummy[:, :N_TARGET_FEATURES] = y_data.reshape(-1, n_pc)
        unscaled = scaler.inverse_transform(dummy)
        return unscaled[:, :N_TARGET_FEATURES].reshape(n_samples, out_len, N_TARGET_FEATURES)

    y_pred_unscaled = inv_scale(y_pred_scaled)
    y_true_unscaled = inv_scale(Y_test)

    for i in range(N_TARGET_FEATURES):
        r2_pc = r2_score(y_true_unscaled[:, :, i].reshape(-1), y_pred_unscaled[:, :, i].reshape(-1))
        print(f"   -> Target Mode {i+1} R² : {r2_pc:.4f}")

def run_all_trainings():
    set_seed(42)
    os.makedirs(os.path.dirname(PATHS['model_path']), exist_ok=True)
    
    if SPATIAL_MODE == 'EOF':
        for comp in ['u', 'v']:
            train_component(comp)
    elif SPATIAL_MODE == 'MEOF':
        train_component('meof') 
    else:
        print(f"❌ ERROR: Mode spasial '{SPATIAL_MODE}' tidak valid.")

if __name__ == "__main__":
    run_all_trainings()