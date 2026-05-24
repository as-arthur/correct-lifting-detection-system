# dashboard.py
import os
import pandas as pd
import glob
import csv
import datetime
import joblib
import numpy as np
from pathlib import Path
from collections import deque
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ===== KONFIGURASI =====
MODEL_PATH = "model\\random_forest_model.pkl"
SCALER_PATH = "model\\scaler.pkl"
PREDICTIONS_DIR = "predictions"
DATA_DIR = "dataUser"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# ===== GLOBAL VARIABEL =====
rf_model = None          # Model Random Forest
scaler = None            # Scaler untuk normalisasi fitur

# Buffer global: menyimpan data sensor terbaru dari MPU1, MPU2, dan BME280
# Setiap entri: {'epoch_ms': int, 'data': dict}
buffer = {
    'mpu1': None,
    'mpu2': None,
    'bmp': None
}

# Riwayat prediksi (in-memory) untuk dashboard live, max 500 entri
predictions_history = deque(maxlen=500)

def find_latest_files():
    imu_files = glob.glob(os.path.join(DATA_DIR, "imu_*.csv"))
    bmp_files = glob.glob(os.path.join(DATA_DIR, "bmp_*.csv"))
    if not imu_files or not bmp_files:
        return None, None
    latest_imu = max(imu_files, key=os.path.getmtime)
    latest_bmp = max(bmp_files, key=os.path.getmtime)
    return latest_imu, latest_bmp

# Pemetaan label dari integer ke string
LABEL_MAP = {0: "Salah", 1: "Benar", 2: "Berdiri"}

# ===== FUNGSI MODEL =====
def load_models():
    """Memuat model dan scaler dari file."""
    global rf_model, scaler
    try:
        rf_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model dan scaler berhasil dimuat.")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        rf_model = None
        scaler = None

def prepare_features(mpu1_data, mpu2_data, bmp_data):
    """Menggabungkan data sensor menjadi array fitur 14 dimensi."""
    features = [
        float(mpu1_data.get('ax', 0)), float(mpu1_data.get('ay', 0)), float(mpu1_data.get('az', 0)),
        float(mpu1_data.get('gx', 0)), float(mpu1_data.get('gy', 0)), float(mpu1_data.get('gz', 0)),
        float(mpu2_data.get('ax', 0)), float(mpu2_data.get('ay', 0)), float(mpu2_data.get('az', 0)),
        float(mpu2_data.get('gx', 0)), float(mpu2_data.get('gy', 0)), float(mpu2_data.get('gz', 0)),
        float(bmp_data.get('pressure_hpa', 1013.25)),
        float(bmp_data.get('altitude_m', 0))
    ]
    return np.array(features).reshape(1, -1)

def process_imu_file(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['epoch_ms'] = df['epoch_ms'].astype(int)
    df = df.drop_duplicates(subset=['epoch_ms', 'mpu_id'], keep='first')
    pivoted = df.pivot(index='epoch_ms', columns='mpu_id',
                       values=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    pivoted.columns = [f'{col[0]}{col[1]}' for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    datetime_map = df.groupby('epoch_ms')['datetime'].first()
    pivoted['datetime'] = pivoted['epoch_ms'].map(datetime_map)
    expected = ['epoch_ms', 'datetime', 'ax1','ay1','az1','gx1','gy1','gz1',
                'ax2','ay2','az2','gx2','gy2','gz2']
    for col in expected:
        if col not in pivoted.columns:
            pivoted[col] = pd.NA
    return pivoted[expected]

def process_bmp_file(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['epoch_ms'] = df['epoch_ms'].astype(int)
    return df[['datetime', 'epoch_ms', 'pressure_hpa', 'altitude_m']]

def predict_from_files():
    imu_file, bmp_file = find_latest_files()
    if not imu_file or not bmp_file:
        print("❌ File tidak ditemukan di DATA_DIR")
        return False
    
    imu_df = process_imu_file(imu_file)
    bmp_df = process_bmp_file(bmp_file)
    
    imu_df = imu_df.sort_values('epoch_ms')
    bmp_df = bmp_df.sort_values('epoch_ms')
    
    merged = pd.merge_asof(bmp_df, imu_df, on='epoch_ms', direction='nearest')
    merged = merged.drop_duplicates(subset=['epoch_ms']).sort_values('epoch_ms')
    
    for _, row in merged.iterrows():
        # Siapkan fitur
        features = [
            row.get('ax1',0), row.get('ay1',0), row.get('az1',0),
            row.get('gx1',0), row.get('gy1',0), row.get('gz1',0),
            row.get('ax2',0), row.get('ay2',0), row.get('az2',0),
            row.get('gx2',0), row.get('gy2',0), row.get('gz2',0),
            row.get('pressure_hpa',1013.25), row.get('altitude_m',0)
        ]
        X = np.array(features).reshape(1,-1)
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        pred_int, pred_label = predict(X_scaled)
        if pred_label:
            # Simpan ke history (hanya jika belum ada dengan epoch_ms yang sama)
            if not any(p['epoch_ms'] == row['epoch_ms'] for p in predictions_history):
                predictions_history.append({
                    'datetime': str(row['datetime']),
                    'epoch_ms': row['epoch_ms'],
                    'prediction': pred_label,
                    'pred_int': pred_int
                })
    # Simpan juga ke file CSV prediksi global
    # (opsional, bisa gunakan fungsi save_prediction_to_csv yang sudah ada)
    return True

def predict(features_scaled):
    """Menjalankan prediksi dengan model."""
    if rf_model is None:
        return None, None
    pred_int = rf_model.predict(features_scaled)[0]
    label = LABEL_MAP.get(pred_int, "Unknown")
    return pred_int, label

def save_prediction_to_csv(epoch_ms, datetime_str, pred_label, features_dict):
    """Menyimpan hasil prediksi ke file CSV global di folder predictions."""
    filepath = os.path.join(PREDICTIONS_DIR, "predictions_global.csv")
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['datetime', 'epoch_ms', 'prediction'] + list(features_dict.keys()))
        row = [datetime_str, epoch_ms, pred_label] + list(features_dict.values())
        writer.writerow(row)

def try_predict(epoch_ms, datetime_str, sensor_type, data_row):
    """
    Update buffer dan coba lakukan prediksi jika ketiga data sensor sudah tersedia.
    Mengembalikan label prediksi jika berhasil, else None.
    """
    # Simpan data ke buffer
    buffer[sensor_type] = {'epoch_ms': epoch_ms, 'data': data_row}

    # Cek kelengkapan buffer
    if all(buffer[t] is not None for t in ['mpu1', 'mpu2', 'bmp']):
        mpu1_entry = buffer['mpu1']
        mpu2_entry = buffer['mpu2']
        bmp_entry = buffer['bmp']
        # Periksa selisih waktu maksimal 500 ms 
        times = [mpu1_entry['epoch_ms'], mpu2_entry['epoch_ms'], bmp_entry['epoch_ms']]
        if max(times) - min(times) <= 500:
            # Siapkan fitur dan lakukan scaling
            features_raw = prepare_features(mpu1_entry['data'], mpu2_entry['data'], bmp_entry['data'])
            if scaler:
                features_scaled = scaler.transform(features_raw)
            else:
                features_scaled = features_raw
            pred_int, pred_label = predict(features_scaled)
            if pred_label:
                # Simpan ke CSV
                features_dict = {
                    'ax1': mpu1_entry['data'].get('ax'), 'ay1': mpu1_entry['data'].get('ay'), 'az1': mpu1_entry['data'].get('az'),
                    'gx1': mpu1_entry['data'].get('gx'), 'gy1': mpu1_entry['data'].get('gy'), 'gz1': mpu1_entry['data'].get('gz'),
                    'ax2': mpu2_entry['data'].get('ax'), 'ay2': mpu2_entry['data'].get('ay'), 'az2': mpu2_entry['data'].get('az'),
                    'gx2': mpu2_entry['data'].get('gx'), 'gy2': mpu2_entry['data'].get('gy'), 'gz2': mpu2_entry['data'].get('gz'),
                    'pressure_hpa': bmp_entry['data'].get('pressure_hpa'),
                    'altitude_m': bmp_entry['data'].get('altitude_m')
                }
                save_prediction_to_csv(epoch_ms, datetime_str, pred_label, features_dict)
                # Simpan ke history in-memory
                predictions_history.append({
                    'datetime': datetime_str,
                    'epoch_ms': epoch_ms,
                    'prediction': pred_label,
                    'pred_int': int(pred_int)
                })
                # Kosongkan buffer setelah prediksi berhasil
                for t in ['mpu1', 'mpu2', 'bmp']:
                    buffer[t] = None
                return pred_label
    return None

@app.route("/process_files")
def process_files():
    if predict_from_files():
        return jsonify({"status": "ok", "message": f"{len(predictions_history)} predictions loaded"})
    else:
        return jsonify({"status": "error", "message": "No files found"}), 404

# ===== ENDPOINTS =====
@app.route("/data", methods=["POST"])
def handle_data():
    """
    Endpoint untuk menerima data dari ESP8266.
    Mendukung data MPU (mpu_id=1 atau 2) dan data BME280.
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "reason": "No JSON data"}), 400

    # Ekstrak timestamp
    dt_str = data.get("datetime", datetime.datetime.now().isoformat())
    ep_ms = data.get("epoch_ms")
    if ep_ms is None:
        ep_ms = int(datetime.datetime.now().timestamp() * 1000)
    else:
        ep_ms = int(ep_ms)

    # Data MPU
    if "mpu_id" in data:
        sensor_type = 'mpu1' if data.get('mpu_id') == '1' else 'mpu2'
        sensor_row = {
            'ax': data.get('ax'), 'ay': data.get('ay'), 'az': data.get('az'),
            'gx': data.get('gx'), 'gy': data.get('gy'), 'gz': data.get('gz')
        }
        pred_label = try_predict(ep_ms, dt_str, sensor_type, sensor_row)
        response = {"status": "ok", "sensor": "mpu", "mpu_id": data.get('mpu_id')}
        if pred_label:
            response["prediction"] = pred_label
        return jsonify(response)

    # Data BME280
    elif "temperature_c" in data or "pressure_hpa" in data:
        sensor_type = 'bmp'
        sensor_row = {
            'pressure_hpa': data.get('pressure_hpa'),
            'altitude_m': data.get('altitude_m')
        }
        pred_label = try_predict(ep_ms, dt_str, sensor_type, sensor_row)
        response = {"status": "ok", "sensor": "bmp"}
        if pred_label:
            response["prediction"] = pred_label
        return jsonify(response)

    else:
        return jsonify({"status": "error", "reason": "Unknown sensor data"}), 400

@app.route("/live_data")
def live_data():
    """Menyediakan data prediksi terbaru untuk dashboard live."""
    history = list(predictions_history)
    latest = history[-1] if history else None

    now_ms = int(datetime.datetime.now().timestamp() * 1000)
    last_10s_ms = now_ms - 10000
    recent = [p for p in history if p['epoch_ms'] >= last_10s_ms]
    counts_10s = {"Salah": 0, "Benar": 0, "Berdiri": 0}
    for p in recent:
        counts_10s[p['prediction']] += 1

    total_counts = {"Salah": 0, "Benar": 0, "Berdiri": 0}
    for p in history:
        total_counts[p['prediction']] += 1

    return jsonify({
        "latest_prediction": latest,
        "recent_predictions": history[-20:],
        "counts_last_10s": counts_10s,
        "total_counts": total_counts,
        "timestamp_ms": now_ms
    })

@app.route("/check_data")
def check_data():
    """
    Endpoint untuk mengecek apakah data dari ESP8266 sudah masuk.
    Berguna untuk debugging dan monitoring.
    """
    # Status buffer (apakah masing-masing sensor sudah pernah mengirim data)
    buffer_status = {
        'mpu1': buffer.get('mpu1') is not None,
        'mpu2': buffer.get('mpu2') is not None,
        'bmp': buffer.get('bmp') is not None
    }
    
    # Data terakhir yang tersimpan di buffer (jika ada)
    last_mpu1 = buffer['mpu1']['data'] if buffer.get('mpu1') else None
    last_mpu2 = buffer['mpu2']['data'] if buffer.get('mpu2') else None
    last_bmp = buffer['bmp']['data'] if buffer.get('bmp') else None
    
    # Waktu kedatangan data terakhir (epoch_ms)
    last_mpu1_time = buffer['mpu1']['epoch_ms'] if buffer.get('mpu1') else None
    last_mpu2_time = buffer['mpu2']['epoch_ms'] if buffer.get('mpu2') else None
    last_bmp_time = buffer['bmp']['epoch_ms'] if buffer.get('bmp') else None
    
    # Statistik prediksi
    total_predictions = len(predictions_history)
    latest_prediction = predictions_history[-1] if predictions_history else None
    
    # Hitung waktu sejak data terakhir diterima (dalam detik)
    now_ms = int(datetime.datetime.now().timestamp() * 1000)
    def time_since(ms):
        if ms is None:
            return None
        return round((now_ms - ms) / 1000.0, 1)
    
    return jsonify({
        "status": "ok",
        "server_time": datetime.datetime.now().isoformat(),
        "buffer": buffer_status,
        "last_data": {
            "mpu1": last_mpu1,
            "mpu2": last_mpu2,
            "bmp": last_bmp
        },
        "last_data_age_seconds": {
            "mpu1": time_since(last_mpu1_time),
            "mpu2": time_since(last_mpu2_time),
            "bmp": time_since(last_bmp_time)
        },
        "total_predictions_made": total_predictions,
        "latest_prediction": latest_prediction
    })    

@app.route("/list_files")
def list_files():
    imu_files = glob.glob(os.path.join(DATA_DIR, "imu_*.csv"))
    bmp_files = glob.glob(os.path.join(DATA_DIR, "bmp_*.csv"))
    return jsonify({
        "imu_files": [os.path.basename(f) for f in imu_files],
        "bmp_files": [os.path.basename(f) for f in bmp_files]
    })

@app.route("/")
def live_dashboard():
    if len(predictions_history) == 0:
        predict_from_files()   # coba muat dari file
    return render_template('dashboard.html', file_id="")


# ===== MAIN =====
if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5001, debug=True)