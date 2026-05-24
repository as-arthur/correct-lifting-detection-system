import os
import csv
import json
import threading
from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# ---------- KONFIGURASI ----------
DATA_DIR = r"D:\Semester_7\Skripsian\bachelor-last-project\dataUser"
PREDICTIONS_DIR = r"D:\Semester_7\Skripsian\bachelor-last-project\predictions"
MODEL_PATH = r"D:\Semester_7\Skripsian\bachelor-last-project\random_forest_model.pkl"
SCALER_PATH = r"D:\Semester_7\Skripsian\bachelor-last-project\scaler.pkl"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

app = Flask(__name__)

# ---------- GLOBAL VARIABLES ----------
rf_model = None
scaler = None

# Buffer untuk menyimpan data sensor terbaru (tanpa file_id)
# Struktur: {
#   'mpu1': {'data': {...}, 'epoch_ms': int, 'datetime': str},
#   'mpu2': {...},
#   'bmp': {...}
# }
buffer = {}
buffer_lock = threading.Lock()

# Riwayat prediksi untuk keperluan dashboard (max 500)
prediction_history = deque(maxlen=500)

# Label mapping (sesuai model)
LABEL_MAP = {0: "Salah", 1: "Benar", 2: "Berdiri"}

# ---------- LOAD MODEL & SCALER ----------
def load_ml_models():
    global rf_model, scaler
    try:
        rf_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model and scaler loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load ML model: {e}")
        rf_model = None
        scaler = None

load_ml_models()

# ---------- FUNGSI FEATURE ENGINEERING (Mengikuti dataCombine.py) ----------
def prepare_features_from_buffer(mpu1_row, mpu2_row, bmp_row):
    """
    Menggabungkan data MPU1, MPU2, dan BME280 menjadi 14 fitur numerik.
    Urutan fitur persis seperti yang diharapkan model:
    ax1, ay1, az1, gx1, gy1, gz1,
    ax2, ay2, az2, gx2, gy2, gz2,
    pressure_hpa, altitude_m
    """
    features = [
        float(mpu1_row.get('ax', 0)), float(mpu1_row.get('ay', 0)), float(mpu1_row.get('az', 0)),
        float(mpu1_row.get('gx', 0)), float(mpu1_row.get('gy', 0)), float(mpu1_row.get('gz', 0)),
        float(mpu2_row.get('ax', 0)), float(mpu2_row.get('ay', 0)), float(mpu2_row.get('az', 0)),
        float(mpu2_row.get('gx', 0)), float(mpu2_row.get('gy', 0)), float(mpu2_row.get('gz', 0)),
        float(bmp_row.get('pressure_hpa', 1013.25)),
        float(bmp_row.get('altitude_m', 0))
    ]
    return np.array(features).reshape(1, -1)

def predict_from_features(features_scaled):
    """Melakukan prediksi menggunakan model global."""
    if rf_model is None:
        return None, "Model not loaded"
    pred = rf_model.predict(features_scaled)[0]
    label = LABEL_MAP.get(pred, "Unknown")
    return int(pred), label

# ---------- FUNGSI PREDIKSI DENGAN BUFFER ----------
def try_predict(sensor_type, data_row, epoch_ms, datetime_str):
    """
    Update buffer dan lakukan prediksi jika semua sensor sudah tersedia
    dalam rentang waktu 500 ms. Setelah prediksi, kosongkan buffer.
    """
    with buffer_lock:
        buffer[sensor_type] = {
            'data': data_row,
            'epoch_ms': epoch_ms,
            'datetime': datetime_str
        }
        
        required = ['mpu1', 'mpu2', 'bmp']
        if all(k in buffer for k in required):
            mpu1 = buffer['mpu1']
            mpu2 = buffer['mpu2']
            bmp = buffer['bmp']
            times = [mpu1['epoch_ms'], mpu2['epoch_ms'], bmp['epoch_ms']]
            if max(times) - min(times) <= 500:  # toleransi 500 ms
                # Siapkan fitur
                X_raw = prepare_features_from_buffer(
                    mpu1['data'], mpu2['data'], bmp['data']
                )
                # Scaling
                if scaler is not None:
                    X_scaled = scaler.transform(X_raw)
                else:
                    X_scaled = X_raw
                # Prediksi
                pred_int, pred_label = predict_from_features(X_scaled)
                if pred_label:
                    # Simpan ke CSV prediksi (format gabungan + label)
                    save_prediction_record(
                        epoch_ms=mpu1['epoch_ms'],  # bisa pakai salah satu
                        datetime_str=mpu1['datetime'],
                        pred_label=pred_label,
                        mpu1=mpu1['data'],
                        mpu2=mpu2['data'],
                        bmp=bmp['data']
                    )
                    # Simpan ke memory history
                    prediction_history.append({
                        'datetime': mpu1['datetime'],
                        'epoch_ms': mpu1['epoch_ms'],
                        'prediction': pred_label,
                        'pred_int': pred_int
                    })
                    # Hapus buffer agar tidak memprediksi ulang data yang sama
                    for k in required:
                        buffer.pop(k, None)
                    return pred_label
    return None

def save_prediction_record(epoch_ms, datetime_str, pred_label, mpu1, mpu2, bmp):
    """Simpan hasil prediksi ke predictions/predictions.csv dengan format lengkap."""
    filepath = os.path.join(PREDICTIONS_DIR, "predictions.csv")
    file_exists = os.path.exists(filepath)
    
    # Kolom sama seperti hasil merge di dataCombine.py + prediksi
    fieldnames = [
        'datetime', 'epoch_ms', 'prediction',
        'ax1', 'ay1', 'az1', 'gx1', 'gy1', 'gz1',
        'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2',
        'pressure_hpa', 'altitude_m'
    ]
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {
            'datetime': datetime_str,
            'epoch_ms': epoch_ms,
            'prediction': pred_label,
            'ax1': mpu1.get('ax'), 'ay1': mpu1.get('ay'), 'az1': mpu1.get('az'),
            'gx1': mpu1.get('gx'), 'gy1': mpu1.get('gy'), 'gz1': mpu1.get('gz'),
            'ax2': mpu2.get('ax'), 'ay2': mpu2.get('ay'), 'az2': mpu2.get('az'),
            'gx2': mpu2.get('gx'), 'gy2': mpu2.get('gy'), 'gz2': mpu2.get('gz'),
            'pressure_hpa': bmp.get('pressure_hpa'),
            'altitude_m': bmp.get('altitude_m')
        }
        writer.writerow(row)

# ---------- HANDLER DATA DARI ESP8266 ----------
@app.route("/data", methods=["POST"])
def upload_data():
    try:
        raw = request.get_data(as_text=True)
        print("---- NEW POST /data ----")
        print("Raw body:", raw[:200])
    except Exception as e:
        print("Logging error:", e)

    data = request.get_json(force=True, silent=True)
    if not data:
        # fallback form data
        data = {
            "datetime": request.form.get("datetime"),
            "epoch_ms": request.form.get("epoch_ms"),
            "ts_millis": request.form.get("ts_millis"),
            "mpu_id": request.form.get("mpu_id"),
            "ax": request.form.get("ax"),
            "ay": request.form.get("ay"),
            "az": request.form.get("az"),
            "gx": request.form.get("gx"),
            "gy": request.form.get("gy"),
            "gz": request.form.get("gz"),
            "temperature_c": request.form.get("temperature_c"),
            "pressure_hpa": request.form.get("pressure_hpa"),
            "altitude_m": request.form.get("altitude_m"),
            "sensor": request.form.get("sensor"),
        }

    # Deteksi jenis sensor
    if data and ("sensor" in data or "temperature_c" in data):
        return handle_bmp_data(data)
    else:
        return handle_mpu_data(data)

def handle_mpu_data(data):
    # Simpan ke file CSV di dataUser (global, tanpa file_id)
    now = datetime.now()
    filename = f"imu_{now.strftime('%Y-%m-%d')}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    file_exists = os.path.exists(filepath)
    
    try:
        with open(filepath, "a", newline='') as f:
            if not file_exists:
                f.write("datetime,epoch_ms,ts_millis,mpu_id,ax,ay,az,gx,gy,gz\n")
            
            dt_str = data.get("datetime", now.isoformat())
            epoch_ms = data.get("epoch_ms")
            if epoch_ms is None or epoch_ms == "":
                epoch_ms = int(now.timestamp() * 1000)
            else:
                epoch_ms = int(epoch_ms)
            ts_millis = data.get("ts_millis", "")
            mpu_id = data.get("mpu_id", "")
            ax = data.get("ax", "")
            ay = data.get("ay", "")
            az = data.get("az", "")
            gx = data.get("gx", "")
            gy = data.get("gy", "")
            gz = data.get("gz", "")
            
            f.write(f"{dt_str},{epoch_ms},{ts_millis},{mpu_id},{ax},{ay},{az},{gx},{gy},{gz}\n")
    except Exception as e:
        print("Error writing MPU file:", e)
        return jsonify({"status":"error","reason":str(e)}), 500
    
    # Update buffer dan coba prediksi
    sensor_type = 'mpu1' if str(mpu_id) == '1' else 'mpu2'
    sensor_row = {
        'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz
    }
    pred_label = try_predict(sensor_type, sensor_row, epoch_ms, dt_str)
    
    response = {"status": "ok", "saved_to": filename}
    if pred_label:
        response["prediction"] = pred_label
    return jsonify(response)

def handle_bmp_data(data):
    now = datetime.now()
    filename = f"bmp_{now.strftime('%Y-%m-%d')}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    file_exists = os.path.exists(filepath)
    
    try:
        with open(filepath, "a", newline='') as f:
            if not file_exists:
                f.write("datetime,epoch_ms,ts_millis,sensor,temperature_c,pressure_hpa,altitude_m\n")
            
            dt_str = data.get("datetime", now.isoformat())
            epoch_ms = data.get("epoch_ms")
            if epoch_ms is None or epoch_ms == "":
                epoch_ms = int(now.timestamp() * 1000)
            else:
                epoch_ms = int(epoch_ms)
            ts_millis = data.get("ts_millis", "")
            sensor = data.get("sensor", "BME280")
            temp = data.get("temperature_c", "")
            press = data.get("pressure_hpa", "")
            alt = data.get("altitude_m", "")
            
            f.write(f"{dt_str},{epoch_ms},{ts_millis},{sensor},{temp},{press},{alt}\n")
    except Exception as e:
        print("Error writing BMP file:", e)
        return jsonify({"status":"error","reason":str(e)}), 500
    
    # Update buffer dan coba prediksi
    sensor_row = {
        'pressure_hpa': press,
        'altitude_m': alt
    }
    pred_label = try_predict('bmp', sensor_row, epoch_ms, dt_str)
    
    response = {"status": "ok", "saved_to": filename}
    if pred_label:
        response["prediction"] = pred_label
    return jsonify(response)

# ---------- ENDPOINT UNTUK DASHBOARD LIVE ----------
@app.route("/live_data")
def live_data():
    """Mengembalikan data prediksi terbaru, statistik 10 detik, total."""
    if not prediction_history:
        return jsonify({
            "latest_prediction": None,
            "recent_predictions": [],
            "counts_last_10s": {"Salah": 0, "Benar": 0, "Berdiri": 0},
            "total_counts": {"Salah": 0, "Benar": 0, "Berdiri": 0},
            "timestamp_ms": int(datetime.now().timestamp() * 1000)
        })
    
    now_ms = int(datetime.now().timestamp() * 1000)
    last_10s_ms = now_ms - 10000
    
    history_list = list(prediction_history)
    latest = history_list[-1] if history_list else None
    
    counts_10s = {"Salah": 0, "Benar": 0, "Berdiri": 0}
    total_counts = {"Salah": 0, "Benar": 0, "Berdiri": 0}
    for p in history_list:
        total_counts[p['prediction']] += 1
        if p['epoch_ms'] >= last_10s_ms:
            counts_10s[p['prediction']] += 1
    
    return jsonify({
        "latest_prediction": latest,
        "recent_predictions": history_list[-20:],
        "counts_last_10s": counts_10s,
        "total_counts": total_counts,
        "timestamp_ms": now_ms
    })

@app.route("/dashboard")
def dashboard():
    """Halaman HTML untuk live dashboard."""
    return render_template("live_dashboard.html")

# ---------- RUN SERVER ----------
if __name__ == "__main__":
    # Jalankan di port 5001 agar tidak bentrok dengan app.py asli (port 5000)
    app.run(host="0.0.0.0", port=5001, debug=True)