# app.py
import os
import re
import csv
import math
import datetime
import joblib
import requests
import threading
import numpy as np
import pandas as pd
import warnings
import json
from collections import deque, defaultdict
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from collections import deque, defaultdict
from flask import Flask, request, send_file, abort, jsonify, render_template, redirect, url_for
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model/random_forest_model.pkl"
SCALER_PATH = "model/scaler.pkl"
CONFIG_FILE = "model/model_config.json"

with open('model/feature_names.json', 'r') as f:
    EXPECTED_FEATURES = json.load(f)


DEFAULT_CONFIG = {
    "confidence_threshold": 0.2,      # minimal probabilitas untuk dianggap valid
    "buzzer_duration": 1.5,           # detik buzzer menyala
    "prediction_interval_ms": 300,    # interval prediksi dari frontend (ms)
    "buzzer_enabled": True,           # nyalakan buzzer saat prediksi "Salah"
    "sensor_stale_timeout_ms": 5000   # batas waktu data sensor kadaluarsa (ms)
}

# Buffer global untuk menampung 20 data terakhir per file_id/sesi
data_buffers = defaultdict(lambda: deque(maxlen=20))
FS = 50.0 # Sesuaikan dengan frekuensi sampling ESP32 Anda (cek di Notebook)

# === FUNGSI EKSTRAKSI FITUR (SALIN DARI NOTEBOOK) ===
def time_features_1d(x, fs):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2: return {k: 0.0 for k in ['zero_crossing','sma','rms','autocorr_lag1','peak_to_peak','kurtosis','skewness']}
    std_val = np.std(x)
    is_constant = std_val < 1e-10
    zero_cross = np.sum(np.diff(np.sign(x)) != 0) / n
    sma = np.sum(np.abs(x)) / n
    rms = np.sqrt(np.mean(x**2))
    if is_constant or n < 3 or np.std(x[:-1]) < 1e-10 or np.std(x[1:]) < 1e-10: autocorr = 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                cc = np.corrcoef(x[:-1], x[1:])
                val = cc[0, 1]
                autocorr = 0.0 if (np.isnan(val) or np.isinf(val)) else float(val)
            except Exception: autocorr = 0.0
    ptp = float(np.max(x) - np.min(x))
    if is_constant: kur, skew_val = 0.0, 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            kur, skew_val = float(kurtosis(x, fisher=True, bias=False)), float(skew(x, bias=False))
    return {'zero_crossing': zero_cross, 'sma': sma, 'rms': rms, 'autocorr_lag1': autocorr, 'peak_to_peak': ptp, 'kurtosis': kur, 'skewness': skew_val}

def freq_features_1d(x, fs):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4: return {'dominant_freq': 0.0, 'dominant_power': 0.0, 'spectral_entropy': 0.0, 'band_0.00_0.83hz': 0.0, 'band_0.83_1.67hz': 0.0, 'band_1.67_2.50hz': 0.0}
    nperseg = min(256, n)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    Pxx = np.maximum(Pxx, 1e-12)
    idx_max = np.argmax(Pxx)
    dom_freq, dom_power = f[idx_max], Pxx[idx_max]
    p_norm = Pxx / np.sum(Pxx)
    spectral_entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12))
    bands = {'band_0.00_0.83hz': (0.0, 0.83), 'band_0.83_1.67hz': (0.83, 1.67), 'band_1.67_2.50hz': (1.67, 2.50)}
    band_energies = {label: float(np.sum(Pxx[(f >= fmin) & (f < fmax)])) for label, (fmin, fmax) in bands.items()}
    return {'dominant_freq': dom_freq, 'dominant_power': dom_power, 'spectral_entropy': spectral_entropy, **band_energies}

def extract_window_features(df_window):
    feat_dict = {}
    sensor_cols = ['ax1','ay1','az1','gx1','gy1','gz1','ax2','ay2','az2','gx2','gy2','gz2','pressure_hpa','altitude_m']
    for col in sensor_cols:
        if col in df_window.columns:
            values = df_window[col].values
            for k, v in time_features_1d(values, FS).items(): feat_dict[f'{col}_{k}'] = v
            for k, v in freq_features_1d(values, FS).items(): feat_dict[f'{col}_{k}'] = v
            
    # Fitur Interaksi & Magnitudo (Sama persis dengan Notebook)
    if all(c in df_window.columns for c in ['ax1','ax2']):
        diff = df_window['ax1'].values - df_window['ax2'].values
        feat_dict['diff_ax'], feat_dict['diff_ax_std'] = np.mean(diff), np.std(diff)
    # ... (Lakukan hal yang sama untuk diff_ay, diff_az, diff_gx, diff_gy, diff_gz)
    
    if all(c in df_window.columns for c in ['ax1','ay1','az1']):
        mag1 = np.sqrt(df_window['ax1']**2 + df_window['ay1']**2 + df_window['az1']**2)
        feat_dict['acc_mag1_mean'], feat_dict['acc_mag1_std'] = np.mean(mag1), np.std(mag1)
    # ... (Lakukan hal yang sama untuk acc_mag2, gyro_mag1, gyro_mag2)

    # Hapus fitur yang dibuang saat training
    feat_dict.pop('pressure_hpa_zero_crossing', None)
    feat_dict.pop('altitude_m_zero_crossing', None)
    
    return feat_dict

def load_config():
    """Muat konfigurasi dari file JSON, jika tidak ada buat default"""
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # pastikan semua key default ada
            for key, val in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = val
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return DEFAULT_CONFIG.copy()

def save_config(config):
    """Simpan konfigurasi ke file JSON"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

# Cache model dan scaler
_rf_model = None
_scaler = None

def load_model():
    global _rf_model
    if _rf_model is None:
        _rf_model = joblib.load(MODEL_PATH)
    return _rf_model

def load_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _scaler

SAVE_DIR = r"dataUser"
os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
esp_ip_map = {}

def send_buzzer_command(file_id, turn_on=True):
    """Kirim perintah ke ESP untuk menyalakan/mematikan buzzer"""
    ip = esp_ip_map.get(file_id)
    if not ip:
        print(f"[WARNING] Tidak tahu IP ESP untuk file_id {file_id}")
        return False
    state = "1" if turn_on else "0"
    url = f"http://{ip}/buzzer?state={state}"
    try:
        resp = requests.get(url, timeout=1)
        if resp.status_code == 200:
            print(f"[OK] Buzzer {file_id} -> {'ON' if turn_on else 'OFF'}")
            return True
        else:
            print(f"[ERROR] Gagal, status {resp.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

# sanitize file id: hanya huruf, angka, underscore, dash; sisanya diganti underscore
def sanitize_file_id(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)

def get_today_filepath(prefix="imu", file_id=None, date_obj=None):
    if date_obj is None:
        date_obj = datetime.datetime.now()
    today = date_obj.strftime("%Y-%m-%d")
    if file_id:
        filename = f"{prefix}_{today}_{file_id}.csv"
    else:
        filename = f"{prefix}_{today}.csv"
    return os.path.join(SAVE_DIR, filename)

# --- Fungsi: Membaca data dari file CSV ---
def read_csv_data(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"[DEBUG] File not found: {file_path}")
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        print(f"[DEBUG] File is empty: {file_path}")
        return data

    lines = content.splitlines()
    
    if lines and 'datetime' in lines[0].lower():
        print(f"[DEBUG] Header found and skipped: {lines[0]}")
        lines = lines[1:]

    raw_data_rows = []
    for line in lines:
        import re
        parts = re.split(r'(bagus|jelek),5', line)
        for i in range(0, len(parts) - 1, 2):
            data_part = parts[i]
            suffix_part = parts[i + 1] if i + 1 < len(parts) and parts[i + 1] in ['bagus', 'jelek'] else ''
            row_text = data_part + suffix_part + ",5"
            row_text = row_text.strip()
            if row_text:
                raw_data_rows.append(row_text)

    for row_text in raw_data_rows:
        row = row_text.split(',')
        if len(row) == 12 and row[10].lower() in ['bagus', 'jelek']:
            processed_row = [val if val.strip() != '' else None for val in row]
            data.append(processed_row)
        else:
            print(f"[DEBUG] Invalid row (not 12 columns or prediction not good/bad) after parsing: '{row_text}' -> {row} (length: {len(row)})")

    print(f"[DEBUG] Number of data collected: {len(data)}")
    return data

# --- Fungsi: Menghitung statistik untuk dashboard ---
def get_dashboard_data(file_path):
    raw_data = read_csv_data(file_path)
    if not raw_data:
        print(f"[DEBUG] No data found in {file_path} after parsing.")
        return None
    
    all_predictions = []
    for row in raw_data:
        if len(row) >= 12:
            pred_val = row[10]
            if pred_val and pred_val.lower() in ['bagus', 'jelek']:
                row_dict = {
                    'datetime': row[0],
                    'epoch_ms': row[1],
                    'ts_millis': row[2],
                    'mpu_id': row[3],
                    'ax': row[4],
                    'ay': row[5],
                    'az': row[6],
                    'gx': row[7],
                    'gy': row[8],
                    'gz': row[9],
                    'prediction': row[10],
                    'subject': row[11]
                }
                all_predictions.append(row_dict)
        else:
            print(f"[DEBUG] Row has less than 12 columns, skipping: {row}")

    if not all_predictions:
        print(f"[DEBUG] Tidak ada prediksi 'bagus'/'jelek' ditemukan di {file_path} setelah filtering.")
        return None

    def time_to_seconds(time_str):
        parts = time_str.split(':')
        if len(parts) == 2:
            mins, secs = parts
            hours = 0
        elif len(parts) == 3:
            hours, mins, secs = parts
        else:
            print(f"[DEBUG] Time format unrecognized: {time_str}")
            return 0
        
        try:
            hours = int(hours)
            mins = int(mins)
            secs_parts = secs.split('.')
            whole_secs = int(secs_parts[0])
            millisecs = int(secs_parts[1]) if len(secs_parts) > 1 else 0
        except ValueError:
            print(f"[DEBUG] Error parsing time: {time_str}")
            return 0
        
        total_seconds = hours * 3600 + mins * 60 + whole_secs + millisecs / 10.0
        return total_seconds

    def group_by_10s(all_predictions):
        predictions_with_seconds = []
        for row in all_predictions:
            time_str = row['datetime']
            seconds = time_to_seconds(time_str)
            predictions_with_seconds.append((seconds, row))

        predictions_with_seconds.sort(key=lambda x: x[0])

        grouped_data = {}
        for seconds, row in predictions_with_seconds:
            interval_start = int(seconds // 10) * 10
            interval_end = interval_start + 10
            interval_key = f"{interval_start}-{interval_end}"

            if interval_key not in grouped_data:
                grouped_data[interval_key] = {'bagus': 0, 'jelek': 0}

            status = row['prediction']
            if status == 'bagus':
                grouped_data[interval_key]['bagus'] += 1
            elif status == 'jelek':
                grouped_data[interval_key]['jelek'] += 1

        return grouped_data

    total_predictions = len(all_predictions)
    good_predictions = [row for row in all_predictions if row['prediction'] == 'bagus']
    bad_predictions = [row for row in all_predictions if row['prediction'] == 'jelek']

    durations_good = []
    durations_bad = []
    current_status = None
    start_time = None
    for row in all_predictions:
        time_str = row['datetime']
        status = row['prediction']
        current_time_sec = time_to_seconds(time_str)
        if start_time is None:
            start_time = current_time_sec
            current_status = status
        elif status != current_status:
            duration = current_time_sec - start_time
            durations_good.append(abs(duration)) if current_status == 'bagus' else durations_bad.append(abs(duration))
            start_time = current_time_sec
            current_status = status
    
    if start_time is not None and current_status is not None:
        final_time = time_to_seconds(all_predictions[-1]['datetime'])
        duration = final_time - start_time
        if current_status == 'bagus':
            durations_good.append(abs(duration))
        elif current_status == 'jelek':
            durations_bad.append(abs(duration))

    total_duration_good = sum(durations_good)
    total_duration_bad = sum(durations_bad)
    avg_duration_good = sum(durations_good) / len(durations_good) if durations_good else 0
    avg_duration_bad = sum(durations_bad) / len(durations_bad) if durations_bad else 0

    intervals = []
    for i in range(1, len(all_predictions)):
        time_prev = time_to_seconds(all_predictions[i-1]['datetime'])
        time_curr = time_to_seconds(all_predictions[i]['datetime'])
        interval = time_curr - time_prev
        intervals.append(abs(interval))

    avg_interval = sum(intervals) / len(intervals) if intervals else 0

    anomalies = []
    for row in all_predictions:
        try:
            if row['prediction'] == 'jelek':
                gy_val_str = row['gy']
                az_val_str = row['az']
                if gy_val_str is None or az_val_str is None or gy_val_str == '' or az_val_str == '':
                    continue
                gy_val = float(gy_val_str)
                az_val = float(az_val_str)
                is_gy_anomaly = abs(gy_val) > 10
                is_az_anomaly = abs(az_val) > 12 or abs(az_val) < 0.1
                if is_gy_anomaly or is_az_anomaly:
                    anomalies.append({
                        'datetime': row['datetime'],
                        'mpu_id': row['mpu_id'],
                        'gy': abs(gy_val),
                        'az': az_val,
                        'note': 'Extreme Anomaly' if (is_gy_anomaly and is_az_anomaly) else ('Unusual Vibration' if is_gy_anomaly else 'Sudden Change')
                    })
        except (ValueError, KeyError):
            continue

    top_anomalies = sorted(anomalies, key=lambda x: x['gy'], reverse=True)[:3]

    overall_data_labels = ['Good', 'Bad']
    overall_data_values = [len(good_predictions), len(bad_predictions)]
    overall_data_colors = [
        'rgba(75, 192, 192, 0.2)',
        'rgba(255, 99, 132, 0.2)'
    ]
    overall_data_borderColors = [
        'rgb(75, 192, 192)',
        'rgb(255, 99, 132)'
    ]

    grouped_by_10s_data = group_by_10s(all_predictions)

    posture_time_labels = list(grouped_by_10s_data.keys())
    good_counts_per_interval = [grouped_by_10s_data[interval]['bagus'] for interval in posture_time_labels]
    bad_counts_per_interval = [grouped_by_10s_data[interval]['jelek'] for interval in posture_time_labels]

    posture_over_time_barchart_data = {
        'labels': posture_time_labels,
        'datasets': [
            {
                'label': 'Good',
                'data': good_counts_per_interval,
                'backgroundColor': 'rgba(75, 192, 192, 0.6)',
                'borderColor': 'rgb(75, 192, 192)',
                'borderWidth': 1
            },
            {
                'label': 'Bad',
                'data': bad_counts_per_interval,
                'backgroundColor': 'rgba(255, 99, 132, 0.6)',
                'borderColor': 'rgb(255, 99, 132)',
                'borderWidth': 1
            }
        ]
    }

    filename = os.path.basename(file_path)
    name_part = filename.split('.')[0]
    name = name_part.split('_')[-1] if '_' in name_part else name_part
    first_time_str = all_predictions[0]['datetime'] if all_predictions else 'N/A'
    last_time_str = all_predictions[-1]['datetime'] if all_predictions else 'N/A'
    duration_worn_seconds = time_to_seconds(last_time_str) - time_to_seconds(first_time_str) if all_predictions else 0

    user_summary = {
        'user_id': '5',
        'name': name.capitalize(),
        'total_sessions': 1,
        'duration_worn': f"{abs(duration_worn_seconds):.2f} seconds",
        'last_active': last_time_str
    }

    return {
        'overview': {
            'total': total_predictions,
            'bagus': len(good_predictions),
            'jelek': len(bad_predictions),
            'bagus_pct': round((len(good_predictions) / total_predictions * 100) if total_predictions > 0 else 0, 1),
            'jelek_pct': round((len(bad_predictions) / total_predictions * 100) if total_predictions > 0 else 0, 1),
            'sensor_teraktif': 'MPU_ID 1' if sum(1 for r in all_predictions if r['mpu_id'] == '1') >= sum(1 for r in all_predictions if r['mpu_id'] == '2') else 'MPU_ID 2'
        },
        'durations': {
            'total_good': abs(total_duration_good),
            'total_bad': total_duration_bad,
            'avg_good': avg_duration_good,
            'avg_bad': avg_duration_bad
        },
        'intervals': {
            'avg': abs(avg_interval)
        },
        'buzzer_triggers': top_anomalies,
        'overall_data': {
            'labels': overall_data_labels,
            'datasets': [{
                'data': overall_data_values,
                'backgroundColor': overall_data_colors,
                'borderColor': overall_data_borderColors,
                'borderWidth': 1,
            }]
        },
        'posture_over_time_barchart': posture_over_time_barchart_data,
        'user_summary': user_summary
    }



def get_latest_model_input(file_id=None):
    """
    Membaca data terbaru dari file IMU (MPU1 & MPU2) dan BMP,
    lalu mengembalikan dictionary berisi 14 fitur yang diperlukan model.
    Jika data sensor tidak tersedia, nilai fitur diisi 0.
    
    Returns:
        dict: selalu berisi 'success': True dan 'data' (14 fitur)
    """
    # Helper functions
    def to_float(val):
        try:
            return float(val) if val not in (None, "") else 0.0
        except:
            return 0.0
    
    def to_int(val):
        try:
            return int(val) if val not in (None, "") else 0
        except:
            return 0
    
    # Inisialisasi semua fitur dengan 0
    features = {
        "ax1": 0.0, "ay1": 0.0, "az1": 0.0,
        "gx1": 0.0, "gy1": 0.0, "gz1": 0.0,
        "ax2": 0.0, "ay2": 0.0, "az2": 0.0,
        "gx2": 0.0, "gy2": 0.0, "gz2": 0.0,
        "pressure_hpa": 0.0,
        "altitude_m": 0.0
    }
    
    raw_sources = {
        "mpu1_time_ms": None,
        "mpu2_time_ms": None,
        "bmp_time_ms": None,
        "mpu1_datetime": None,
        "mpu2_datetime": None,
        "bmp_datetime": None
    }
    
    # 1. Baca data IMU dan BMP
    imu_rows = read_latest_rows_by_prefix("imu", file_id=file_id, max_lines=200)
    bmp_rows = read_latest_rows_by_prefix("bmp", file_id=file_id, max_lines=50)
    
    # 2. Pisahkan MPU1 dan MPU2
    mpu1_list = []
    mpu2_list = []
    for row in imu_rows:
        mpu_id = str(row.get("mpu_id", "")).strip()
        if mpu_id == "1":
            mpu1_list.append(row)
        elif mpu_id == "2":
            mpu2_list.append(row)
    
    # 3. Ambil data MPU1 dan MPU2 terbaru (tanpa sinkronisasi wajib)
    mpu1_row = None
    mpu2_row = None
    
    if mpu1_list:
        mpu1_row = mpu1_list[-1]  # ambil terbaru
        raw_sources["mpu1_time_ms"] = mpu1_row.get("epoch_ms")
        raw_sources["mpu1_datetime"] = mpu1_row.get("datetime")
    
    if mpu2_list:
        mpu2_row = mpu2_list[-1]
        raw_sources["mpu2_time_ms"] = mpu2_row.get("epoch_ms")
        raw_sources["mpu2_datetime"] = mpu2_row.get("datetime")
    
    # Jika salah satu MPU tidak ada, yang lainnya tetap dipakai (nilai 0 untuk yang hilang)
    # Tapi kita tetap berusaha mencari pasangan yang sinkron jika keduanya ada
    if mpu1_row and mpu2_row:
        # Coba cari pasangan sinkron (selisih <= 100ms)
        matched = None
        for row2 in reversed(mpu2_list):
            ts2 = to_int(row2.get("epoch_ms"))
            if ts2 == 0:
                continue
            best_match = None
            best_diff = float('inf')
            for row1 in reversed(mpu1_list):
                ts1 = to_int(row1.get("epoch_ms"))
                if ts1 == 0:
                    continue
                diff = abs(ts1 - ts2)
                if diff < best_diff and diff <= 100:
                    best_diff = diff
                    best_match = row1
            if best_match:
                matched = (best_match, row2)
                break
        if matched:
            mpu1_row, mpu2_row = matched
            raw_sources["mpu1_time_ms"] = mpu1_row.get("epoch_ms")
            raw_sources["mpu2_time_ms"] = mpu2_row.get("epoch_ms")
    
    # 4. Isi fitur dari MPU1 (jika ada)
    if mpu1_row:
        features["ax1"] = to_float(mpu1_row.get("ax"))
        features["ay1"] = to_float(mpu1_row.get("ay"))
        features["az1"] = to_float(mpu1_row.get("az"))
        features["gx1"] = to_float(mpu1_row.get("gx"))
        features["gy1"] = to_float(mpu1_row.get("gy"))
        features["gz1"] = to_float(mpu1_row.get("gz"))
    
    # 5. Isi fitur dari MPU2 (jika ada)
    if mpu2_row:
        features["ax2"] = to_float(mpu2_row.get("ax"))
        features["ay2"] = to_float(mpu2_row.get("ay"))
        features["az2"] = to_float(mpu2_row.get("az"))
        features["gx2"] = to_float(mpu2_row.get("gx"))
        features["gy2"] = to_float(mpu2_row.get("gy"))
        features["gz2"] = to_float(mpu2_row.get("gz"))
    
    # 6. Data BMP: cari yang terbaru (tidak perlu sinkron ketat)
    bmp_row = None
    if bmp_rows:
        bmp_row = bmp_rows[-1]  # ambil terbaru
        raw_sources["bmp_time_ms"] = bmp_row.get("epoch_ms")
        raw_sources["bmp_datetime"] = bmp_row.get("datetime")
        features["pressure_hpa"] = to_float(bmp_row.get("pressure_hpa"))
        features["altitude_m"] = to_float(bmp_row.get("altitude_m"))
    
    # 7. Kembalikan hasil (selalu success)
    return {
        "success": True,
        "data": features,
        "message": "Data berhasil disusun (nilai 0 untuk sensor yang tidak tersedia).",
        "raw_sources": raw_sources
    }

# --- Route: Home Page dengan Visualisasi Real-time ---
@app.route("/")
def index():
    file_id = request.args.get("id", "")
    error_message = request.args.get("error", "")
    return render_template('index.html', file_id=file_id, error_message=error_message)

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'GET':
        config = load_config()
        return jsonify(config)
    else:
        # POST: update config
        new_config = request.get_json()
        if not new_config:
            return jsonify({"error": "Invalid JSON"}), 400
        current = load_config()
        # update hanya field yang diizinkan
        for key in DEFAULT_CONFIG.keys():
            if key in new_config:
                current[key] = new_config[key]
        if save_config(current):
            return jsonify({"status": "ok", "config": current})
        else:
            return jsonify({"error": "Failed to save config"}), 500
        
@app.route('/settings')
def settings_page():
    return render_template('settings.html')

@app.route("/predict", methods=["GET"])
def predict_realtime():
    file_id = request.args.get("id")
    sanitized_id = sanitize_file_id(file_id)
    config = load_config()
    
    # --- Cek ketersediaan data sensor baru ---
    imu_rows_check = read_latest_rows_by_prefix("imu", file_id=sanitized_id, max_lines=5)
    latest_epoch_ms = 0

    for row in imu_rows_check:
        try:
            ep = int(row.get("epoch_ms", 0))
            if ep < 1500000000000: 
                dt_str = str(row.get("datetime", "")).strip()
                if dt_str:
                    if '.' in dt_str:
                        base, ms = dt_str.split('.')
                        dt_obj = datetime.datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
                        ep = int(dt_obj.timestamp() * 1000) + int(ms.ljust(3, '0')[:3])
                    else:
                        dt_obj = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                        ep = int(dt_obj.timestamp() * 1000)
            if ep > latest_epoch_ms:
                latest_epoch_ms = ep
        except Exception:
            pass

    current_time_ms = int(datetime.datetime.now().timestamp() * 1000)
    stale_limit = config.get("sensor_stale_timeout_ms", 5000)

    if latest_epoch_ms == 0 or (current_time_ms - latest_epoch_ms) > stale_limit:
        return jsonify({
            "prediction": -1,
            "label": "Tidak ada data",
            "probabilities": [],
            "confidence": 0.0,
            "error": "No recent sensor data",
            "debug_info": f"Server Time: {current_time_ms}, Data Time: {latest_epoch_ms}"
        }), 200

    # --- Ambil data WINDOW untuk ekstraksi fitur ---
    try:
        # Ambil lebih banyak baris untuk membentuk window (misal 300 baris)
        imu_rows = read_latest_rows_by_prefix("imu", file_id=sanitized_id, max_lines=300)
        bmp_rows = read_latest_rows_by_prefix("bmp", file_id=sanitized_id, max_lines=50)
        
        mpu1_list = [r for r in imu_rows if str(r.get("mpu_id", "")).strip() == "1"]
        mpu2_list = [r for r in imu_rows if str(r.get("mpu_id", "")).strip() == "2"]
        
        # Samakan panjang data MPU1 dan MPU2
        min_len = min(len(mpu1_list), len(mpu2_list))
        
        if min_len < 10:
            return jsonify({
                "prediction": -1,
                "label": "Mengumpulkan data...",
                "probabilities": [],
                "confidence": 0.0,
                "error": "Not enough window data"
            }), 200

        mpu1_window = mpu1_list[-min_len:]
        mpu2_window = mpu2_list[-min_len:]
        
        def to_f(val):
            try: return float(val) if val not in (None, "", " ") else 0.0
            except: return 0.0

        # Buat DataFrame Window
        df_window = pd.DataFrame({
            'ax1': [to_f(r.get('ax')) for r in mpu1_window],
            'ay1': [to_f(r.get('ay')) for r in mpu1_window],
            'az1': [to_f(r.get('az')) for r in mpu1_window],
            'gx1': [to_f(r.get('gx')) for r in mpu1_window],
            'gy1': [to_f(r.get('gy')) for r in mpu1_window],
            'gz1': [to_f(r.get('gz')) for r in mpu1_window],
            'ax2': [to_f(r.get('ax')) for r in mpu2_window],
            'ay2': [to_f(r.get('ay')) for r in mpu2_window],
            'az2': [to_f(r.get('az')) for r in mpu2_window],
            'gx2': [to_f(r.get('gx')) for r in mpu2_window],
            'gy2': [to_f(r.get('gy')) for r in mpu2_window],
            'gz2': [to_f(r.get('gz')) for r in mpu2_window],
        })
        
        # Ambil data BME terbaru dan terapkan ke seluruh window
        bmp_row = bmp_rows[-1] if bmp_rows else None
        p_val = to_f(bmp_row.get('pressure_hpa')) if bmp_row else 0.0
        a_val = to_f(bmp_row.get('altitude_m')) if bmp_row else 0.0
        
        df_window['pressure_hpa'] = p_val
        df_window['altitude_m'] = a_val

        # 1. Ekstraksi Fitur Jendela (Sesuai Notebook)
        features_dict = extract_window_features(df_window)
        
        # 2. Sesuaikan urutan dan nama kolom dengan EXPECTED_FEATURES
        final_features = {k: features_dict.get(k, 0.0) for k in EXPECTED_FEATURES}
        df_final = pd.DataFrame([final_features], columns=EXPECTED_FEATURES)

        # 3. Prediksi Model
        scaler = load_scaler()
        model = load_model()
        
        X_scaled = scaler.transform(df_final)
        proba = model.predict_proba(X_scaled)[0]   
        pred_idx = int(model.predict(X_scaled)[0])
        confidence = float(max(proba))

        mapping = {0: "Salah", 1: "Benar", 2: "Berdiri"}
        label = mapping.get(pred_idx, "Tidak diketahui")

        threshold = config.get("confidence_threshold", 0.2)
        if confidence < threshold:
            label = "Tidak yakin"
            pred_idx = -2

        if label == "Salah" and sanitized_id and config.get("buzzer_enabled", True):
            duration = config.get("buzzer_duration", 1.5)
            send_buzzer_command(sanitized_id, turn_on=True)
            threading.Timer(duration, lambda: send_buzzer_command(sanitized_id, turn_on=False)).start()

        return jsonify({
            "prediction": pred_idx,
            "label": label,
            "probabilities": proba.tolist(),
            "confidence": confidence,
            "threshold_used": threshold
        })
        
    except Exception as e:
        import traceback
        error_detail = str(e)
        print("=== PREDICTION ERROR ===")
        traceback.print_exc()
        print("========================")
        
        return jsonify({
            "prediction": -1,
            "label": "Error Model",
            "probabilities": [],
            "confidence": 0.0,
            "error": error_detail,
            "debug_info": "Gagal memproses model. Cek terminal Flask untuk detail."
        }), 200


@app.route("/model_input", methods=["GET"])
def show_model_input():
    """
    Menampilkan data yang telah diformat sesuai kebutuhan model.
    Tidak melakukan prediksi, hanya menampilkan hasil olahan data.
    Parameter query string:
        - id (optional): file_id untuk filter dataUser/
        - format (optional): 'json' atau 'html' (default json)
    """
    file_id = request.args.get("id")
    sanitized_id = sanitize_file_id(file_id)
    output_format = request.args.get("format", "json").lower()
    
    result = get_latest_model_input(file_id=sanitized_id)
    
    if output_format == "html":
        # Tampilkan dalam halaman HTML sederhana
        from flask import render_template_string
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Input Model (Tanpa Prediksi)</title>
            <style>
                body { font-family: monospace; margin: 2em; }
                pre { background: #f4f4f4; padding: 1em; border-radius: 5px; }
                .success { color: green; }
                .error { color: red; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Data Siap Model (14 Fitur)</h1>
            {% if result.success %}
                <p class="success">✅ {{ result.message }}</p>
                <h2>Fitur yang akan diberikan ke model:</h2>
                <table>
                    <tr><th>Fitur</th><th>Nilai</th></tr>
                    {% for key, value in result.data.items() %}
                    <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
                    {% endfor %}
                </table>
                <h2>Metadata Sinkronisasi:</h2>
                <pre>{{ result.raw_sources | tojson(indent=2) }}</pre>
            {% else %}
                <p class="error">❌ {{ result.message }}</p>
            {% endif %}
            <hr>
            <p><small>Gunakan parameter <code>?format=json</code> untuk mendapatkan JSON mentah.</small></p>
        </body>
        </html>
        """
        return render_template_string(html_template, result=result)
    else:
        # Default JSON response
        return jsonify({
            "status": "success" if result["success"] else "error",
            "message": result["message"],
            "model_input_data": result.get("data"),
            "alignment_info": result.get("raw_sources")
        }), 200 if result["success"] else 404

# --- Route: Dashboard ---
@app.route("/dashboard")
def dashboard():
    file_path = os.path.join(SAVE_DIR, "01.csv")
    
    if not os.path.exists(file_path):
        available_files = os.listdir(SAVE_DIR) if os.path.exists(SAVE_DIR) else []
        error_msg = f"File not found in directory {SAVE_DIR}. Available files: {available_files}"
        return redirect(url_for('index', error=error_msg))

    data = get_dashboard_data(file_path)

    if not data:
        error_msg = "Data from file is invalid, empty, or not found after parsing."
        return redirect(url_for('index', error=error_msg))

    return render_template('dashboard.html', data=data)


# ---------- existing upload endpoint ----------
@app.route("/data", methods=["POST"])
def upload_data():
    try:
        print("---- NEW POST /data ----")
        print("From:", request.remote_addr)
        print("Headers:")
        for k, v in request.headers.items():
            print(f"  {k}: {v}")
        raw = request.get_data(as_text=True)
        print("Raw body:", raw)
    except Exception as e:
        print("Logging error:", e)

    data = None
    try:
        data = request.get_json(force=True, silent=True)
    except Exception as e:
        print("get_json exception:", e)
        data = None

    if not data:
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
            "file_id": request.form.get("file_id"),
        }

    # Deteksi sensor BMP280
    if data and ("sensor" in data or "temperature_c" in data):
        return handle_bmp_data(data)
    else:
        return handle_mpu_data(data)

def handle_mpu_data(data):
    raw_id = data.get("file_id") if isinstance(data, dict) else None
    file_id = sanitize_file_id(raw_id)

    if file_id and "esp_ip" in data:
        esp_ip_map[file_id] = data["esp_ip"]
        print(f"[INFO] ESP IP for {file_id}: {data['esp_ip']}")

    filepath = get_today_filepath(prefix="imu", file_id=file_id)      # asumsi fungsi ini sudah ada
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)

    try:
        with open(filepath, "a", newline='') as f:
            if not file_exists:
                f.write("datetime,epoch_ms,ts_millis,mpu_id,ax,ay,az,gx,gy,gz\n")

            now = data.get("datetime") or datetime.datetime.now().isoformat()
            epoch_ms = data.get("epoch_ms") or ""
            ts_millis = data.get("ts_millis") or ""
            mpu_id = data.get("mpu_id") or ""
            ax = data.get("ax") or ""
            ay = data.get("ay") or ""
            az = data.get("az") or ""
            gx = data.get("gx") or ""
            gy = data.get("gy") or ""
            gz = data.get("gz") or ""
            

            f.write(f"{now},{epoch_ms},{ts_millis},{mpu_id},{ax},{ay},{az},{gx},{gy},{gz}\n")
    except Exception as e:
        print("Error writing file:", e)
        return jsonify({"status":"error","reason":str(e)}), 500

    print("Saved MPU to", os.path.basename(filepath))
    return jsonify({"status":"ok", "saved_to": os.path.basename(filepath)})

def handle_bmp_data(data):
    raw_id = data.get("file_id") if isinstance(data, dict) else None
    file_id = sanitize_file_id(raw_id)
    # Gunakan fungsi get_today_filepath_bmp atau modifikasi get_today_filepath dengan prefix
    filepath = get_today_filepath(prefix="bmp", file_id=file_id)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)

    try:
        with open(filepath, "a", newline='') as f:
            if not file_exists:
                f.write("datetime,epoch_ms,ts_millis,sensor,temperature_c,pressure_hpa,altitude_m\n")

            now = data.get("datetime") or datetime.datetime.now().isoformat()
            epoch_ms = data.get("epoch_ms") or ""
            ts_millis = data.get("ts_millis") or ""
            sensor = data.get("sensor") or "3"
            temp = data.get("temperature_c") or ""
            press = data.get("pressure_hpa") or ""
            alt = data.get("altitude_m") or ""

            f.write(f"{now},{epoch_ms},{ts_millis},{sensor},{temp},{press},{alt}\n")
    except Exception as e:
        print("Error writing BMP file:", e)
        return jsonify({"status":"error","reason":str(e)}), 500

    print("Saved BMP to", os.path.basename(filepath))
    return jsonify({"status":"ok", "saved_to": os.path.basename(filepath)})

# ---------- helper: read tail lines from file with given prefix ----------
def read_latest_rows_by_prefix(prefix, file_id=None, max_lines=500):
    """
    Membaca baris terakhir dari file CSV dengan prefix tertentu.
    prefix: 'imu' untuk data MPU, 'bmp' untuk data BME280
    """
    path = get_today_filepath(prefix=prefix, file_id=file_id)
    if not os.path.exists(path):
        return []

    dq = deque(maxlen=max_lines)
    with open(path, "r", newline='') as f:
        for line in f:
            dq.append(line.rstrip("\n"))

    rows = []
    if not dq:
        return rows

    all_lines = list(dq)
    # Lewati header jika ada (baris pertama diawali "datetime")
    if all_lines and all_lines[0].startswith("datetime"):
        data_lines = all_lines[1:]
    else:
        data_lines = all_lines

    for ln in data_lines:
        if not ln.strip():
            continue
        try:
            parts = list(csv.reader([ln]))[0]
            if prefix == "imu":  # data MPU (10 kolom)
                while len(parts) < 10:
                    parts.append("")
                d = {
                    "datetime": parts[0],
                    "epoch_ms": parts[1],
                    "ts_millis": parts[2],
                    "mpu_id": parts[3],    # tambahan
                    "ax": parts[4],
                    "ay": parts[5],
                    "az": parts[6],
                    "gx": parts[7],
                    "gy": parts[8],
                    "gz": parts[9],
                }
            else:  # data BME280 (prefix 'bmp', minimal 7 kolom)
                if len(parts) < 7:
                    continue
                d = {
                    "datetime": parts[0],
                    "epoch_ms": parts[1],
                    "ts_millis": parts[2],
                    "sensor": parts[3],
                    "temperature_c": float(parts[4]) if parts[4] else None,
                    "pressure_hpa": float(parts[5]) if parts[5] else None,
                    "altitude_m": float(parts[6]) if parts[6] else None,
                }
            rows.append(d)
        except Exception:
            continue
    return rows

# ---------- endpoint: return latest data from MPU and BME280 ----------
@app.route("/latest")
def latest_json():
    raw_id = request.args.get("id")
    file_id = sanitize_file_id(raw_id)
    try:
        n = int(request.args.get("n", "500"))
    except:
        n = 500

    # Baca data MPU (file prefix "imu")
    mpu_rows = read_latest_rows_by_prefix("imu", file_id=file_id, max_lines=n+10)

    mpu1 = []
    mpu2 = []
    for r in mpu_rows:
        def tofloat(x):
            try:
                return float(x)
            except:
                return None
        entry = {
            "datetime": r["datetime"],
            "epoch_ms": int(r["epoch_ms"]) if str(r["epoch_ms"]).isdigit() else r["epoch_ms"],
            "ts_millis": r["ts_millis"],
            "mpu_id": r["mpu_id"],
            "ax": tofloat(r["ax"]),
            "ay": tofloat(r["ay"]),
            "az": tofloat(r["az"]),
            "gx": tofloat(r["gx"]),
            "gy": tofloat(r["gy"]),
            "gz": tofloat(r["gz"]),
        }
        if str(r["mpu_id"]) == "1" or str(r["mpu_id"]).lower() == "1":
            mpu1.append(entry)
        elif str(r["mpu_id"]) == "2" or str(r["mpu_id"]).lower() == "2":
            mpu2.append(entry)

    # Baca data BME280 (file prefix "bmp")
    bme_rows = read_latest_rows_by_prefix("bmp", file_id=file_id, max_lines=n)

    # Kirimkan semua data dalam satu respons JSON
    return jsonify({
        "mpu1": mpu1,
        "mpu2": mpu2,
        "bme": bme_rows
    })



# ---------- Route: Download CSV dengan format custom ----------
@app.route("/download", methods=["GET"])
def download_data():
    subject_no = request.args.get("subject_no", "").strip()
    subject_name = request.args.get("subject_name", "").strip()
    
    if not subject_no or not subject_name:
        return jsonify({"error": "Missing subject_no or subject_name"}), 400
    
    # Sanitize nama file
    subject_no_safe = sanitize_file_id(subject_no)
    subject_name_safe = sanitize_file_id(subject_name)
    
    # Format: no.subjek_nama_dd_mm_yy.csv
    today = datetime.datetime.now()
    date_format = today.strftime("%d_%m_%y")
    filename = f"{subject_no_safe}_{subject_name_safe}_{date_format}.csv"
    filepath = os.path.join(SAVE_DIR, filename)
    
    # Cari file yang ada di SAVE_DIR yang cocok dengan pattern
    available_files = os.listdir(SAVE_DIR) if os.path.exists(SAVE_DIR) else []
    matching_file = None
    
    for f in available_files:
        if subject_no_safe in f and subject_name_safe in f:
            matching_file = os.path.join(SAVE_DIR, f)
            break
    
    if not matching_file or not os.path.exists(matching_file):
        return jsonify({"error": f"File for subject {subject_no} ({subject_name}) not found"}), 404
    
    try:
        return send_file(matching_file, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000, debug=True)
