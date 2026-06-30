# app.py
import datetime
import uuid
import pandas as pd
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
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from collections import deque, defaultdict
from model_loader import get_pipeline, load_config, save_config
from flask import Flask, request, send_file, abort, jsonify, render_template, redirect, url_for
from sklearn.preprocessing import RobustScaler
from collections import deque, defaultdict


SAVE_DIR = r"dataUser"
os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
esp_ip_map = {}

def send_buzzer_command(file_id, turn_on=True):
    """Kirim perintah ke ESP untuk menyalakan/mematikan buzzer"""
    # Gunakan IP default 10.124.67.39 jika file_id tidak ada di map
    ip = esp_ip_map.get(file_id, "10.109.43.39")
    state = "1" if turn_on else "0"
    url = f"http://{ip}/buzzer?state={state}"
    
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            print(f"[OK] Buzzer -> {'ON' if turn_on else 'OFF'} ")
            return True
        else:
            print(f"[ERROR] Gagal, status {resp.status_code} ")
            return False
    except Exception as e:
        print(f"[ERROR] Gagal kirim buzzer: {e} ")
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

def add_rolling_features(X_df, n=3):
   
    X = X_df.copy()

    # Pilih hanya fitur yang paling relevan untuk rolling
    key_cols = [c for c in ['acc_mag1','acc_mag2','gyro_mag1','gyro_mag2',
                             'roll1','pitch1','diff_roll','diff_pitch']
                if c in X.columns]

    for col in key_cols:
        # Ensure column is numeric before rolling operation
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X[f'{col}_rm{n}'] = (X[col].rolling(n, min_periods=1)
                                   .mean())
        X[f'{col}_rs{n}'] = (X[col].rolling(n, min_periods=1)
                                   .std()
                                   .fillna(0))
    return X

def add_rolling_per_subject(X_df, y_df, groups_arr, n=3):
    """Rolling dihitung ulang per subjek agar tidak melewati batas."""
    parts_X, parts_y = [], []
    for subj in np.unique(groups_arr):
        mask = groups_arr == subj
        X_s = X_df[mask].copy()
        X_s = add_rolling_features(X_s, n=n)
        parts_X.append(X_s)
        parts_y.append(y_df[mask])
    return pd.concat(parts_X).reset_index(drop=True), pd.concat(parts_y).reset_index(drop=True)    

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

    features = {
        "ax1": 0.0, "ay1": 0.0, "az1": 0.0,
        "gx1": 0.0, "gy1": 0.0, "gz1": 0.0,
        "ax2": 0.0, "ay2": 0.0, "az2": 0.0,
        "gx2": 0.0, "gy2": 0.0, "gz2": 0.0,
        "acc_mag_s1": 0.0, "acc_mag_s2": 0.0,
        "gyro_mag_s1": 0.0, "gyro_mag_s2": 0.0
    }

    raw_sources = {
        "mpu1_time_ms": None,
        "mpu2_time_ms": None,
        "mpu1_datetime": None,
        "mpu2_datetime": None,
    }

    try:
        imu_rows = read_latest_rows_by_prefix("imu", file_id=file_id, max_lines=200)
    except Exception as e:
        return {"success": False, "message": f"Gagal membaca file IMU: {e}", "data": features, "raw_sources": raw_sources}

    mpu1_list = []
    mpu2_list = []
    for row in imu_rows:
        mpu_id = str(row.get("mpu_id", "")).strip()
        if mpu_id == "1":
            mpu1_list.append(row)
        elif mpu_id == "2":
            mpu2_list.append(row)

    mpu1_row = mpu1_list[-1] if mpu1_list else None
    mpu2_row = mpu2_list[-1] if mpu2_list else None

    if mpu1_row:
        raw_sources["mpu1_time_ms"] = mpu1_row.get("epoch_ms")
        raw_sources["mpu1_datetime"] = mpu1_row.get("datetime")
    if mpu2_row:
        raw_sources["mpu2_time_ms"] = mpu2_row.get("epoch_ms")
        raw_sources["mpu2_datetime"] = mpu2_row.get("datetime")

    # Sinkronisasi (jika kedua sensor ada)
    if mpu1_row and mpu2_row:
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

    # Isi fitur dari MPU1
    if mpu1_row:
        ax1 = to_float(mpu1_row.get("ax"))
        ay1 = to_float(mpu1_row.get("ay"))
        az1 = to_float(mpu1_row.get("az"))
        gx1 = to_float(mpu1_row.get("gx"))
        gy1 = to_float(mpu1_row.get("gy"))
        gz1 = to_float(mpu1_row.get("gz"))
        features.update({
            "ax1": ax1, "ay1": ay1, "az1": az1,
            "gx1": gx1, "gy1": gy1, "gz1": gz1,
            "acc_mag_s1": (ax1**2 + ay1**2 + az1**2)**0.5,
            "gyro_mag_s1": (gx1**2 + gy1**2 + gz1**2)**0.5
        })

    # Isi fitur dari MPU2
    if mpu2_row:
        ax2 = to_float(mpu2_row.get("ax"))
        ay2 = to_float(mpu2_row.get("ay"))
        az2 = to_float(mpu2_row.get("az"))
        gx2 = to_float(mpu2_row.get("gx"))
        gy2 = to_float(mpu2_row.get("gy"))
        gz2 = to_float(mpu2_row.get("gz"))
        features.update({
            "ax2": ax2, "ay2": ay2, "az2": az2,
            "gx2": gx2, "gy2": gy2, "gz2": gz2,
            "acc_mag_s2": (ax2**2 + ay2**2 + az2**2)**0.5,
            "gyro_mag_s2": (gx2**2 + gy2**2 + gz2**2)**0.5
        })

    return {
        "success": True,
        "data": features,
        "message": "Data berhasil disusun.",
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'session_id' not in data or 'sample' not in data:
        return jsonify({"error": "Missing session_id or sample"}), 400

    session_id = data['session_id']
    sample = data['sample'] # Frontend sudah mengirim 14 kolom langsung (ax1, ay1, dst)

    # 1. Ambil instance pipeline singleton
    pipeline = get_pipeline()

    # 2. Jalankan prediksi
    result = pipeline.predict(sample, session_id=session_id)

    # 3. Mapping hasil pipeline ke format response
    pred_class = result['prediction']
    prob_salah = result['probabilities']['salah']   
    prob_benar = result['probabilities']['benar']   
    extracted_features = result.get('extracted_features', {})  # <--- AMBIL FITUR HASIL EKSTRAKSI

    thresh_ergo = float(request.args.get('thresh_ergo', 0.70))
    thresh_non = float(request.args.get('thresh_non', 0.60))

    # Tentukan label
    if prob_salah >= thresh_non:
        pred = 1
        label = "salah"
        send_buzzer_command(session_id, turn_on=True)  # Nyalakan buzzer untuk "salah"
    elif prob_benar >= thresh_ergo:
        pred = 0
        label = "benar"
        send_buzzer_command(session_id, turn_on=False)  # Matikan buzzer untuk "benar"
    else:
        pred = -1
        label = "tidak pasti"


    # === LOGIKA LOG CSV (history/prediction_log.csv) ===
    try:
        log_dir = 'history'
        log_file = os.path.join(log_dir, 'prediction_log.csv')
        
        # Buat folder 'history' jika belum ada
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        now = datetime.datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        confidence = max(prob_benar, prob_salah)
        
        # 1. Susun data log dasar (Tanpa ID Unik)
        log_data = {
            'date': date_str,
            'time': time_str,
            'prediction': label,
            'confidence': round(confidence, 4)
        }
        
        
        # 3. Tambahkan semua fitur hasil ekstraksi (46 kolom)
        # Catatan: Ini akan menimpa 14 kolom mentah dengan nilai yang sama persis, memastikan konsistensi
        log_data.update(extracted_features)
        
        file_exists = os.path.isfile(log_file)
        
        # Tentukan fieldnames (header CSV)
        if file_exists:
            # Jika file sudah ada, baca header agar urutan kolom tetap konsisten
            with open(log_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                fieldnames = next(reader, None)
        else:
            # Header baru: date, time, prediction, confidence, lalu raw features, lalu extracted features
            fieldnames = ['date', 'time', 'prediction', 'confidence'] + list(extracted_features.keys())
            
        # Tulis ke CSV
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)
            
    except Exception as e:
        print(f"[ERROR] Gagal menulis ke log CSV: {e}")
    # ================================================

    return jsonify({
        "status": "ready",
        "prediction": pred,
        "label": label,
        "probabilities": [prob_benar, prob_salah],
        "thresholds_used": {"benar": thresh_ergo, "salah": thresh_non}
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    if not data or 'window' not in data:
        return jsonify({"error": "Missing 'window' array"}), 400
    
    window_data = data['window']
    if len(window_data) < 50:
        return jsonify({"error": f"Window size must be at least 50, got {len(window_data)}"}), 400

    # ... (Logika pemisahan mpu1 dan mpu2 tetap sama) ...
    
    # Buat DataFrame dengan 14 kolom mentah sesuai RAW_SENSOR_COLS di preprocessing.py
    df = pd.DataFrame({
        'ax1': [d.get('ax', 0) for d in mpu1], 'ay1': [d.get('ay', 0) for d in mpu1], 'az1': [d.get('az', 0) for d in mpu1],
        'gx1': [d.get('gx', 0) for d in mpu1], 'gy1': [d.get('gy', 0) for d in mpu1], 'gz1': [d.get('gz', 0) for d in mpu1],
        'ax2': [d.get('ax', 0) for d in mpu2], 'ay2': [d.get('ay', 0) for d in mpu2], 'az2': [d.get('az', 0) for d in mpu2],
        'gx2': [d.get('gx', 0) for d in mpu2], 'gy2': [d.get('gy', 0) for d in mpu2], 'gz2': [d.get('gz', 0) for d in mpu2],
        'pressure_hpa': 0.0,
        'altitude_m': 0.0
    })

    # Import fungsi preprocessing yang sudah ada
    from preprocessing import add_derived_features, add_rolling_features
    
    # Terapkan feature engineering secara manual untuk batch
    df_derived = add_derived_features(df)
    df_rolled = add_rolling_features(df_derived, n=3)
    df_rolled = df_rolled.replace([np.inf, -np.inf], 0).fillna(0)

    # Ambil baris terakhir sebagai representasi window saat ini
    df_current = df_rolled.iloc[[-1]].copy()

    # Ambil pipeline untuk mendapatkan expected_features, scaler, dan model
    pipeline = get_pipeline()
    
    # Align kolom (pastikan urutan dan kelengkapan kolom)
    for col in pipeline.expected_features:
        if col not in df_current.columns:
            df_current[col] = 0.0
    df_aligned = df_current[pipeline.expected_features]

    # Scaling dan Prediksi
    X_scaled = pipeline.scaler.transform(df_aligned)
    proba = pipeline.model.predict_proba(X_scaled)[0]
    
    prob_benar = float(proba[list(pipeline.model.classes_).index(1)]) if 1 in pipeline.model.classes_ else 0.0
    prob_salah = float(proba[list(pipeline.model.classes_).index(0)]) if 0 in pipeline.model.classes_ else 0.0

    thresh_ergo = float(request.args.get('thresh_ergo', 0.70))
    thresh_non = float(request.args.get('thresh_non', 0.60))

    if prob_salah >= thresh_non:
        pred = 1
        label = "salah"
    elif prob_benar >= thresh_ergo:
        pred = 0
        label = "benar"
    else:
        pred = -1
        label = "tidak pasti"

    return jsonify({
        "prediction": pred,
        "label": label,
        "probabilities": [prob_benar, prob_salah],
        "thresholds_used": {"benar": thresh_ergo, "salah": thresh_non}
    })


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

            buffer_entry = {
            'datetime': now,
            'epoch_ms': epoch_ms,
            'ts_millis': ts_millis,
            'mpu_id': mpu_id,
            'ax': float(ax) if ax else 0.0,
            'ay': float(ay) if ay else 0.0,
            'az': float(az) if az else 0.0,
            'gx': float(gx) if gx else 0.0,
            'gy': float(gy) if gy else 0.0,
            'gz': float(gz) if gz else 0.0,
            }
            

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
    path = get_today_filepath(prefix=prefix, file_id=file_id)
    
    # Fallback: Jika file dengan file_id tidak ada, cari file tanpa file_id
    if not os.path.exists(path) and file_id is not None:
        fallback_path = get_today_filepath(prefix=prefix, file_id=None)
        if os.path.exists(fallback_path):
            path = fallback_path
        else:
            return []
    elif not os.path.exists(path):
        return []

    # === BACA FILE SUPER CEPAT (Chunk-based backwards read) ===
    chunks = []
    total_newlines = 0
    chunk_size = 8192  # Baca 8KB per iterasi
    
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        
        while file_size > 0:
            read_size = min(chunk_size, file_size)
            file_size -= read_size
            f.seek(file_size)
            chunk = f.read(read_size)
            chunks.append(chunk)
            total_newlines += chunk.count(b'\n')
            
            # Berhenti jika sudah dapat cukup baris
            if total_newlines >= max_lines + 2:
                break
                
    # Balik urutan chunk agar kronologis, lalu gabungkan
    chunks.reverse()
    buffer = b''.join(chunks)
    
    # Decode dan pisahkan per baris
    text = buffer.decode('utf-8', errors='ignore')
    all_lines = text.splitlines()
    
    # Lewati header jika ada di baris paling pertama
    if all_lines and all_lines[0].startswith('datetime'):
        all_lines = all_lines[1:]
        
    # Ambil HANYA max_lines terakhir
    data_lines = all_lines[-max_lines:]
    
    # === Parsing CSV (Sama seperti sebelumnya) ===
    rows = []
    for ln in data_lines:
        if not ln.strip(): continue
        try:
            parts = list(csv.reader([ln]))[0]
            if prefix == "imu":
                while len(parts) < 10: parts.append("")
                d = {
                    "datetime": parts[0], "epoch_ms": parts[1], "ts_millis": parts[2],
                    "mpu_id": parts[3], "ax": parts[4], "ay": parts[5], "az": parts[6],
                    "gx": parts[7], "gy": parts[8], "gz": parts[9],
                }
            else:
                if len(parts) < 7: continue
                d = {
                    "datetime": parts[0], "epoch_ms": parts[1], "ts_millis": parts[2],
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
    
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
