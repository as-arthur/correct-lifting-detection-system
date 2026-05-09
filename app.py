# app.py
import os
import re
import csv
import math
import datetime
from collections import deque, defaultdict
from flask import Flask, request, send_file, abort, jsonify, render_template, redirect, url_for

SAVE_DIR = r"dataUser"
os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# sanitize file id: hanya huruf, angka, underscore, dash; sisanya diganti underscore
def sanitize_file_id(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)

def get_today_filepath(file_id=None, date_obj=None):
    if date_obj is None:
        date_obj = datetime.datetime.now()
    today = date_obj.strftime("%Y-%m-%d")
    if file_id:
        return os.path.join(SAVE_DIR, f"{today}_{file_id}.csv")
    else:
        return os.path.join(SAVE_DIR, f"{today}.csv")

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


# --- Route: Home Page dengan Visualisasi Real-time ---
@app.route("/")
def index():
    file_id = request.args.get("id", "")
    error_message = request.args.get("error", "")
    return render_template('index.html', file_id=file_id, error_message=error_message)


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

    raw_id = data.get("file_id") if isinstance(data, dict) else None
    file_id = sanitize_file_id(raw_id)
    filepath = get_today_filepath(file_id=file_id)
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

    print("Saved to", os.path.basename(filepath))
    return jsonify({"status":"ok", "saved_to": os.path.basename(filepath)})

# ---------- helper: read tail lines from today's file and parse CSV ----------
def read_latest_rows(file_id=None, max_lines=500):
    path = get_today_filepath(file_id=file_id)
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
    header = None
    if all_lines and all_lines[0].startswith("datetime"):
        header = all_lines[0].split(",")
        data_lines = all_lines[1:]
    else:
        with open(path, "r", newline='') as f:
            first = f.readline()
            if first.startswith("datetime"):
                header = first.strip().split(",")
        data_lines = all_lines

    for ln in data_lines:
        if not ln.strip():
            continue
        try:
            parts = list(csv.reader([ln]))[0]
            while len(parts) < 10:
                parts.append("")
            d = {
                "datetime": parts[0],
                "epoch_ms": parts[1],
                "ts_millis": parts[2],
                "mpu_id": parts[3],
                "ax": parts[4],
                "ay": parts[5],
                "az": parts[6],
                "gx": parts[7],
                "gy": parts[8],
                "gz": parts[9],
            }
            rows.append(d)
        except Exception:
            continue
    return rows

# ---------- endpoint: return latest data grouped per MPU ----------
@app.route("/latest")
def latest_json():
    raw_id = request.args.get("id")
    file_id = sanitize_file_id(raw_id)
    try:
        n = int(request.args.get("n", "500"))
    except:
        n = 500

    rows = read_latest_rows(file_id=file_id, max_lines=n+10)

    mpu1 = []
    mpu2 = []
    for r in rows:
        def tofloat(x):
            try:
                return float(x)
            except:
                return None
        entry = {
            "datetime": r["datetime"],
            "epoch_ms": int(r["epoch_ms"]) if r["epoch_ms"].isdigit() else r["epoch_ms"],
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

    return jsonify({"mpu1": mpu1, "mpu2": mpu2})

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
