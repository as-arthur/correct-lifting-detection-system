import os
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pandas')

def clean_dataframe(df, required_cols=None):
    """
    Membersihkan dataframe:
    - Menghapus baris yang semua kolom numeriknya kosong (setelah datetime)
    - Menghapus baris yang kolom 'epoch_ms' kosong (wajib ada)
    - Mengisi NaN dengan None agar tidak error saat konversi
    """
    if df.empty:
        return df
    
    # Hapus baris yang epoch_ms kosong (data tidak berguna)
    if 'epoch_ms' in df.columns:
        df = df.dropna(subset=['epoch_ms'])
    
    # Hapus baris yang semua kolom (selain datetime) kosong
    # Tentukan kolom non-datetime
    non_dt_cols = [c for c in df.columns if c != 'datetime']
    if non_dt_cols:
        # Baris dianggap kosong jika semua nilai di non_dt_cols adalah NaN atau string kosong
        mask = df[non_dt_cols].apply(lambda row: row.isna().all() or (row.astype(str).str.strip() == '').all(), axis=1)
        df = df[~mask]
    
    return df

def extract_info_from_filename(filename):
    """
    Ekstrak tipe sensor, tanggal, dan nama dari nama file.
    Contoh: bmp_2026-06-01(adnin).csv -> ('bmp', '2026-06-01', 'adnin')
    """
    pattern = r'(imu|bmp)_(\d{4}-\d{2}-\d{2})\(([^)]+)\)(?:\.csv)?$'
    match = re.match(pattern, os.path.basename(filename))
    if match:
        return match.groups()
    return None, None, None

def safe_read_csv(filepath):
    """
    Membaca CSV dengan penanganan baris buruk dan encoding fallback.
    """
    try:
        # Coba baca dengan engine='python' agar lebih toleran
        df = pd.read_csv(filepath, on_bad_lines='skip', engine='python', encoding='utf-8')
        if df.empty:
            raise ValueError("File kosong setelah skip baris buruk")
        return df
    except Exception as e:
        print(f"Gagal membaca {filepath} dengan utf-8: {e}")
        # Fallback ke latin-1
        df = pd.read_csv(filepath, on_bad_lines='skip', engine='python', encoding='latin-1')
        return df

def pair_imu_sensors(imu_df, time_col='epoch_ms', tolerance_ms=50):
    """
    Menggabungkan dua sensor IMU (mpu_id=1 dan mpu_id=2) berdasarkan waktu terdekat.
    Menghasilkan satu baris per timestamp unik dengan kedua sensor.
    """
    # Filter mpu_id
    df1 = imu_df[imu_df['mpu_id'] == 1].copy()
    df2 = imu_df[imu_df['mpu_id'] == 2].copy()


    if df1.empty or df2.empty:
        raise ValueError("Data IMU harus mengandung mpu_id=1 dan mpu_id=2")
    
    # Konversi kolom numerik jika perlu
    for col in ['ax','ay','az','gx','gy','gz']:
        if col in df1.columns:
            df1[col] = pd.to_numeric(df1[col], errors='coerce')
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
    
    # Siapkan kolom untuk sensor 1
    df1_renamed = df1[[time_col, 'datetime', 'ts_millis', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']].rename(
        columns={'ax': 'ax1', 'ay': 'ay1', 'az': 'az1',
                 'gx': 'gx1', 'gy': 'gy1', 'gz': 'gz1',
                 'datetime': 'datetime1', 'ts_millis': 'ts_millis1'}
    )
    
    # Siapkan kolom untuk sensor 2
    df2_renamed = df2[[time_col, 'datetime', 'ts_millis', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']].rename(
        columns={'ax': 'ax2', 'ay': 'ay2', 'az': 'az2',
                 'gx': 'gx2', 'gy': 'gy2', 'gz': 'gz2',
                 'datetime': 'datetime2', 'ts_millis': 'ts_millis2'}
    )
    
    # Timeline: semua timestamp unik dari kedua sensor
    all_times = sorted(pd.concat([df1[time_col], df2[time_col]]).unique())
    timeline = pd.DataFrame({time_col: all_times})
    
    # Merge nearest untuk sensor 1
    merged1 = pd.merge_asof(
        timeline.sort_values(time_col),
        df1_renamed.sort_values(time_col),
        on=time_col,
        direction='nearest',
        tolerance=tolerance_ms
    )
    
    # Merge nearest untuk sensor 2
    merged2 = pd.merge_asof(
        timeline.sort_values(time_col),
        df2_renamed.sort_values(time_col),
        on=time_col,
        direction='nearest',
        tolerance=tolerance_ms
    )
    
    # Gabungkan kedua sensor
    paired = pd.merge(merged1, merged2, on=time_col, suffixes=('', '_drop'))
    
    # Pilih datetime dan ts_millis dari sensor 1 jika ada, fallback ke sensor 2
    paired['datetime'] = paired['datetime1'].fillna(paired['datetime2'])
    paired['ts_millis'] = paired['ts_millis1'].fillna(paired['ts_millis2'])
    paired.drop(['datetime1', 'datetime2', 'ts_millis1', 'ts_millis2'], axis=1, inplace=True)
    
    # Hapus baris yang tidak memiliki datetime sama sekali
    paired = paired.dropna(subset=['datetime'])
    
    return paired

def merge_with_bme(paired_imu, bme_df, time_col='epoch_ms', tolerance_ms=500):
    """
    Menggabungkan data tekanan dan altitude dari BME ke setiap baris IMU
    berdasarkan waktu terdekat.
    """
    # Pastikan kolom numerik di BME
    for col in ['pressure_hpa', 'altitude_m']:
        if col in bme_df.columns:
            bme_df[col] = pd.to_numeric(bme_df[col], errors='coerce')
    
    bme_sub = bme_df[[time_col, 'pressure_hpa', 'altitude_m']].dropna(subset=[time_col]).copy()
    
    # Jika BME kosong, beri nilai NA dan return
    if bme_sub.empty:
        paired_imu['pressure_hpa'] = pd.NA
        paired_imu['altitude_m'] = pd.NA
        return paired_imu
    
    # === PERBAIKAN: samakan tipe data kolom kunci ===
    # Konversi ke float (mampu menampung NaN) dan buang baris dengan NaN
    paired_imu[time_col] = pd.to_numeric(paired_imu[time_col], errors='coerce').astype(float)
    bme_sub[time_col] = pd.to_numeric(bme_sub[time_col], errors='coerce').astype(float)
    
    paired_imu = paired_imu.dropna(subset=[time_col])
    bme_sub = bme_sub.dropna(subset=[time_col])
    # ==============================================
    
    result = pd.merge_asof(
        paired_imu.sort_values(time_col),
        bme_sub.sort_values(time_col),
        on=time_col,
        direction='nearest',
        tolerance=tolerance_ms
    )
    return result

def process_subject(bmp_file, imu_file, subject_id, nama, output_dir,
                    imu_tolerance_ms=50, bme_tolerance_ms=500):
    """
    Memproses satu subjek: membaca file bmp dan imu, menggabungkan,
    lalu menyimpan ke CSV.
    """
    # Baca file dengan toleransi baris rusak
    imu_df = safe_read_csv(imu_file)
    bme_df = safe_read_csv(bmp_file)
    
    if imu_df.empty:
        print(f"Peringatan: File IMU {imu_file} kosong atau tidak terbaca. Subjek {nama} dilewati.")
        return
    if bme_df.empty:
        print(f"Peringatan: File BME {bmp_file} kosong atau tidak terbaca. Subjek {nama} dilewati.")
        return
    
    # Bersihkan dataframe dari baris kosong/rusak
    imu_df = clean_dataframe(imu_df, required_cols=['epoch_ms'])
    bme_df = clean_dataframe(bme_df, required_cols=['epoch_ms'])
    
    if imu_df.empty:
        print(f"Tidak ada data IMU valid untuk {nama}. Subjek dilewati.")
        return
    if bme_df.empty:
        print(f"Tidak ada data BME valid untuk {nama}. Subjek dilewati.")
        return
    
    # Konversi datetime secara fleksibel (menerima format spasi atau 'T')
    # Gunakan format='mixed' untuk menangani berbagai format
    try:
        imu_df['datetime'] = pd.to_datetime(imu_df['datetime'], errors='coerce', format='mixed')
        bme_df['datetime'] = pd.to_datetime(bme_df['datetime'], errors='coerce', format='mixed')
    except Exception as e:
        print(f"Error konversi datetime untuk {nama}: {e}")
        # Fallback: coba ekstrak dengan regex atau abaikan
        imu_df['datetime'] = pd.to_datetime(imu_df['datetime'], errors='coerce')
        bme_df['datetime'] = pd.to_datetime(bme_df['datetime'], errors='coerce')
    
    # Hapus baris dengan datetime NaT
    imu_df = imu_df.dropna(subset=['datetime'])
    bme_df = bme_df.dropna(subset=['datetime'])
    
    if imu_df.empty or bme_df.empty:
        print(f"Data datetime tidak valid untuk {nama}. Subjek dilewati.")
        return
    
    # Konversi epoch_ms ke numerik
    imu_df['epoch_ms'] = pd.to_numeric(imu_df['epoch_ms'], errors='coerce')
    bme_df['epoch_ms'] = pd.to_numeric(bme_df['epoch_ms'], errors='coerce')
    imu_df = imu_df.dropna(subset=['epoch_ms'])
    bme_df = bme_df.dropna(subset=['epoch_ms'])
    
    if imu_df.empty:
        print(f"Tidak ada epoch_ms valid di IMU untuk {nama}. Subjek dilewati.")
        return
    
    # Pairing sensor IMU
    try:
        paired_imu = pair_imu_sensors(imu_df, time_col='epoch_ms', tolerance_ms=imu_tolerance_ms)
    except ValueError as e:
        print(f"Gagal pairing IMU untuk {nama}: {e}. Subjek dilewati.")
        return
    
    if paired_imu.empty:
        print(f"Hasil pairing IMU kosong untuk {nama}. Subjek dilewati.")
        return
    
    # Gabung dengan BME
    final = merge_with_bme(paired_imu, bme_df, time_col='epoch_ms', tolerance_ms=bme_tolerance_ms)
    
    # Tambahkan subject_id dan nama
    final['subject_id'] = subject_id
    final['nama'] = nama
    
    # Ekstrak date dan time dari datetime
    final['date'] = final['datetime'].dt.date
    final['time'] = final['datetime'].dt.time
    
    # Urutan kolom output
    columns_order = [
        'subject_id', 'nama', 'date', 'time', 'epoch_ms', 'ts_millis',
        'ax1', 'ay1', 'az1', 'gx1', 'gy1', 'gz1',
        'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2',
        'pressure_hpa', 'altitude_m'
    ]
    
    # Pastikan semua kolom ada (isi NaN jika tidak)
    for col in columns_order:
        if col not in final.columns:
            final[col] = pd.NA
    
    final = final[columns_order]
    
    # Simpan file per subjek
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"gabungan_sensor_{subject_id}.csv")
    final.to_csv(out_path, index=False)
    print(f"Tersimpan: {out_path} (baris: {len(final)})")
    return final

def process_all_files(input_dir, output_dir, start_id=14,
                      imu_tolerance_ms=50, bme_tolerance_ms=500):
    """
    Memindai folder input, mencari pasangan file bmp_* dan imu_*,
    memproses setiap subjek, dan menyimpan hasil dengan ID berurutan.
    """
    # Kumpulkan file bmp dan imu berdasarkan (tanggal, nama)
    bmp_dict = {}
    imu_dict = {}
    
    for fname in os.listdir(input_dir):
        if not fname.endswith('.csv'):
            continue
        full_path = os.path.join(input_dir, fname)
        sensor_type, date_str, name = extract_info_from_filename(fname)
        if sensor_type == 'bmp':
            bmp_dict[(date_str, name)] = full_path
        elif sensor_type == 'imu':
            imu_dict[(date_str, name)] = full_path
    
    # Cari pasangan yang lengkap
    common_keys = set(bmp_dict.keys()) & set(imu_dict.keys())
    if not common_keys:
        print("Tidak ditemukan pasangan file bmp dan imu yang cocok.")
        return
    
    current_id = start_id
    for key in sorted(common_keys):
        date_str, name = key
        print(f"\nMemproses: {name} (tanggal {date_str})")
        process_subject(
            bmp_dict[key], imu_dict[key],
            subject_id=current_id,
            nama=name,
            output_dir=output_dir,
            imu_tolerance_ms=imu_tolerance_ms,
            bme_tolerance_ms=bme_tolerance_ms
        )
        current_id += 1
    
    print(f"\nSelesai. Total subjek diproses: {current_id - start_id}")

# Contoh jika dijalankan langsung
if __name__ == "__main__":
    # Gunakan raw string (r'...') atau double backslash untuk path Windows
    process_all_files(
        input_dir=r'D:\Semester_7\Skripsian\bachelor-last-project\gabunganSensor\Fathur',
        output_dir=r'D:\Semester_7\Skripsian\bachelor-last-project\gabunganSensor',
        start_id=14
    )