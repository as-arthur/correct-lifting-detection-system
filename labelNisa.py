import pandas as pd
from datetime import datetime

# ========== KONFIGURASI ==========
input_file = "DataNisaDina\\imu_2026-05-21(kelvin).csv"
output_file = "DataNisaDina\\dataNisaLabel\\kelvin.csv"

start_datetime = "2026-05-21 17:49:00"   # awal rentang waktu
end_datetime   = "2026-05-21 17:54:00"   # akhir rentang waktu

# Informasi subjek (sesuaikan)
subject_id = "SUBJ001"
nama = "Kelvin"
# =================================

def main():
    # 1. Baca file CSV
    df = pd.read_csv(input_file)
    
    # 2. Konversi kolom datetime ke tipe datetime Pandas
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 3. Filter data berdasarkan rentang waktu
    start = pd.to_datetime(start_datetime)
    end   = pd.to_datetime(end_datetime)
    mask = (df['datetime'] >= start) & (df['datetime'] <= end)
    df_filtered = df.loc[mask].copy()
    
    if df_filtered.empty:
        print("Tidak ada data dalam rentang waktu yang ditentukan.")
        return
    
    # 4. Pisahkan data berdasarkan mpu_id
    df_1 = df_filtered[df_filtered['mpu_id'] == 1].copy()
    df_2 = df_filtered[df_filtered['mpu_id'] == 2].copy()
    
    # 5. Gabungkan berdasarkan datetime (inner join)
    #    Gunakan kolom datetime sebagai kunci penggabungan
    df_merged = pd.merge(df_1, df_2, on='datetime', suffixes=('1', '2'))
    
    # 6. Pilih kolom yang diperlukan dan beri nama sesuai format
    df_merged['date'] = df_merged['datetime'].dt.date
    df_merged['time'] = df_merged['datetime'].dt.time
    
    # Kolom yang dipertahankan: epoch_ms1, ts_millis1, ax1, ay1, az1, gx1, gy1, gz1,
    # dan dari sensor2: ax2, ay2, az2, gx2, gy2, gz2 (epoch_ms dan ts_millis seharusnya sama, kita pakai dari sensor1)
    df_final = pd.DataFrame({
        'subject_id': subject_id,
        'nama': nama,
        'date': df_merged['date'],
        'time': df_merged['time'],
        'epoch_ms': df_merged['epoch_ms1'],
        'ts_millis': df_merged['ts_millis1'],
        'ax1': df_merged['ax1'],
        'ay1': df_merged['ay1'],
        'az1': df_merged['az1'],
        'gx1': df_merged['gx1'],
        'gy1': df_merged['gy1'],
        'gz1': df_merged['gz1'],
        'ax2': df_merged['ax2'],
        'ay2': df_merged['ay2'],
        'az2': df_merged['az2'],
        'gx2': df_merged['gx2'],
        'gy2': df_merged['gy2'],
        'gz2': df_merged['gz2']
    })
    
    # 7. Hitung label setiap 10 detik (berdasarkan datetime gabungan)
    interval_num = ((df_merged['datetime'] - start).dt.total_seconds() // 10).astype(int)
    df_final['label'] = interval_num.apply(lambda x: 'ergonomis' if x % 2 == 0 else 'non-ergonomis')
    
    # 8. Simpan ke file output
    df_final.to_csv(output_file, index=False)
    print(f"Berhasil! Data telah diformat ulang dan dilabeli, disimpan ke '{output_file}'")
    print(f"Jumlah baris (setiap baris = pasangan sensor 1 & 2): {len(df_final)}")
    print("\nContoh 5 baris pertama (kolom terpilih):")
    print(df_final[['subject_id', 'nama', 'date', 'time', 'label']].head())

if __name__ == "__main__":
    main()