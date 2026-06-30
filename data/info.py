import pandas as pd
from pathlib import Path

# =====================================================
# KONFIGURASI
# =====================================================
DATA_FOLDER = r"D:\Semester_7\Skripsian\bachelor-last-project\data"   # ganti sesuai folder Anda

# =====================================================
# FUNGSI MEMBACA FILE
# =====================================================
def load_csv_files(folder):
    folder = Path(folder)

    bmp_files = []
    imu_files = []

    for file in folder.rglob("*.csv"):
        try:
            df = pd.read_csv(file, nrows=5)

            cols = set(df.columns)

            # BMP
            if {"temperature_c", "pressure_hpa", "altitude_m"}.issubset(cols):
                bmp_files.append(file)

            # IMU
            elif {"ax", "ay", "az", "gx", "gy", "gz"}.issubset(cols):
                imu_files.append(file)

        except Exception as e:
            print(f"Gagal membaca {file}: {e}")

    return bmp_files, imu_files


# =====================================================
# GABUNG DATA
# =====================================================
def merge_files(file_list):
    dfs = []

    for f in file_list:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            dfs.append(df)

        except Exception as e:
            print(f"Error {f}: {e}")

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# =====================================================
# INFORMASI DATASET
# =====================================================
def dataset_info(df, name):

    print("\n" + "=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)

    print(f"Jumlah baris  : {len(df):,}")
    print(f"Jumlah kolom  : {len(df.columns)}")

    print("\nKolom:")
    for c in df.columns:
        print(f"  - {c}")

    # datetime
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")

        print("\nRentang waktu:")
        print("  Awal :", dt.min())
        print("  Akhir:", dt.max())

        durasi = dt.max() - dt.min()
        print("  Durasi:", durasi)

    # missing values
    print("\nMissing values:")
    missing = df.isna().sum()

    for col, val in missing.items():
        if val > 0:
            print(f"  {col}: {val}")

    if missing.sum() == 0:
        print("  Tidak ada missing value")

    # statistik numerik
    num_cols = df.select_dtypes(include="number")

    if len(num_cols.columns) > 0:
        print("\nStatistik numerik:")
        print(num_cols.describe().round(3))

    # distribusi sensor
    if "sensor" in df.columns:
        print("\nDistribusi sensor:")
        print(df["sensor"].value_counts())

    if "mpu_id" in df.columns:
        print("\nDistribusi MPU:")
        print(df["mpu_id"].value_counts())


# =====================================================
# MAIN
# =====================================================
def main():

    bmp_files, imu_files = load_csv_files(DATA_FOLDER)

    print(f"File BMP ditemukan : {len(bmp_files)}")
    print(f"File IMU ditemukan : {len(imu_files)}")

    bmp_df = merge_files(bmp_files)
    imu_df = merge_files(imu_files)

    # simpan hasil gabungan
    if not bmp_df.empty:
        bmp_df.to_csv("merged_bmp.csv", index=False)

    if not imu_df.empty:
        imu_df.to_csv("merged_imu.csv", index=False)

    # informasi dataset
    if not bmp_df.empty:
        dataset_info(bmp_df, "BMP")

    if not imu_df.empty:
        dataset_info(imu_df, "IMU")

    # total keseluruhan
    print("\n" + "=" * 60)
    print("RINGKASAN")
    print("=" * 60)

    print(f"Total baris BMP : {len(bmp_df):,}")
    print(f"Total baris IMU : {len(imu_df):,}")
    print(f"Total baris     : {len(bmp_df)+len(imu_df):,}")


if __name__ == "__main__":
    main()