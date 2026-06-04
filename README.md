# Correct Lifting Detection System

Ini adalah tugas akhir untuk memenuhi syarat akademik di Universitas Malikussaleh.  
Sistem ini menggunakan sensor MPU6050 (2 buah) dan BMP280 (1 buah) yang terhubung ke dua modul ESP8266 untuk mendeteksi postur pengangkatan beban secara real-time. Data dikirim ke server Flask melalui HTTP POST, disimpan dalam file CSV, dan ditampilkan pada dashboard web.

## Daftar Bahan

1. 2 x Sensor MPU6050 (akselerometer + giroskop)
2. 1 x Sensor BMP280 (tekanan, suhu, ketinggian)
3. 2 x Modul ESP8266 (NodeMCU v3)
4. Kabel jumper untuk koneksi I2C dan power
5. 2 x Kabel MicroUSB
6. Sumber daya (power bank atau adaptor 5V)

## Skema Koneksi

Setiap ESP8266 dihubungkan dengan satu MPU6050 dan satu BMP280 melalui jalur I2C.  
Konfigurasi pin yang digunakan:

| NodeMCU | Perangkat | Pin |
|---------|-----------|-----|
| ESP1 (MPU1) | MPU6050 | SDA → D2 (GPIO4), SCL → D1 (GPIO5) |
| ESP1 | BMP280 | SDA → D2 (GPIO4), SCL → D1 (GPIO5) |
| ESP2 (MPU2) | MPU6050 | SDA → D2 (GPIO4), SCL → D1 (GPIO5) |

Setiap sensor MPU6050 menggunakan alamat I2C yang berbeda:

- MPU pertama: `0x68`
- MPU kedua: `0x69`

Sensor BMP280 menggunakan alamat `0x76` (atau `0x77` jika modul berbeda).

Skema posisi pemasangan sensor pada tubuh operator:

<img width="454" height="264" alt="Sketsa_postur" src="https://github.com/user-attachments/assets/6e83e356-5503-4bc0-af5c-ef49ab00f44c" />

## Instalasi dan Konfigurasi

### 1. Persiapan Software

- Install Arduino IDE (atau PlatformIO) untuk mengunggah kode ke ESP8266.
- Install library berikut melalui Library Manager:
  - Adafruit MPU6050
  - Adafruit BMP280
  - Adafruit Sensor
  - ESP8266WiFi
  - ESP8266HTTPClient
  - ESP8266WebServer
  - LittleFS

- Di komputer (laptop/server) install Python 3.8+ dan Flask:

```bash
pip install flask
```

### 2. Konfigurasi Kode pada ESP8266

- Buka file `sketch_mpu6050.ino` dan `sketch_bmp280.ino` (atau kode yang telah disediakan).

- Ubah parameter WiFi pada bagian `CONFIG`:

```cpp
#define WIFI_SSID "nama_wifi_anda"
#define WIFI_PASS "password_wifi"
```

- Ubah alamat IP server Flask (sesuaikan dengan IP laptop yang menjalankan Flask):

```cpp
#define SERVER_HOST "10.243.214.200"
#define SERVER_PORT 5000
```

- Pastikan pin I2C sesuai dengan skema di atas.

- Unggah kode ke masing-masing ESP8266 menggunakan kabel MicroUSB.

### 3. Menjalankan Server Flask

- Di laptop, buka terminal pada folder proyek yang berisi file `app.py`.

- Jalankan perintah:

```bash
python app.py
```

- Server akan berjalan pada:
  - `http://0.0.0.0:5000`
  - `http://localhost:5000`

- Pastikan laptop dan ESP8266 terhubung ke jaringan WiFi yang sama.

## Fitur Sistem

- **Pengiriman data real-time**  
  ESP8266 mengirim data akselerasi, giroskop, suhu, tekanan, dan ketinggian setiap 500 ms (atau sesuai interval yang diatur).

- **Penyimpanan lokal (LittleFS)**  
  Data juga disimpan dalam file CSV di ESP8266 (opsional).

- **Server Flask**  
  Menerima data melalui endpoint `/data`, lalu menyimpannya ke file CSV harian berdasarkan ID subjek.

- **Dashboard Web**  
  Halaman web menggunakan Jinja2 dan Chart.js untuk menampilkan:
  - Grafik magnitude akselerasi dan giroskop (200 data terakhir)
  - Tabel 10 data terbaru per sensor
  - Indikator koneksi
  - Sampling Rate (Hz)
  - Data Rate (sampel/detik)
  - Tombol untuk memuat data subjek berbeda dan mengunduh CSV

## Cara Penggunaan

1. Nyalakan kedua ESP8266 yang sudah terhubung ke sensor.

2. Jalankan server Flask di laptop.

3. Buka browser dan akses alamat:

```text
http://<ip_laptop>:5000
```

4. Pada halaman web, isi nomor subjek (misalnya `05`) pada kolom **File ID**, lalu klik **Load Data**.

5. Amati grafik dan tabel yang menunjukkan data MPU1, MPU2, dan BMP280.

6. Untuk mengunduh data subjek tertentu:
   - Isi **No. Subjek**
   - Isi **Nama Subjek**
   - Klik **Download CSV**

7. Untuk menghentikan sistem, tutup terminal Flask atau tekan:

```text
Ctrl + C
```

##  Struktur Folder

```text
correct-lifting-detection-system/
├── MPU6050/                     # Kode untuk ESP8266 pertama (MPU6050 + BMP280)
│   └── MPU6050.ino
├── bmp280/                      # Kode untuk ESP8266 kedua (BMP280 saja)
│   └── bmp280.ino
├── dataUser/                    # 📥 Data mentah dari sensor (CSV)
│   ├── imu_*.csv                # Data MPU6050 (akselerasi & giroskop)
│   └── bmp_*.csv                # Data BMP280 (suhu, tekanan, ketinggian)
├── gabunganSensor/              # 🔗 Data hasil sinkronisasi dan penggabungan sensor
├── dataLabel/                   # 🏷️ Data berlabel siap untuk pelatihan model
├── model/                       # 🤖 Model machine learning yang sudah dilatih
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   └── model_config.json
├── templates/                   # 🎨 Template HTML untuk dashboard web
│   ├── index.html
│   ├── dashboard.html
│   ├── latest.html
│   └── settings.html
├── static/                      # 🎨 File CSS untuk styling
│   └── style.css
├── data/                        # 📊 Data CSV tambahan untuk pengujian
│   └── bmp_*.csv / imu_*.csv
├── app.py                       # 🚀 Aplikasi utama server Flask
├── dataCombine.py               # 🔄 Script untuk menggabungkan data antar sensor
├── README.md
├── .gitignore
├── Skripsi_Fathur.docx          # 📄 Dokumentasi skripsi dalam Word
├── Skripsi_Fathur.pdf           # 📄 Dokumentasi skripsi dalam PDF
└── catatan.txt                  # Catatan atau dokumentasi singkat
```

---

##  Penjelasan Folder Data

### `dataUser/`

Data mentah yang dikirim dari ESP8266 ke server Flask.
Disimpan dalam file CSV dengan format:

* `imu_*.csv` → Data MPU6050 (akselerasi & giroskop)
* `bmp_*.csv` → Data BMP280 (suhu, tekanan, dan ketinggian)

### `gabunganSensor/`

Berisi hasil penggabungan data dari tiga sensor:

* MPU1
* MPU2
* BMP280

Data telah disinkronkan berdasarkan waktu (`timestamp`) sehingga siap digunakan untuk analisis lanjutan.

### `dataLabel/`

Berisi data dari folder `gabunganSensor/` yang telah diberi label aktivitas atau kategori tertentu untuk kebutuhan:

* Pelatihan model machine learning
* Evaluasi model
* Prediksi gerakan lifting


## Struktur Output Data

File CSV untuk sensor MPU (disimpan di server) memiliki format nama:

```text
imu_<id>_<tanggal>.csv
```

Contoh isi file:

```csv
datetime,epoch_ms,ts_millis,mpu_id,ax,ay,az,gx,gy,gz
2026-05-10 10:12:34.567,1234567890,1234567,1,-0.04468,-0.88281,0.41772,-1.290,0.542,0.618
...
```

File CSV untuk sensor BMP280 disimpan dengan format nama:

```text
bmp_<id>_<tanggal>.csv
```

Contoh isi file:

```csv
datetime,epoch_ms,ts_millis,sensor,temperature_c,pressure_hpa,altitude_m
2026-05-10 10:12:35.123,1234567891,1234568,3,28.45,1010.23,45.67
...
```

## Struktur Folder dan Alur Data

Berikut adalah alur pengolahan data dari sensor hingga siap digunakan untuk pelatihan model:

*   **`dataUser/`** : **Data Mentah (Raw Data)**. Folder ini berisi data langsung dari sensor MPU6050 (akselerasi & giroskop) dan BMP280 (tekanan, suhu, ketinggian) yang dikirimkan oleh ESP8266 ke server Flask. Data disimpan dalam file CSV per subjek dan per hari, misalnya `imu_<subjek>_<tanggal>.csv` dan `bmp_<subjek>_<tanggal>.csv`[reference:0]. Ini adalah titik awal sebelum dilakukan pemrosesan lebih lanjut.

*   **`gabunganSensor/`** : **Data Gabungan Antar Sensor**. Pada tahap ini, data dari tiga sensor (MPU1, MPU2, dan BMP280) yang memiliki stempel waktu (`timestamp`) berbeda akan disinkronkan dan digabungkan menjadi satu file. Proses ini penting untuk memastikan bahwa data dari semua sensor merepresentasikan momen yang sama saat seorang operator mengangkat beban, sehingga analisis postur dapat dilakukan secara akurat.

*   **`dataLabel/`** : **Data Berlabel**. Ini adalah data dari folder `gabunganSensor/` yang telah melalui proses pelabelan. Pelabelan bisa berupa kategori seperti `angkat benar` atau `angkat salah`, berdasarkan pada nilai sensor atau validasi manual. Data dalam folder ini sudah siap untuk digunakan sebagai *dataset* dalam proses pelatihan model *machine learning* (misalnya untuk deteksi postur secara otomatis).

## Struktur Output Data (di dalam folder `dataUser`)

File CSV untuk sensor MPU (disimpan di server) memiliki format nama: `imu_<subjek>_<tanggal>.csv`. Contoh isi file:
... (contoh isi)
File CSV untuk sensor BMP280 disimpan dengan format nama: `bmp_<subjek>_<tanggal>.csv`. Contoh isi file:
... (contoh isi)

Data mentah di folder `dataUser` ini kemudian akan diproses ke tahap berikutnya (penggabungan dan pelabelan) yang dijelaskan pada bagian **Struktur Folder dan Alur Data**.

## Catatan

- Pastikan tegangan yang diberikan ke sensor adalah `3.3V` (jangan `5V`) untuk mencegah kerusakan.

- Jika sensor tidak terdeteksi, aktifkan konfigurasi berikut pada kode ESP8266 untuk memeriksa alamat I2C:

```cpp
#define I2C_SCAN_ON_BOOT true
```

- Untuk mencapai frekuensi sampling yang stabil (20–50 Hz), disarankan:
  - Menonaktifkan penyimpanan LittleFS:

```cpp
SAVE_TO_LITTLEFS false
```

  - Atau menggunakan metode pengiriman batch data.

## Kontribusi

Proyek ini dikembangkan secara individual sebagai tugas akhir.  
Saran dan perbaikan dapat disampaikan melalui *issue* pada repositori.
