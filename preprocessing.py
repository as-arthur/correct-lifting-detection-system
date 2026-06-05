"""
preprocessing.py
================
Pipeline preprocessing yang IDENTIK dengan notebook LOSO_of_skripsiModel.ipynb.

URUTAN PIPELINE (harus dipertahankan persis):
  1. Validasi & konversi input → 14 kolom sensor mentah
  2. add_derived_features()   → +16 fitur turunan  = 30 total
  3. add_rolling_features()   → +16 fitur rolling  = 46 total
  4. align_to_expected()      → pastikan urutan kolom konsisten
  5. Scaler.transform()       → via model_loader
  6. Model.predict()          → via model_loader

Jangan ubah logika di sini tanpa memperbarui notebook.
"""

import warnings
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

# 14 kolom sensor mentah — URUTAN INI WAJIB DIPERTAHANKAN
RAW_SENSOR_COLS = [
    'ax1', 'ay1', 'az1',
    'gx1', 'gy1', 'gz1',
    'ax2', 'ay2', 'az2',
    'gx2', 'gy2', 'gz2',
    'pressure_hpa', 'altitude_m',
]

# Kolom kunci untuk rolling (sama persis dengan notebook)
ROLLING_KEY_COLS = [
    'acc_mag1', 'acc_mag2',
    'gyro_mag1', 'gyro_mag2',
    'roll1', 'pitch1',
    'diff_roll', 'diff_pitch',
]

ROLLING_N = 3  # N=3 pada 5 Hz ≈ 0.6 detik delay (sama dengan notebook)


# ---------------------------------------------------------------------------
# Tahap 1 — Validasi & normalisasi input mentah
# ---------------------------------------------------------------------------

def validate_and_normalize_raw(data: dict) -> dict:
    """
    Pastikan semua 14 kolom sensor ada dan bertipe float.
    Kolom yang hilang diisi 0.0 (konsisten dengan fillna(0) di notebook).

    Parameters
    ----------
    data : dict
        Dict bebas dari request JSON.

    Returns
    -------
    dict
        Dict yang sudah lengkap dan bersih (14 key, semua float).

    Raises
    ------
    ValueError
        Jika nilai tidak dapat dikonversi ke float.
    """
    cleaned = {}
    errors = []

    for col in RAW_SENSOR_COLS:
        raw_val = data.get(col, 0.0)
        try:
            val = float(raw_val) if raw_val not in (None, '') else 0.0
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            cleaned[col] = val
        except (TypeError, ValueError):
            errors.append(f"Kolom '{col}' tidak dapat dikonversi ke float: {raw_val!r}")

    if errors:
        raise ValueError("; ".join(errors))

    return cleaned


# ---------------------------------------------------------------------------
# Tahap 2 — add_derived_features (SALIN PERSIS DARI NOTEBOOK)
# ---------------------------------------------------------------------------

def add_derived_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur turunan per baris — dihitung instan tanpa buffer.
    Menambah ~16 fitur dari 14 kolom sensor.

    Fungsi ini disalin PERSIS dari notebook; tidak ada modifikasi logika.
    """
    X = X_df.copy()

    # Pastikan semua kolom sensor numerik & bersih (sama dengan notebook)
    sensor_numeric_cols = [
        'ax1', 'ay1', 'az1', 'gx1', 'gy1', 'gz1',
        'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2',
        'pressure_hpa', 'altitude_m',
    ]
    for col in sensor_numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(0)
            X[col] = X[col].replace([np.inf, -np.inf], 0)

    # ── Magnitudo akselerometer (intensitas gerakan keseluruhan) ──────
    if all(c in X.columns for c in ['ax1', 'ay1', 'az1']):
        X['acc_mag1'] = np.sqrt(X['ax1']**2 + X['ay1']**2 + X['az1']**2)
    if all(c in X.columns for c in ['ax2', 'ay2', 'az2']):
        X['acc_mag2'] = np.sqrt(X['ax2']**2 + X['ay2']**2 + X['az2']**2)

    # ── Magnitudo gyroscope (kecepatan rotasi) ────────────────────────
    if all(c in X.columns for c in ['gx1', 'gy1', 'gz1']):
        X['gyro_mag1'] = np.sqrt(X['gx1']**2 + X['gy1']**2 + X['gz1']**2)
    if all(c in X.columns for c in ['gx2', 'gy2', 'gz2']):
        X['gyro_mag2'] = np.sqrt(X['gx2']**2 + X['gy2']**2 + X['gz2']**2)

    # ── Selisih antar sensor (perbedaan gerak antara dua titik tubuh) ──
    for axis in ['x', 'y', 'z']:
        for s in ['a', 'g']:
            c1, c2 = f'{s}{axis}1', f'{s}{axis}2'
            if c1 in X.columns and c2 in X.columns:
                X[f'diff_{s}{axis}'] = X[c1] - X[c2]

    # ── Sudut kemiringan / tilt (sangat relevan untuk postur) ─────────
    if all(c in X.columns for c in ['ax1', 'ay1', 'az1']):
        X['roll1']  = np.arctan2(
            X['ay1'],
            np.sqrt(X['ax1']**2 + X['az1']**2) + 1e-9
        )
        X['pitch1'] = np.arctan2(
            -X['ax1'],
            np.sqrt(X['ay1']**2 + X['az1']**2) + 1e-9
        )
    if all(c in X.columns for c in ['ax2', 'ay2', 'az2']):
        X['roll2']  = np.arctan2(
            X['ay2'],
            np.sqrt(X['ax2']**2 + X['az2']**2) + 1e-9
        )
        X['pitch2'] = np.arctan2(
            -X['ax2'],
            np.sqrt(X['ay2']**2 + X['az2']**2) + 1e-9
        )

    # ── Beda sudut antar sensor (indikator ketidakselarasan postur) ───
    if all(c in X.columns for c in ['roll1', 'roll2']):
        X['diff_roll']  = X['roll1'] - X['roll2']
        X['diff_pitch'] = X['pitch1'] - X['pitch2']

    return X


# ---------------------------------------------------------------------------
# Tahap 3 — add_rolling_features (SALIN PERSIS DARI NOTEBOOK)
# ---------------------------------------------------------------------------

def add_rolling_features(X_df: pd.DataFrame, n: int = ROLLING_N) -> pd.DataFrame:
    """
    Rolling pendek HANYA pada fitur kunci.
    N=3 di 5 Hz = 0.6 detik delay.
    Hanya 16 fitur tambahan (8 kolom × 2 statistik).

    Fungsi ini disalin PERSIS dari notebook; tidak ada modifikasi logika.
    """
    X = X_df.copy()

    # Pilih hanya fitur yang paling relevan untuk rolling (sama dengan notebook)
    key_cols = [c for c in ROLLING_KEY_COLS if c in X.columns]

    for col in key_cols:
        # Pastikan kolom numerik sebelum operasi rolling (sama dengan notebook)
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X[f'{col}_rm{n}'] = (
            X[col].rolling(n, min_periods=1).mean()
        )
        X[f'{col}_rs{n}'] = (
            X[col].rolling(n, min_periods=1).std().fillna(0)
        )

    return X


def add_rolling_per_subject(
    X_df: pd.DataFrame,
    y_df: pd.Series,
    groups_arr: np.ndarray,
    n: int = ROLLING_N,
) -> tuple:
    """
    Rolling dihitung ulang per subjek agar tidak melewati batas.
    Disalin PERSIS dari notebook (digunakan saat training/evaluasi).
    """
    parts_X, parts_y = [], []
    for subj in np.unique(groups_arr):
        mask = groups_arr == subj
        X_s = X_df[mask].copy()
        X_s = add_rolling_features(X_s, n=n)
        parts_X.append(X_s)
        parts_y.append(y_df[mask])
    return (
        pd.concat(parts_X).reset_index(drop=True),
        pd.concat(parts_y).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Tahap 4 — Alignment fitur
# ---------------------------------------------------------------------------

def align_to_expected(
    df: pd.DataFrame,
    expected_features: list,
) -> pd.DataFrame:
    """
    Pastikan DataFrame memiliki kolom yang tepat dan dalam urutan yang benar.
    Kolom yang hilang diisi 0, kolom ekstra dibuang.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame setelah add_derived + add_rolling.
    expected_features : list
        Daftar nama fitur dari feature_names.json.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom persis sesuai expected_features.
    """
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        warnings.warn(
            f"[alignment] {len(missing)} kolom tidak ditemukan, diisi 0: {missing}",
            RuntimeWarning,
            stacklevel=2,
        )
        for col in missing:
            df[col] = 0.0

    # Pilih dan urutkan kolom sesuai expected_features
    return df[expected_features].copy()


# ---------------------------------------------------------------------------
# Pipeline utama — inference single sample (dengan rolling buffer)
# ---------------------------------------------------------------------------

class InferencePipeline:
    """
    Pipeline lengkap untuk inferensi satu sampel.

    Menggunakan rolling buffer (deque maxlen=3) per session_id agar
    fitur rolling konsisten dengan perilaku saat training.

    Penggunaan
    ----------
    pipeline = InferencePipeline(model, scaler, expected_features)
    result   = pipeline.predict(raw_dict, session_id='user_01')
    """

    LABEL_MAP = {0: 'salah', 1: 'benar'}
    CONFIDENCE_MIN = 0.0  # threshold minimum; set di endpoint jika perlu

    def __init__(self, model, scaler, expected_features: list, rolling_n: int = ROLLING_N):
        self.model             = model
        self.scaler            = scaler
        self.expected_features = expected_features
        self.rolling_n         = rolling_n
        self._buffers: dict    = {}   # session_id → list of raw rows

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        raw_dict: dict,
        session_id: str = 'default',
    ) -> dict:
        """
        Jalankan pipeline lengkap untuk satu sampel.

        Parameters
        ----------
        raw_dict : dict
            Dict dengan 14 kolom sensor mentah.
        session_id : str
            ID sesi untuk mengelola rolling buffer terpisah.

        Returns
        -------
        dict
            {
              'prediction': int (0 atau 1),
              'label': str ('salah' atau 'benar'),
              'probabilities': {'salah': float, 'benar': float},
              'feature_count': int,
            }
        """
        # 1. Validasi & normalisasi
        cleaned = validate_and_normalize_raw(raw_dict)

        # 2. Update rolling buffer untuk sesi ini
        buf = self._get_buffer(session_id)
        buf.append(cleaned)

        # 3. Bangun DataFrame dari buffer (max ROLLING_N baris)
        df_buf = pd.DataFrame(list(buf))

        # 4. add_derived_features (identik dengan notebook)
        df_derived = add_derived_features(df_buf)

        # 5. add_rolling_features (identik dengan notebook)
        df_rolled = add_rolling_features(df_derived, n=self.rolling_n)
        df_rolled = df_rolled.replace([np.inf, -np.inf], 0).fillna(0)

        # 6. Ambil baris terakhir (prediksi saat ini)
        df_current = df_rolled.iloc[[-1]].copy()

        # 7. Align ke expected features
        df_aligned = align_to_expected(df_current, self.expected_features)

        # 8. Scale
        X_scaled = self.scaler.transform(df_aligned)

        # 9. Predict
        pred_class  = int(self.model.predict(X_scaled)[0])
        probas      = self.model.predict_proba(X_scaled)[0]
        classes     = list(self.model.classes_)
        prob_salah  = float(probas[classes.index(0)]) if 0 in classes else 0.0
        prob_benar  = float(probas[classes.index(1)]) if 1 in classes else 0.0

        return {
            'prediction'   : pred_class,
            'label'        : self.LABEL_MAP.get(pred_class, 'unknown'),
            'probabilities': {'salah': round(prob_salah, 4), 'benar': round(prob_benar, 4)},
            'feature_count': len(self.expected_features),
        }

    def reset_session(self, session_id: str = 'default') -> None:
        """Hapus rolling buffer untuk sesi tertentu."""
        self._buffers.pop(session_id, None)

    def reset_all_sessions(self) -> None:
        """Hapus semua rolling buffer."""
        self._buffers.clear()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_buffer(self, session_id: str) -> list:
        """Ambil atau buat rolling buffer untuk sesi ini."""
        if session_id not in self._buffers:
            self._buffers[session_id] = []
        buf = self._buffers[session_id]
        # Pertahankan hanya rolling_n baris terakhir
        if len(buf) >= self.rolling_n:
            self._buffers[session_id] = buf[-(self.rolling_n - 1):]
        return self._buffers[session_id]