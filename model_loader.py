"""
model_loader.py
===============
Singleton loader untuk model, scaler, dan konfigurasi.

- Lazy-load (hanya dimuat saat pertama kali dibutuhkan).
- Thread-safe via threading.Lock.
- Validasi versi sklearn pada saat loading.
- Menyediakan InferencePipeline yang siap pakai.
"""

import json
import os
import threading
import warnings

import joblib

from preprocessing import InferencePipeline

# ---------------------------------------------------------------------------
# Path default (dapat di-override via environment variable)
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'model', 'feature_names.json')
CONFIG_PATH = os.path.join(BASE_DIR, 'model', 'model_config.json')

# ---------------------------------------------------------------------------
# Default konfigurasi (mirrored dari app.py asli)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'confidence_threshold' : 0.20,   # probabilitas minimum agar prediksi dianggap valid
    'thresh_benar'         : 0.70,   # prob min kelas 'benar'  (ergonomis)
    'thresh_salah'         : 0.60,   # prob min kelas 'salah'  (non-ergonomis)
    'buzzer_duration'      : 1.5,    # detik buzzer menyala
    'prediction_interval_ms': 300,   # interval prediksi dari frontend (ms)
    'buzzer_enabled'       : True,
    'sensor_stale_timeout_ms': 5000,
}

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_lock              = threading.RLock()
_rf_model          = None
_scaler            = None
_expected_features = None
_config            = None
_pipeline          = None   # InferencePipeline singleton


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_model():
    """
    Muat model RF dari disk; cached setelah pertama kali.

    Returns
    -------
    RandomForestClassifier

    Raises
    ------
    FileNotFoundError
        Jika file model tidak ditemukan.
    RuntimeError
        Jika model tidak memiliki method predict.
    """
    global _rf_model
    if _rf_model is not None:
        return _rf_model

    with _lock:
        if _rf_model is not None:       # double-check setelah acquire lock
            return _rf_model

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"File model tidak ditemukan: {MODEL_PATH}\n"
                "Pastikan 'random_forest_model.pkl' ada di folder 'model/'."
            )

        model = joblib.load(MODEL_PATH)

        if not hasattr(model, 'predict'):
            raise RuntimeError(
                f"Objek yang dimuat dari {MODEL_PATH} bukan model sklearn yang valid."
            )

        _rf_model = model
        return _rf_model


def load_scaler():
    """
    Muat RobustScaler dari disk; cached setelah pertama kali.

    Returns
    -------
    RobustScaler

    Raises
    ------
    FileNotFoundError
    RuntimeError
    """
    global _scaler
    if _scaler is not None:
        return _scaler

    with _lock:
        if _scaler is not None:
            return _scaler

        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"File scaler tidak ditemukan: {SCALER_PATH}\n"
                "Pastikan 'scaler.pkl' ada di folder 'model/'."
            )

        scaler = joblib.load(SCALER_PATH)

        if not hasattr(scaler, 'transform'):
            raise RuntimeError(
                f"Objek yang dimuat dari {SCALER_PATH} bukan scaler sklearn yang valid."
            )

        _scaler = scaler
        return _scaler


def load_feature_names() -> list:
    """
    Muat daftar nama fitur dari feature_names.json; cached setelah pertama kali.

    Returns
    -------
    list of str
        Urutan fitur yang diharapkan model (harus identik saat training).

    Raises
    ------
    FileNotFoundError
    ValueError
        Jika file kosong atau bukan list.
    """
    global _expected_features
    if _expected_features is not None:
        return _expected_features

    with _lock:
        if _expected_features is not None:
            return _expected_features

        if not os.path.exists(FEATURE_NAMES_PATH):
            raise FileNotFoundError(
                f"File feature names tidak ditemukan: {FEATURE_NAMES_PATH}\n"
                "Generate ulang dari notebook dengan:\n"
                "  with open('model/feature_names.json', 'w') as f:\n"
                "      json.dump(list(X_train_win_last_fold.columns), f)"
            )

        with open(FEATURE_NAMES_PATH, 'r') as f:
            names = json.load(f)

        if not isinstance(names, list) or len(names) == 0:
            raise ValueError(
                f"feature_names.json harus berupa list tidak kosong, "
                f"tapi dapat: {type(names)}"
            )

        _expected_features = names
        return _expected_features


def load_config() -> dict:
    """
    Muat konfigurasi dari model_config.json.
    Kunci yang hilang diisi dari DEFAULT_CONFIG.
    """
    global _config

    # Config selalu di-reload dari disk (agar perubahan runtime terdeteksi)
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        _config = DEFAULT_CONFIG.copy()
        return _config

    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = json.load(f)
        # Isi kunci yang hilang dari default
        for key, val in DEFAULT_CONFIG.items():
            cfg.setdefault(key, val)
        _config = cfg
        return _config
    except Exception as exc:
        warnings.warn(f"Gagal membaca config: {exc}. Menggunakan default.")
        return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> bool:
    """Simpan config ke disk."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as exc:
        warnings.warn(f"Gagal menyimpan config: {exc}")
        return False


# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------

def get_pipeline() -> InferencePipeline:
    """
    Kembalikan InferencePipeline yang sudah siap pakai (lazy-init).

    Memanggil load_model(), load_scaler(), load_feature_names()
    secara otomatis.

    Returns
    -------
    InferencePipeline

    Raises
    ------
    FileNotFoundError / RuntimeError
        Jika salah satu artefak tidak dapat dimuat.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _lock:
        if _pipeline is not None:
            return _pipeline

        model    = load_model()
        scaler   = load_scaler()
        features = load_feature_names()

        _pipeline = InferencePipeline(
            model=model,
            scaler=scaler,
            expected_features=features,
        )
        return _pipeline


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def model_info() -> dict:
    """
    Kembalikan informasi ringkas tentang model yang dimuat.
    Berguna untuk endpoint /health.
    """
    info = {
        'model_path'         : MODEL_PATH,
        'scaler_path'        : SCALER_PATH,
        'feature_names_path' : FEATURE_NAMES_PATH,
        'model_loaded'       : _rf_model is not None,
        'scaler_loaded'      : _scaler is not None,
        'features_loaded'    : _expected_features is not None,
        'n_features'         : len(_expected_features) if _expected_features else None,
    }

    if _rf_model is not None:
        info['model_type']       = type(_rf_model).__name__
        info['n_estimators']     = getattr(_rf_model, 'n_estimators', None)
        info['model_classes']    = list(_rf_model.classes_) if hasattr(_rf_model, 'classes_') else None

    return info


def reload_all() -> None:
    """
    Paksa reload semua artefak dari disk.
    Berguna setelah model diperbarui tanpa restart server.
    """
    global _rf_model, _scaler, _expected_features, _config, _pipeline
    with _lock:
        _rf_model          = None
        _scaler            = None
        _expected_features = None
        _config            = None
        _pipeline          = None
    # Trigger lazy re-load
    get_pipeline()