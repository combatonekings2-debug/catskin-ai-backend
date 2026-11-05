"""
CNN Service untuk CatSkin AI
Menangani preprocessing gambar dan prediksi dengan model MobileNetV2
"""

import numpy as np
from PIL import Image
import os
import json

class CNNService:
    """
    Service untuk prediksi penyakit kulit kucing.

    MODE:
    - MOCK : Random prediction (demo, sebelum model dilatih)
    - REAL : Prediksi dengan model MobileNetV2 yang sudah dilatih
    """

    # Default class names — sesuai output class_indices di NAHAWAND.ipynb Cell 10
    DEFAULT_CLASS_NAMES = [
        "Dermatitis",
        "Flea Allergy",
        "Healthy",
        "Ringworm",
        "Scabies",
    ]

    def __init__(self, model_path: str = "models/model_kulit_kucing_v2_final.keras"):
        self.model_path = model_path
        self.model      = None
        self.mode       = "MOCK"
        self.class_names = self._load_class_names()
        self._load_model()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _load_class_names(self) -> list:
        """
        Baca urutan kelas dari model_info.json yang dihasilkan notebook.
        Fallback ke DEFAULT_CLASS_NAMES jika file tidak ada.
        """
        info_path = os.path.join(os.path.dirname(self.model_path), "model_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
                names = info.get("class_names", self.DEFAULT_CLASS_NAMES)
                print(f"[OK] Class names dari model_info.json: {names}")
                return names
            except Exception as e:
                print(f"[WARN] Gagal baca model_info.json: {e}")
        return list(self.DEFAULT_CLASS_NAMES)

    def _load_model(self):
        """Load model .keras atau .h5 jika tersedia, fallback ke MOCK."""
        import zipfile
        import shutil

        # Cari path model yang valid
        target_path = self.model_path
        models_dir = os.path.dirname(target_path) or "models"

        # Cek apakah ada active_model.json
        active_json = os.path.join(models_dir, "active_model.json")
        if os.path.exists(active_json):
            try:
                with open(active_json, "r", encoding="utf-8") as f:
                    am = json.load(f).get("active_model")
                    if am and os.path.exists(os.path.join(models_dir, am)):
                        target_path = os.path.join(models_dir, am)
            except Exception:
                pass

        if not os.path.exists(target_path):
            # Cek file .keras terbaru di models_dir
            keras_files = [
                os.path.join(models_dir, f)
                for f in os.listdir(models_dir)
                if f.endswith(".keras") and not f.startswith(".")
            ]
            if keras_files:
                keras_files.sort(key=os.path.getmtime, reverse=True)
                target_path = keras_files[0]

        if not os.path.exists(target_path):
            print(f"[WARN] Model tidak ditemukan di: {target_path}")
            print("[INFO] Menggunakan MOCK mode")
            return

        try:
            import tensorflow as tf

            # Tangani jika file berformat ZIP (.keras) tapi dinamai .h5
            actual_load_path = target_path
            if target_path.endswith(".h5") and zipfile.is_zipfile(target_path):
                keras_alias = target_path.rsplit(".", 1)[0] + ".keras"
                if not os.path.exists(keras_alias):
                    shutil.copy2(target_path, keras_alias)
                actual_load_path = keras_alias

            # Patch kompatibilitas Keras 3.13 -> 3.12 (quantization_config)
            try:
                import keras
                _orig_dense_init = keras.layers.Dense.__init__
                def _patched_dense_init(self, *args, **kwargs):
                    kwargs.pop('quantization_config', None)
                    return _orig_dense_init(self, *args, **kwargs)
                keras.layers.Dense.__init__ = _patched_dense_init
            except Exception:
                pass

            self.model = tf.keras.models.load_model(actual_load_path)
            self.model_path = actual_load_path
            self.mode  = "REAL"
            print(f"[OK] Model AI Aktif berhasil dimuat: {actual_load_path}")
            print(f"     Class names : {self.class_names}")
        except Exception as e:
            print(f"[WARN] Gagal load model: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] Fallback ke MOCK mode")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def is_model_loaded(self) -> bool:
        return self.model is not None and self.mode == "REAL"

    def get_mode(self) -> str:
        return self.mode

    def reload_model(self, model_path: str = None):
        """Reload model setelah deploy baru."""
        if model_path:
            self.model_path = model_path
        self.model = None
        self.mode  = "MOCK"
        self.class_names = self._load_class_names()
        self._load_model()

    def preprocess_image(self, image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
        """
        Preprocess gambar untuk input CNN MobileNetV2:
        1. Resize ke 224x224
        2. Convert ke RGB
        3. Menyesuaikan normalisasi: jika model memiliki layer internal preprocess_input/rescaling,
           biarkan pixel dalam rentang [0, 255]. Jika model lawas tanpa layer tersebut, bagi 255.0.
        4. Expand dims -> shape (1, 224, 224, 3)
        """
        image = image.resize(target_size)
        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.array(image, dtype="float32")

        has_internal_preprocess = False
        if self.model is not None:
            for l in getattr(self.model, "layers", []):
                lname = getattr(l, "name", "").lower()
                if "preprocess" in lname or "rescaling" in lname or "normalization" in lname:
                    has_internal_preprocess = True
                    break

        if not has_internal_preprocess:
            arr = arr / 255.0

        return np.expand_dims(arr, axis=0)

    def predict(self, processed_image: np.ndarray) -> dict:
        """
        Prediksi kelas dari gambar yang sudah dipreprocess.

        Returns:
            {
                "class_id"     : int,
                "class_name"   : str,
                "confidence"   : float (0–100),
                "probabilities": list[float],
            }
        """
        if self.mode == "REAL" and self.model is not None:
            return self._real_predict(processed_image)
        return self._mock_predict()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _real_predict(self, processed_image: np.ndarray) -> dict:
        predictions  = self.model.predict(processed_image, verbose=0)
        class_id     = int(np.argmax(predictions[0]))
        confidence   = float(predictions[0][class_id]) * 100
        probabilities = [round(float(p) * 100, 2) for p in predictions[0]]

        return {
            "class_id"     : class_id,
            "class_name"   : self.class_names[class_id],
            "confidence"   : round(confidence, 2),
            "probabilities": probabilities,
        }

    def _mock_predict(self) -> dict:
        raw_probs  = np.random.dirichlet(np.ones(len(self.class_names))) * 100
        class_id   = int(np.argmax(raw_probs))
        confidence = float(raw_probs[class_id])

        return {
            "class_id"     : class_id,
            "class_name"   : self.class_names[class_id],
            "confidence"   : round(confidence, 2),
            "probabilities": [round(float(p), 2) for p in raw_probs],
        }
