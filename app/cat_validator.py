"""
CatSkin AI - Cat Skin Image Validator & Out-of-Distribution (OOD) Detector
Memastikan bahwa gambar yang masuk ke sistem CNN benar-benar merupakan kulit/tubuh kucing
dan menolak objek lain (truk, jalanan, kardus, perabotan, ruangan, manusia, benda mati, dsb.)
"""

import numpy as np
from PIL import Image
import math
import os
import json

class CatSkinValidator:
    """
    Validator multi-tahap cerdas:
    1. AI Object Detector (ImageNet MobileNetV2): Menolak truk, mobil, jalan raya, perabotan, barang, dsb.
    2. Analisis Spektrum Warna Biologis Kulit & Bulu Mamalia (YCbCr + RGB space)
    3. Analisis Rasio Kegelapan / Kecerahan Ekstrem
    4. Analisis Entropi & Distribusi Probabilitas Softmax
    """

    MIN_ORGANIC_RATIO = 40.0
    MAX_EXTREME_DARK = 40.0
    MAX_EXTREME_BRIGHT = 50.0
    MAX_ARTIFICIAL_COLD = 35.0
    MIN_TEXTURE_VARIATION = 8.0
    MIN_MODEL_CONFIDENCE = 25.0

    _imagenet_model = None
    _class_idx = None

    @classmethod
    def _get_imagenet_model(cls):
        """Lazy load pre-trained ImageNet MobileNetV2 untuk deteksi objek non-hewan"""
        if cls._imagenet_model is None:
            try:
                import tensorflow as tf
                cls._imagenet_model = tf.keras.applications.MobileNetV2(
                    weights="imagenet", include_top=True
                )
                idx_path = tf.keras.utils.get_file(
                    'imagenet_class_index.json',
                    'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json',
                    cache_subdir='models'
                )
                with open(idx_path, 'r', encoding='utf-8') as f:
                    cls._class_idx = json.load(f)
                print("[OK] ImageNet Object Detector loaded for non-cat rejection.")
            except Exception as e:
                print(f"[WARN] Gagal memuat ImageNet detector: {e}")
                cls._imagenet_model = False
        return cls._imagenet_model

    @classmethod
    def detect_general_object(cls, image: Image.Image) -> dict:
        """
        Mendeteksi apakah citra adalah kucing/hewan vs benda mati/jalanan/kendaraan
        menggunakan 1.000 kategori ImageNet.
        """
        model = cls._get_imagenet_model()
        if not model or model is False or not cls._class_idx:
            return {"is_animal": True, "animal_score": 50.0, "top1_label": "unknown"}

        try:
            import tensorflow as tf
            img = image.convert("RGB").resize((224, 224))
            x = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.expand_dims(np.array(img, dtype="float32"), 0)
            )
            preds = model.predict(x, verbose=0)[0]
            top5_idx = np.argsort(preds)[-5:][::-1]
            top_labels = [cls._class_idx[str(i)][1] for i in top5_idx]
            top_scores = [float(preds[i]) for i in top5_idx]

            # Kelas ImageNet:
            # 0 s/d 397: Animalia (mamalia, felid, canid, burung, dsb.)
            # 281 s/d 290: Kucing domestik & felidae (tabby, tiger_cat, Persian_cat, Siamese_cat, Egyptian_cat, cougar, lynx)
            # 398 s/d 999: Benda mati, kendaraan, jalan, furnitur, elektronik, pakaian, lanskap
            animal_score = sum(top_scores[i] for i, idx in enumerate(top5_idx) if idx < 398) * 100.0
            has_feline = any(281 <= idx <= 290 for idx in top5_idx)

            # Jika skor hewan minimal 5% ATAU ada kelas felid di top-5, anggap berunsur hewan
            is_animal = (animal_score >= 5.0) or has_feline
            top1_clean = top_labels[0].replace("_", " ")

            return {
                "is_animal": is_animal,
                "animal_score": round(animal_score, 1),
                "top1_label": top1_clean,
                "top_predictions": [l.replace("_", " ") for l in top_labels[:3]]
            }
        except Exception as e:
            print(f"[WARN] Error dalam detect_general_object: {e}")
            return {"is_animal": True, "animal_score": 50.0, "top1_label": "error"}

    @classmethod
    def analyze_visual_features(cls, image: Image.Image) -> dict:
        """
        Menganalisis karakteristik spektrum warna biologis kulit dan bulu.
        """
        img = image.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32)

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        y  = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

        skin_mask = (cr >= 128) & (cr <= 185) & (cb >= 70) & (cb <= 140)
        fur_warm = (r > g) & (g >= b * 0.7) & (r > 35) & (y > 30) & (y < 240)
        fur_neutral = (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (y >= 40) & (y <= 235)

        organic_mask = skin_mask | fur_warm | fur_neutral
        organic_ratio = float(np.mean(organic_mask)) * 100.0

        extreme_dark_ratio = float(np.mean(y < 20)) * 100.0
        extreme_bright_ratio = float(np.mean(y > 245)) * 100.0
        artificial_cold = float(np.mean((b > r * 1.3) & (b > g * 1.1) & (b > 60))) * 100.0

        std_per_channel = [float(np.std(r)), float(np.std(g)), float(np.std(b))]
        avg_std = float(np.mean(std_per_channel))

        is_visually_plausible = (
            organic_ratio >= cls.MIN_ORGANIC_RATIO and
            extreme_dark_ratio <= cls.MAX_EXTREME_DARK and
            extreme_bright_ratio <= cls.MAX_EXTREME_BRIGHT and
            artificial_cold <= cls.MAX_ARTIFICIAL_COLD and
            avg_std >= cls.MIN_TEXTURE_VARIATION
        )

        return {
            "organic_ratio": round(organic_ratio, 2),
            "extreme_dark_ratio": round(extreme_dark_ratio, 2),
            "extreme_bright_ratio": round(extreme_bright_ratio, 2),
            "artificial_cold": round(artificial_cold, 2),
            "texture_variation": round(avg_std, 2),
            "is_visually_plausible": is_visually_plausible,
        }

    @classmethod
    def validate_prediction_distribution(cls, probabilities: list) -> dict:
        """
        Menganalisis entropi dan margin distribusi probabilitas.
        """
        if not probabilities or len(probabilities) < 2:
            return {"is_confident": False, "reason": "Distribusi probabilitas tidak lengkap."}

        sorted_probs = sorted(probabilities, reverse=True)
        top1 = sorted_probs[0]
        top2 = sorted_probs[1]
        margin = top1 - top2

        entropy = 0.0
        for p in probabilities:
            if p > 0.0001:
                p_norm = p / 100.0 if p > 1.0 else p
                entropy -= p_norm * math.log(p_norm + 1e-12)

        is_confident = (top1 >= cls.MIN_MODEL_CONFIDENCE)

        return {
            "top_confidence": round(top1, 2),
            "runner_up_confidence": round(top2, 2),
            "margin": round(margin, 2),
            "entropy": round(entropy, 3),
            "is_confident": is_confident,
        }

    @classmethod
    def evaluate(cls, image: Image.Image, probabilities: list) -> dict:
        """
        Evaluasi komprehensif:
        1. Cek apakah objek adalah hewan/kucing vs non-hewan (truk, jalanan, perabotan, barang)
        2. Cek fitur spektrum warna biologis
        3. Cek distribusi probabilitas model
        """
        obj_detect = cls.detect_general_object(image)
        visual = cls.analyze_visual_features(image)
        dist = cls.validate_prediction_distribution(probabilities)

        reasons = []
        is_valid_cat_skin = True

        # Tolak jika objek yang terdeteksi secara jelas adalah benda mati / kendaraan / non-hewan
        if not obj_detect["is_animal"]:
            is_valid_cat_skin = False
            top_obj = obj_detect["top1_label"]
            reasons.append(
                f"Sistem mendeteksi objek non-hewan: '{top_obj}'. Foto bukan merupakan kucing atau hewan berbulu."
            )

        # Tolak jika karakteristik visual tidak memenuhi kriteria organik
        if not visual["is_visually_plausible"]:
            is_valid_cat_skin = False
            if visual["organic_ratio"] < cls.MIN_ORGANIC_RATIO:
                reasons.append(
                    f"Spektrum warna dan tekstur kulit/bulu kucing tidak terdeteksi (rasio biologis: {visual['organic_ratio']}%)."
                )
            if visual["extreme_dark_ratio"] > cls.MAX_EXTREME_DARK:
                reasons.append(
                    f"Foto didominasi area gelap/bayangan ({visual['extreme_dark_ratio']}%), bukan foto area kulit kucing."
                )
            if visual["extreme_bright_ratio"] > cls.MAX_EXTREME_BRIGHT:
                reasons.append("Foto terlalu terang atau menyerupai dokumen/kertas putih.")
            if visual["artificial_cold"] > cls.MAX_ARTIFICIAL_COLD:
                reasons.append("Warna didominasi spektrum sintetis/buatan.")

        user_friendly_message = (
            "Foto tidak terdeteksi sebagai kulit kucing yang valid. "
            "Sistem AI mendeteksi bahwa gambar yang Anda unggah bukan merupakan area lesi kulit atau bulu kucing (misalnya foto kardus, kendaraan, jalanan, manusia, atau benda mati). "
            "Mohon unggah foto jarak dekat (close-up) pada area kulit kucing yang mengalami masalah."
        )

        return {
            "is_valid_cat_skin": is_valid_cat_skin,
            "user_friendly_message": user_friendly_message,
            "reasons": reasons,
            "object_detection": obj_detect,
            "visual_analysis": visual,
            "distribution_analysis": dist,
        }
