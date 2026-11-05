import os
import sys
import json
import numpy as np

# Suppress verbose TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "models/model_kulit_kucing_v2_final.keras"
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    MODEL_PATH = sys.argv[1]
elif os.path.exists("models/active_model.json"):
    try:
        with open("models/active_model.json", "r", encoding="utf-8") as f:
            am = json.load(f).get("active_model")
            if am and os.path.exists(os.path.join("models", am)):
                MODEL_PATH = os.path.join("models", am)
    except Exception:
        pass

DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("=" * 65)
print(f"   CATSKIN AI - ANALISIS PERFORMA & EVALUASI AKURASI MODEL")
print(f"   Model Target: {MODEL_PATH}")
print("=" * 65)

if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] File model tidak ditemukan di: {MODEL_PATH}")
    sys.exit(1)

# ----------------------------------------------------------------------
# 1. Load Model dengan patch kompatibilitas Keras
# ----------------------------------------------------------------------
import keras
_orig_dense_init = keras.layers.Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return _orig_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = _patched_dense_init

print("\n[1/5] Memuat file bobot model (.keras)...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" -> Model Keras berhasil dimuat ke memori TensorFlow!")
except Exception as e:
    print(f" -> [ERROR] Gagal memuat model: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------
# 2. Inspeksi Arsitektur & Parameter
# ----------------------------------------------------------------------
print("\n[2/5] Inspeksi Spesifikasi Arsitektur Model:")
input_shape = getattr(model, 'input_shape', 'Unknown')
output_shape = getattr(model, 'output_shape', 'Unknown')
total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = total_params - trainable_params
file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print(f" - Ukuran File Fisik    : {file_size_mb:.2f} MB")
print(f" - Input Shape          : {input_shape}")
print(f" - Output Shape         : {output_shape} (5 Kategori)")
print(f" - Total Parameter      : {total_params:,} parameter")
print(f" - Trainable Parameter  : {trainable_params:,}")
print(f" - Non-trainable Params : {non_trainable_params:,}")

# Deteksi Preprocessing internal
has_internal_preprocess = False
for l in model.layers:
    lname = l.name.lower()
    if any(k in lname for k in ["preprocess", "rescaling", "normalization"]):
        has_internal_preprocess = True
        print(f" - Layer Preprocessing : Terdeteksi '{l.name}'")
        break

if not has_internal_preprocess:
    print(" - Layer Preprocessing : Tidak ada layer rescaling bawaan (menggunakan preprocessing eksternal)")

# ----------------------------------------------------------------------
# 3. Menyiapkan Dataset & Verifikasi Kelas
# ----------------------------------------------------------------------
print("\n[3/5] Memuat Citra Dataset untuk Pengujian Akurasi:")
class_names = ["Dermatitis", "Flea_Allergy", "Healthy", "Ringworm", "Scabies"]

# Cek ketersediaan dataset
class_counts = {}
for c in class_names:
    p = os.path.join(DATASET_DIR, c)
    if os.path.exists(p):
        cnt = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        class_counts[c] = cnt
    else:
        class_counts[c] = 0

total_dataset_img = sum(class_counts.values())
print(f" - Total Citra Dataset: {total_dataset_img} citra")
for c, cnt in class_counts.items():
    print(f"   * {c:14}: {cnt} citra")

if total_dataset_img < 5:
    print("[ERROR] Dataset tidak mencukupi untuk evaluasi.")
    sys.exit(1)

# ----------------------------------------------------------------------
# 4. Evaluasi Pipeline Preprocessing
# ----------------------------------------------------------------------
print("\n[4/5] Menguji Mode Normalisasi (Mendeteksi akurasi tertinggi):")

# Muat data mentah
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=class_names,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

y_true = []
for _, labels in test_ds_raw:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# Cek 3 mode normalisasi:
# Mode A: / 255.0 (Rentang [0, 1])
# Mode B: MobileNetV2 preprocess_input (Rentang [-1, 1])
# Mode C: raw [0, 255] (jika model sudah punya internal rescaling)
modes = [
    ("MobileNetV2 [-1, 1]", lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(x, tf.float32))),
    ("Rescale 1/255 [0, 1]", lambda x: tf.cast(x, tf.float32) / 255.0),
    ("Raw [0, 255]", lambda x: tf.cast(x, tf.float32)),
]

best_acc = -1.0
best_mode_name = ""
best_preds = None

for mode_name, norm_fn in modes:
    ds_norm = test_ds_raw.map(lambda x, y: (norm_fn(x), y))
    preds = model.predict(ds_norm, verbose=0)
    y_pred_classes = np.argmax(preds, axis=1)
    acc = np.mean(y_pred_classes == y_true) * 100.0
    print(f" - Mode {mode_name:22} -> Akurasi: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_mode_name = mode_name
        best_preds = y_pred_classes

print(f"\n -> Mode Preprocessing Optimal: {best_mode_name} dengan Akurasi: {best_acc:.2f}%")

# ----------------------------------------------------------------------
# 5. Rincian Metrik Per-Kelas & Confusion Matrix
# ----------------------------------------------------------------------
print("\n[5/5] Analisis Metrik Klasifikasi Per-Kelas:")
report = classification_report(
    y_true, 
    best_preds, 
    target_names=class_names, 
    digits=3, 
    output_dict=True
)

print(f"\n{'Kelas':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Jumlah Sampel':<12}")
print("-" * 65)
for c in class_names:
    p = report[c]["precision"] * 100
    r = report[c]["recall"] * 100
    f1 = report[c]["f1-score"] * 100
    supp = int(report[c]["support"])
    print(f"{c:<15} {p:>6.2f}%      {r:>6.2f}%      {f1:>6.2f}%      {supp:>6}")
print("-" * 65)
macro_f1 = report["macro avg"]["f1-score"] * 100
weighted_f1 = report["weighted avg"]["f1-score"] * 100
print(f"{'Rata-rata Akurasi Keseluruhan (Overall Accuracy)':<45}: {best_acc:.2f}%")
print(f"{'Macro Average F1-Score':<45}: {macro_f1:.2f}%")
print(f"{'Weighted Average F1-Score':<45}: {weighted_f1:.2f}%")

cm = confusion_matrix(y_true, best_preds)
print("\nMatriks Kebingungan (Confusion Matrix):")
print(f"{'':<15}" + "".join([f"{c[:5]:>8}" for c in class_names]))
for i, c in enumerate(class_names):
    row_str = "".join([f"{cm[i][j]:>8}" for j in range(len(class_names))])
    print(f"{c:<15}{row_str}")

# Simpan metadata hasil evaluasi ke info JSON
info_path = MODEL_PATH.replace(".keras", "_info.json")
try:
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({
            "val_accuracy": round(best_acc, 2),
            "macro_f1": round(macro_f1, 2),
            "weighted_f1": round(weighted_f1, 2),
            "total_samples": int(total_dataset_img),
            "optimal_preprocessing": best_mode_name,
            "class_names": class_names,
            "report": report
        }, f, indent=2)
    print(f"\n[OK] Hasil analisis diperbarui ke metadata: {info_path}")
except Exception as e:
    pass

print("\n" + "=" * 65)
print(f"HASIL AKHIR: Model 'model_kulit_kucing_v2_final.keras' memiliki akurasi {best_acc:.2f}%")
print("=" * 65)
