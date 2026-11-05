"""
Training Service — Mengelola proses training CNN secara async
dengan progress tracking real-time.
"""

import os
import json
import time
import threading
import numpy as np
from datetime import datetime
from typing import Optional


class TrainingService:
    def __init__(self):
        self._status = {
            "status": "idle",          # idle | training | completed | error | stopped
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "train_acc": [],
            "val_acc": [],
            "train_loss": [],
            "val_loss": [],
            "message": "Belum ada training",
            "started_at": None,
            "finished_at": None,
            "best_accuracy": None,
            "model_saved": None,
        }
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def is_training(self) -> bool:
        return self._status["status"] == "training"

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def stop_training(self):
        self._stop_flag = True

    def reload_model(self, path: str):
        """Tidak digunakan di sini, tapi bisa di-override oleh CNNService."""
        pass

    def start_training(
        self,
        dataset_dir: str,
        models_dir: str,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        mode: str = "real",
    ):
        """Dijalankan sebagai background task oleh FastAPI."""
        self._stop_flag = False
        with self._lock:
            self._status.update({
                "status": "training",
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": epochs,
                "train_acc": [],
                "val_acc": [],
                "train_loss": [],
                "val_loss": [],
                "message": "Menyiapkan pipeline dataset TensorFlow...",
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
                "best_accuracy": None,
                "model_saved": None,
            })

        try:
            self._run_training(dataset_dir, models_dir, epochs, batch_size, learning_rate, mode=mode)
        except Exception as e:
            import traceback
            traceback.print_exc()
            with self._lock:
                self._status["status"]  = "error"
                self._status["message"] = f"Kendala pelatihan: {str(e)}"

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _update(self, **kwargs):
        with self._lock:
            self._status.update(kwargs)

    def _run_training(self, dataset_dir, models_dir, epochs, batch_size, learning_rate, mode="real"):
        if mode == "simulation":
            self._simulated_training(epochs)
            return

        # --- Jalankan pelatihan nyata dengan TensorFlow ---
        try:
            import tensorflow as tf
            self._tf_training(dataset_dir, models_dir, epochs, batch_size, learning_rate, tf)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[TrainingService] Error pelatihan nyata: {e}")
            with self._lock:
                self._status["status"] = "error"
                self._status["message"] = f"Gagal melatih: {str(e)}"

    def _tf_training(self, dataset_dir, models_dir, epochs, batch_size, learning_rate, tf):
        """Training nyata dengan TensorFlow + MobileNetV2."""
        self._update(message="Memuat dan memvalidasi dataset citra...")

        IMG_SIZE = (224, 224)
        class_names = ["Dermatitis", "Flea_Allergy", "Healthy", "Ringworm", "Scabies"]

        # Cek jumlah gambar
        all_dirs = [os.path.join(dataset_dir, c) for c in class_names]
        counts   = [len(os.listdir(d)) for d in all_dirs if os.path.exists(d)]
        if sum(counts) < 10:
            raise ValueError("Dataset terlalu sedikit untuk training nyata (minimal 10 gambar).")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            class_names=class_names,
            validation_split=0.2, subset="training",
            seed=42, image_size=IMG_SIZE, batch_size=batch_size,
            label_mode="categorical"
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            class_names=class_names,
            validation_split=0.2, subset="validation",
            seed=42, image_size=IMG_SIZE, batch_size=batch_size,
            label_mode="categorical"
        )

        num_classes = len(class_names)
        total_batches = len(train_ds)
        AUTOTUNE    = tf.data.AUTOTUNE
        train_ds    = train_ds.prefetch(AUTOTUNE)
        val_ds      = val_ds.prefetch(AUTOTUNE)

        # 1. Pipeline Augmentasi Medis Kuat (Mencegah Overfitting)
        augment = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.1),
        ])

        # 2. Inisialisasi Backbone MobileNetV2 (Pre-trained ImageNet)
        self._update(message="Menginisialisasi arsitektur MobileNetV2...", progress=3)
        base = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
        base.trainable = False

        # 3. Lapisan Head Klasifikasi Ber-regularisasi Tinggi (L2 Regularizer + Dual Dropout)
        inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
        x       = augment(inputs)
        x       = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        x       = base(x, training=False)
        x       = tf.keras.layers.GlobalAveragePooling2D()(x)
        x       = tf.keras.layers.BatchNormalization()(x)
        x       = tf.keras.layers.Dropout(0.4)(x)
        x       = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(2e-4)
        )(x)
        x       = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model   = tf.keras.Model(inputs, outputs)

        # 4. Loss Function dengan Label Smoothing 0.1 untuk Mencegah Overconfidence
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=["accuracy"]
        )

        # Alokasi epoch: 60% Warmup agar classifier head mapan, 40% Fine-tuning lembut
        phase1_epochs = max(3, int(epochs * 0.6)) if epochs >= 6 else epochs
        phase2_epochs = epochs - phase1_epochs

        best_weights = None
        best_val_acc = 0.0

        class LiveProgressCallback(tf.keras.callbacks.Callback):
            def __init__(cb_self, phase_name, start_epoch_offset):
                super().__init__()
                cb_self.phase_name = phase_name
                cb_self.start_epoch_offset = start_epoch_offset
                cb_self.current_rel_epoch = 0

            def on_epoch_begin(cb_self, epoch, logs=None):
                cb_self.current_rel_epoch = epoch
                abs_epoch = cb_self.start_epoch_offset + epoch + 1
                with self._lock:
                    self._status["current_epoch"] = abs_epoch
                    self._status["message"] = f"Epoch {abs_epoch}/{epochs} ({cb_self.phase_name}) • Memproses data..."

            def on_train_batch_end(cb_self, batch, logs=None):
                if self._stop_flag:
                    cb_self.model.stop_training = True
                    return
                logs = logs or {}
                batch_loss = logs.get("loss", 0.0)
                batch_acc = logs.get("accuracy", 0.0)
                abs_epoch_idx = cb_self.start_epoch_offset + cb_self.current_rel_epoch
                progress_fraction = (abs_epoch_idx + (batch + 1) / max(1, total_batches)) / epochs
                pct = min(99, max(2, int(progress_fraction * 100)))
                with self._lock:
                    self._status["progress"] = pct
                    self._status["message"] = (
                        f"Epoch {abs_epoch_idx + 1}/{epochs} ({cb_self.phase_name}) "
                        f"• Batch {batch + 1}/{total_batches} "
                        f"• Loss: {batch_loss:.3f} • Acc: {batch_acc * 100:.1f}%"
                    )

            def on_epoch_end(cb_self, epoch, logs=None):
                nonlocal best_weights, best_val_acc
                if self._stop_flag:
                    cb_self.model.stop_training = True
                    return
                logs = logs or {}
                train_acc = round(logs.get("accuracy", 0) * 100, 2)
                val_acc = round(logs.get("val_accuracy", 0) * 100, 2)
                abs_epoch = cb_self.start_epoch_offset + epoch + 1
                pct = int((abs_epoch / epochs) * 100)

                # Simpan bobot terbaik untuk proteksi anti-overfitting
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = cb_self.model.get_weights()

                with self._lock:
                    self._status["current_epoch"] = abs_epoch
                    self._status["progress"]      = pct
                    self._status["train_acc"].append(train_acc)
                    self._status["val_acc"].append(val_acc)
                    self._status["train_loss"].append(round(logs.get("loss", 0), 4))
                    self._status["val_loss"].append(round(logs.get("val_loss", 0), 4))
                    self._status["message"] = (
                        f"Epoch {abs_epoch}/{epochs} ({cb_self.phase_name}) selesai "
                        f"— Val Acc: {val_acc:.1f}% (Terbaik: {best_val_acc:.1f}%)"
                    )

        # -------------------------------------------------------------
        # FASE 1: Feature Extraction (Warmup Head Classifier)
        # -------------------------------------------------------------
        self._update(message="Memulai Fase 1: Feature Extraction (Backbone Beku)...", progress=5)
        cb_phase1 = LiveProgressCallback("Fase 1: Warmup", start_epoch_offset=0)
        
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=phase1_epochs,
            callbacks=[cb_phase1],
            verbose=0
        )

        if self._stop_flag:
            self._update(
                status="stopped",
                message="Training dihentikan oleh pengguna.",
                finished_at=datetime.now().isoformat()
            )
            return

        # -------------------------------------------------------------
        # FASE 2: Selective Fine-Tuning (Unfreeze 12 Layer Teratas Saja)
        # -------------------------------------------------------------
        if phase2_epochs > 0 and not self._stop_flag:
            self._update(
                message="Memulai Fase 2: Fine-Tuning Lembut (Membuka 12 Layer Teratas)...",
                progress=int((phase1_epochs / epochs) * 100)
            )

            base.trainable = True
            # Freeze layer dasar (0 s/d 142), HANYA buka 12 layer teratas (1 bottleneck block terakhir)
            for layer in base.layers[:-12]:
                layer.trainable = False

            # Learning rate sangat lembut (3e-5) agar filter pre-trained ImageNet tidak teracak
            fine_tune_lr = min(learning_rate * 0.03, 3e-5)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
                loss=loss_fn,
                metrics=["accuracy"]
            )

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0
            )

            cb_phase2 = LiveProgressCallback("Fase 2: Fine-Tuning", start_epoch_offset=phase1_epochs)

            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=phase2_epochs,
                callbacks=[cb_phase2, reduce_lr],
                verbose=0
            )

        if self._stop_flag:
            self._update(
                status="stopped",
                message="Training dihentikan oleh pengguna.",
                finished_at=datetime.now().isoformat()
            )
            return

        # -------------------------------------------------------------
        # Proteksi Anti-Overfitting: Kembalikan Bobot dengan Akurasi Tertinggi
        # -------------------------------------------------------------
        if best_weights is not None:
            model.set_weights(best_weights)

        final_val_acc = best_val_acc if best_val_acc > 0 else (self._status["val_acc"][-1] if self._status["val_acc"] else 0)

        # Simpan file model hasil training nyata
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_mobilenetv2_{timestamp}.keras"
        model_path = os.path.join(models_dir, model_name)
        model.save(model_path)

        # Simpan cadangan format .h5
        h5_name = f"model_mobilenetv2_{timestamp}.h5"
        try:
            model.save(os.path.join(models_dir, h5_name))
        except Exception:
            pass

        # Perbarui juga slot model aktif di produksi
        try:
            model.save(os.path.join(models_dir, "cnn_model.h5"))
        except Exception:
            pass

        val_acc = final_val_acc

        # Simpan metadata model info JSON agar terbaca di Model Management
        info = {
            "val_accuracy": val_acc,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "class_names": class_names,
            "created_at": datetime.now().isoformat(),
            "mode": "REAL"
        }
        with open(os.path.join(models_dir, f"model_mobilenetv2_{timestamp}_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        with open(os.path.join(models_dir, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        self._update(
            status="completed",
            progress=100,
            message=f"Pelatihan TensorFlow Asli Selesai! Akurasi: {val_acc:.1f}%",
            finished_at=datetime.now().isoformat(),
            best_accuracy=val_acc,
            model_saved=model_name
        )

    def _simulated_training(self, epochs: int):
        """Simulasi training visual instan untuk demo."""
        import random
        self._update(message="TensorFlow tidak tersedia — menjalankan simulasi training...")
        time.sleep(1)

        base_acc  = 60.0
        base_loss = 1.5

        for epoch in range(epochs):
            if self._stop_flag:
                self._update(status="stopped", message="Training dihentikan.", finished_at=datetime.now().isoformat())
                return
            time.sleep(0.8)

            noise       = random.uniform(-1.5, 2.5)
            train_acc   = min(99.0, base_acc + (epoch / epochs) * 35 + noise)
            val_acc     = min(98.0, base_acc + (epoch / epochs) * 30 + noise * 0.8)
            train_loss  = max(0.05, base_loss - (epoch / epochs) * 1.3 + random.uniform(-0.05, 0.05))
            val_loss    = max(0.08, base_loss - (epoch / epochs) * 1.1 + random.uniform(-0.05, 0.1))
            pct         = int(((epoch + 1) / epochs) * 100)

            with self._lock:
                self._status["current_epoch"] = epoch + 1
                self._status["progress"]      = pct
                self._status["train_acc"].append(round(train_acc, 2))
                self._status["val_acc"].append(round(val_acc, 2))
                self._status["train_loss"].append(round(train_loss, 4))
                self._status["val_loss"].append(round(val_loss, 4))
                self._status["message"]       = f"Epoch {epoch+1}/{epochs} — val_acc: {val_acc:.1f}% (simulasi)"

        final_acc = self._status["val_acc"][-1] if self._status["val_acc"] else 0
        self._update(
            status="completed", progress=100,
            message=f"Simulasi training selesai! Akurasi: {final_acc:.1f}%",
            finished_at=datetime.now().isoformat(), best_accuracy=final_acc,
            model_saved="simulated_model.h5"
        )
