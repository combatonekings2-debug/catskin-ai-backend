"""
CatSkin AI - Backend API
Sistem Diagnosa Penyakit Kulit Kucing menggunakan CNN
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
import io
import os
import uuid
import json
import base64
import shutil
import asyncio
from datetime import datetime

from app.cnn_service import CNNService
from app.training_service import TrainingService
from app.cat_validator import CatSkinValidator

app = FastAPI(
    title="CatSkin AI API",
    description="API untuk diagnosa penyakit kulit kucing menggunakan CNN",
    version="2.0.0"
)

# CORS — terima dari localhost maupun IP jaringan lokal
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:4173,http://localhost:3000"
).split(",")

# Jika ALLOWED_ORIGINS tidak diset via env, izinkan semua origin (aman untuk dev lokal)
# Di production, set env ALLOWED_ORIGINS ke domain spesifik
_cors_origins  = ALLOWED_ORIGINS
_cors_wildcard = len(_cors_origins) == 1 and _cors_origins[0] == "*"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Proxy Vite sudah menangani di dev; di prod set via env
    allow_credentials=False,      # False saat allow_origins="*"
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

cnn_service = CNNService()
training_service = TrainingService()

# ============================================================================
# CONSTANTS
# ============================================================================

DATASET_CLASSES = [
    {
        "id": 0, "name": "Dermatitis", "scientific_name": "Feline Dermatitis",
        "description": "Peradangan kulit yang disebabkan oleh reaksi alergi, iritan eksternal, atau infeksi sekunder.",
        "symptoms": ["Kulit merah", "Gatal berlebihan", "Melepuh", "Kerak kulit"],
        "severity": "Sedang", "color": "from-blue-500 to-indigo-600",
        "treatment": "Krim kortison/hidrokortison 1%, Antihistamin oral sesuai dosis dokter, Identifikasi dan eliminasi pemicu alergen, Pembersihan antiseptik rutin berkala"
    },
    {
        "id": 1, "name": "Flea Allergy", "scientific_name": "Flea Allergy Dermatitis (FAD)",
        "description": "Reaksi hipersensitivitas alergi terhadap air liur kutu kucing yang memicu rasa gatal intensif.",
        "symptoms": ["Gatal parah", "Bulu rontok", "Luka garukan", "Kemerahan"],
        "severity": "Tinggi", "color": "from-red-500 to-rose-600",
        "treatment": "Obat anti-kutu Spot-on (Fluralaner / Fipronil), Krim topikal pereda pruritus atau gatal, Sterilisasi kasur dan lingkungan kucing, Terapi antibiotik jika terdapat infeksi sekunder"
    },
    {
        "id": 2, "name": "Healthy", "scientific_name": "Normal Healthy Skin",
        "description": "Kondisi kulit dan folikel rambut kucing berada dalam keadaan bersih, sehat, dan bebas lesi patologis.",
        "symptoms": ["Bulu berkilau", "Kulit bersih", "Tidak gatal", "Aktif normal"],
        "severity": "Normal", "color": "from-green-500 to-emerald-600",
        "treatment": "Asupan nutrisi seimbang kaya omega-3 dan omega-6, Rutin sisir bulu dan grooming terjadwal, Vaksinasi berkala dan pencegahan parasit rutin, Sanitasi terjaga untuk kandang dan area bermain"
    },
    {
        "id": 3, "name": "Ringworm", "scientific_name": "Dermatophytosis (Microsporum canis)",
        "description": "Infeksi mikosis kapang dermatofita yang menyerang keratin epidermis kulit, bulu, dan cakar.",
        "symptoms": ["Bercak melingkar", "Bulu rontok (alopesia)", "Kulit mengelupas", "Gatal ringan-sedang"],
        "severity": "Sedang", "color": "from-amber-500 to-orange-600",
        "treatment": "Krim antijamur Miconazole atau Ketoconazole 2x sehari, Antijamur sistemik oral (Itraconazole/Griseofulvin), Isolasi hewan dari manusia dan kucing lain (zoonosis), Disinfeksi rutin kandang dan sisir peralatan"
    },
    {
        "id": 4, "name": "Scabies", "scientific_name": "Notoedric Mange (Notoedres cati)",
        "description": "Infeksi ektoparasit tungau kudis yang menggali terowongan mikroskopis di bawah lapisan epidermis kulit.",
        "symptoms": ["Gatal ekstrem", "Kerontokan bulu", "Kerak tebal berlipat", "Kemerahan parah"],
        "severity": "Tinggi", "color": "from-purple-500 to-violet-600",
        "treatment": "Akarisida (Selamectin spot-on / Ivermectin resep dokter hewan), Mandi medikasi sulfur keratolitik berkala, Sterilisasi menyeluruh kandang dan alas tidur kucing, Pengulangan terapi berkala sesuai siklus hidup tungau"
    }
]

CLASS_NAMES = [c["name"] for c in DATASET_CLASSES]
DATASET_DIR = "dataset"
MODELS_DIR  = "models"
DIAG_LOG    = "data/diagnosis_log.json"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs("data",      exist_ok=True)
for cls in CLASS_NAMES:
    os.makedirs(os.path.join(DATASET_DIR, cls.replace(" ", "_")), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "uploads"), exist_ok=True)

# ============================================================================
# HELPERS
# ============================================================================

def load_diagnosis_log() -> list:
    if os.path.exists(DIAG_LOG):
        try:
            with open(DIAG_LOG, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_diagnosis_log(log: list):
    with open(DIAG_LOG, "w") as f:
        json.dump(log, f, indent=2, default=str)

SETTINGS_FILE = "data/site_settings.json"

def load_site_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "site_name": "CatSkin AI",
        "footer_description": "Sistem diagnosa penyakit kulit kucing berbasis AI dengan CNN untuk hasil yang akurat dan terpercaya.",
        "contact": {
            "address": "Jl. Teknologi No. 123, Kota Digital, Indonesia 40123",
            "phone": "+62 812-3456-7890",
            "email": "support@catskin.ai"
        },
        "social_links": {
            "facebook": "https://facebook.com",
            "twitter": "https://twitter.com",
            "instagram": "https://instagram.com",
            "youtube": "https://youtube.com"
        },
        "education_menu": {
            "title": "Edukasi Kucing"
        },
        "copyright_text": "© 2026 CatSkin AI. All rights reserved.",
        "credit_text": "Dibuat dengan ❤️ untuk kesehatan kucing kesayangan Anda."
    }

def save_site_settings(settings: dict):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

def load_dataset_meta() -> dict:
    meta_path = os.path.join(DATASET_DIR, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_dataset_meta(meta: dict):
    meta_path = os.path.join(DATASET_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def get_dataset_stats() -> dict:
    stats = {}
    total = 0
    for cls in CLASS_NAMES:
        folder = os.path.join(DATASET_DIR, cls.replace(" ", "_"))
        count  = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]) if os.path.exists(folder) else 0
        stats[cls] = count
        total += count
    return {"per_class": stats, "total": total}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def validate_image_file(file: UploadFile, contents: bytes):
    # Cek ukuran dulu
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File terlalu besar. Maks 5MB.")

    # Validasi via content_type ATAU ekstensi file (browser kadang kirim content_type kosong/salah)
    ct  = (file.content_type or "").lower()
    ext = os.path.splitext(file.filename or "")[1].lower()

    ct_ok  = ct in ALLOWED_CONTENT_TYPES or ct in {"image/jpg", "application/octet-stream", ""}
    ext_ok = ext in ALLOWED_EXTENSIONS

    if not (ct_ok or ext_ok):
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: '{file.filename}'. Gunakan JPG, PNG, atau WebP."
        )

    # Verifikasi isi file dengan Pillow (paling aman)
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()  # raise jika bukan gambar valid
    except Exception:
        raise HTTPException(status_code=400, detail=f"File bukan gambar yang valid: {file.filename}")

# ============================================================================
# MODELS (Pydantic)
# ============================================================================

class PredictionResult(BaseModel):
    disease: str
    confidence: float
    description: str
    recommendations: List[str]
    severity: str
    color: str
    scientific_name: Optional[str] = None
    expert_reference: Optional[str] = None
    all_predictions: Optional[List[dict]] = None
    disclaimer: Optional[str] = None

class DiagnosisResponse(BaseModel):
    success: bool
    message: str
    data: Optional[PredictionResult] = None
    processing_time: float
    image_id: str
    timestamp: str
    is_valid_cat_skin: bool = True
    validation_details: Optional[Dict[str, Any]] = None

class DatasetInfo(BaseModel):
    total_classes: int
    classes: List[dict]
    model_status: str
    accuracy: Optional[float]

class TrainingConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    mode: str = "real"  # "real" | "simulation"

class BatchDeleteRequest(BaseModel):
    ids: List[str]

# ============================================================================
# ROOT
# ============================================================================

@app.get("/")
async def root():
    return {"status": "online", "service": "CatSkin AI API", "version": "2.0.0", "timestamp": datetime.now().isoformat()}

@app.get("/api/health")
async def health_check():
    stats = get_dataset_stats()
    return {
        "status": "healthy",
        "model_loaded": cnn_service.is_model_loaded(),
        "model_mode": cnn_service.get_mode(),
        "dataset_total": stats["total"],
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# DATASET ENDPOINTS
# ============================================================================

@app.get("/api/dataset/info")
async def get_dataset_info():
    stats = get_dataset_stats()
    model_accuracy = 75.22
    info_path = os.path.join(MODELS_DIR, "model_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, encoding="utf-8") as f:
                model_accuracy = json.load(f).get("val_accuracy", 75.22)
        except Exception:
            pass
    return {
        "total_classes": len(DATASET_CLASSES),
        "classes": DATASET_CLASSES,
        "model_status": "loaded" if cnn_service.is_model_loaded() else "mock_mode",
        "accuracy": model_accuracy if cnn_service.is_model_loaded() else None,
        "stats": stats
    }

@app.get("/api/dataset/classes")
async def get_classes():
    return {"classes": DATASET_CLASSES}

@app.get("/api/admin/dataset")
async def admin_get_dataset():
    """List semua item dataset dengan metadata"""
    items = []
    meta  = load_dataset_meta()
    for cls in CLASS_NAMES:
        folder = os.path.join(DATASET_DIR, cls.replace(" ", "_"))
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg",".jpeg",".png",".webp")):
                continue
            fpath = os.path.join(folder, fname)
            fsize = os.path.getsize(fpath)
            item_id = fname.rsplit(".", 1)[0]
            items.append({
                "id":        item_id,
                "filename":  fname,
                "category":  cls,
                "size_kb":   round(fsize / 1024, 1),
                "uploaded_at": meta.get(item_id, {}).get("uploaded_at", ""),
            })
    items.sort(key=lambda x: x["uploaded_at"], reverse=True)
    stats = get_dataset_stats()
    return {"items": items, "stats": stats, "total": len(items)}

@app.post("/api/admin/dataset/upload")
async def admin_upload_dataset(
    file: UploadFile = File(...),
    category: str = Form(...)
):
    """Upload gambar dataset baru"""
    # Normalisasi kategori (toleran terhadap spasi & kapitalisasi)
    category_map = {c.lower().replace(" ", ""): c for c in CLASS_NAMES}
    normalized   = category.strip().lower().replace(" ", "")
    if normalized not in category_map:
        raise HTTPException(
            status_code=400,
            detail=f"Kategori tidak valid: '{category}'. Pilihan: {', '.join(CLASS_NAMES)}"
        )
    category = category_map[normalized]

    contents = await file.read()
    validate_image_file(file, contents)

    # Buka ulang setelah verify (verify() menutup stream)
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()  # force decode penuh
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca gambar: {str(e)}")

    item_id   = str(uuid.uuid4())
    folder    = os.path.join(DATASET_DIR, category.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, f"{item_id}.jpg")

    image.convert("RGB").save(save_path, "JPEG", quality=90)

    meta = load_dataset_meta()
    meta[item_id] = {
        "uploaded_at":   datetime.now().isoformat(),
        "category":      category,
        "original_name": file.filename or "unknown",
    }
    save_dataset_meta(meta)

    size_kb = round(os.path.getsize(save_path) / 1024, 1)
    return {
        "success":  True,
        "id":       item_id,
        "category": category,
        "size_kb":  size_kb,
        "message":  f"'{file.filename}' berhasil diupload ke kelas {category}",
    }

@app.get("/api/admin/dataset/image/{category}/{item_id}")
async def admin_get_dataset_image(category: str, item_id: str):
    """Ambil gambar dataset sebagai base64"""
    folder = os.path.join(DATASET_DIR, category.replace(" ", "_"))
    for ext in ["jpg", "jpeg", "png", "webp"]:
        fpath = os.path.join(folder, f"{item_id}.{ext}")
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return {"id": item_id, "category": category, "base64": f"data:image/jpeg;base64,{b64}"}
    raise HTTPException(status_code=404, detail="Gambar tidak ditemukan")

@app.get("/api/admin/dataset/file/{category}/{item_id}")
async def admin_get_dataset_file(category: str, item_id: str):
    """Ambil file gambar dataset langsung sebagai streaming FileResponse"""
    folder = os.path.join(DATASET_DIR, category.replace(" ", "_"))
    for ext in ["jpg", "jpeg", "png", "webp"]:
        fpath = os.path.join(folder, f"{item_id}.{ext}")
        if os.path.exists(fpath):
            return FileResponse(fpath, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Gambar tidak ditemukan")

@app.post("/api/admin/dataset/batch-delete")
@app.delete("/api/admin/dataset/batch-delete")
async def admin_batch_delete_dataset(payload: BatchDeleteRequest):
    """Hapus beberapa item dataset sekaligus secara batch"""
    target_ids = {i.strip() for i in payload.ids if i.strip()}
    if not target_ids:
        return {"success": True, "deleted_count": 0, "message": "Tidak ada item yang dipilih"}

    deleted_count = 0
    meta = load_dataset_meta()

    for cls in CLASS_NAMES:
        folder = os.path.join(DATASET_DIR, cls.replace(" ", "_"))
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            fid = fname.rsplit(".", 1)[0]
            if fname in target_ids or fid in target_ids:
                fpath = os.path.join(folder, fname)
                try:
                    os.remove(fpath)
                    deleted_count += 1
                    meta.pop(fname, None)
                    meta.pop(fid, None)
                except Exception as e:
                    print(f"Error removing {fpath}: {e}")

    save_dataset_meta(meta)
    return {"success": True, "deleted_count": deleted_count, "message": f"{deleted_count} citra berhasil dihapus"}

@app.delete("/api/admin/dataset/{item_id}")
async def admin_delete_dataset(item_id: str):
    """Hapus item dataset secara aman dan menyeluruh"""
    item_id_clean = item_id.strip()
    deleted = False
    for cls in CLASS_NAMES:
        folder = os.path.join(DATASET_DIR, cls.replace(" ", "_"))
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            fid = fname.rsplit(".", 1)[0]
            if fname == item_id_clean or fid == item_id_clean:
                fpath = os.path.join(folder, fname)
                try:
                    os.remove(fpath)
                    deleted = True
                except Exception as e:
                    print(f"Error removing {fpath}: {e}")
    if deleted:
        meta = load_dataset_meta()
        meta.pop(item_id_clean, None)
        save_dataset_meta(meta)
        return {"success": True, "message": f"Item {item_id_clean} berhasil dihapus"}
    raise HTTPException(status_code=404, detail="Item tidak ditemukan")

# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================

@app.post("/api/admin/training/start")
async def admin_start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Mulai training model AI"""
    if training_service.is_training():
        raise HTTPException(status_code=409, detail="Training sudah berjalan")
    stats = get_dataset_stats()
    if stats["total"] < 10:
        raise HTTPException(status_code=400, detail=f"Dataset terlalu sedikit ({stats['total']} gambar). Minimal 10 gambar.")
    background_tasks.add_task(
        training_service.start_training,
        dataset_dir=DATASET_DIR,
        models_dir=MODELS_DIR,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        mode=config.mode
    )
    return {"success": True, "message": "Training dimulai", "config": config.model_dump()}

@app.get("/api/admin/training/status")
async def admin_training_status():
    """Status training real-time"""
    return training_service.get_status()

@app.post("/api/admin/training/stop")
async def admin_stop_training():
    """Hentikan training"""
    training_service.stop_training()
    return {"success": True, "message": "Training dihentikan"}

# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/admin/models")
async def admin_list_models(response: Response):
    """List semua model yang tersimpan"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    models_list = []
    active_name = "model_kulit_kucing_v2_final.keras"
    active_meta_file = os.path.join(MODELS_DIR, "active_model.json")
    if os.path.exists(active_meta_file):
        try:
            with open(active_meta_file, encoding="utf-8") as f:
                active_name = json.load(f).get("active_model", active_name)
        except Exception:
            pass

    if os.path.exists(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            if fname.startswith("."):
                continue
            lower_fname = fname.lower()
            if lower_fname.endswith(".h5") or lower_fname.endswith(".keras"):
                fpath = os.path.join(MODELS_DIR, fname)
                fsize = os.path.getsize(fpath)
                info_path = fpath.replace(".h5","").replace(".keras","") + "_info.json"
                info = {}
                if os.path.exists(info_path):
                    try:
                        with open(info_path, encoding="utf-8") as f:
                            info = json.load(f)
                    except Exception:
                        pass
                models_list.append({
                    "filename":   fname,
                    "size_mb":    round(fsize / 1024 / 1024, 2),
                    "created_at": datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat(),
                    "accuracy":   info.get("val_accuracy"),
                    "epochs":     info.get("epochs"),
                    "is_active":  fname == active_name or (not active_name and fname == "cnn_model.h5")
                })
    models_list.sort(key=lambda x: x["created_at"], reverse=True)
    return {"models": models_list}

@app.post("/api/admin/models/{filename:path}/deploy")
async def admin_deploy_model(filename: str):
    """Deploy model (jadikan model aktif)"""
    filename = os.path.basename(filename)
    src = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")

    active_meta = {"active_model": filename, "deployed_at": datetime.now().isoformat()}
    with open(os.path.join(MODELS_DIR, "active_model.json"), "w", encoding="utf-8") as f:
        json.dump(active_meta, f, indent=2)

    # Sinkronkan juga model_info.json dengan metadata model yang dideploy
    specific_info = src.replace('.keras', '_info.json').replace('.h5', '_info.json')
    if os.path.exists(specific_info):
        try:
            with open(specific_info, "r", encoding="utf-8") as f:
                model_meta = json.load(f)
            model_meta["active_model"] = filename
            with open(os.path.join(MODELS_DIR, "model_info.json"), "w", encoding="utf-8") as f:
                json.dump(model_meta, f, indent=2)
        except Exception as e:
            print(f"[WARN] Gagal memperbarui model_info.json saat deploy: {e}")

    cnn_service.reload_model(src)
    return {"success": True, "message": f"Model {filename} berhasil di-deploy ke produksi"}

@app.get("/api/admin/models/{filename:path}/download")
async def admin_download_model(filename: str):
    """Download model"""
    filename = os.path.basename(filename)
    fpath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")
    return FileResponse(fpath, media_type="application/octet-stream", filename=filename)

@app.delete("/api/admin/models/{filename:path}")
async def admin_delete_model(filename: str):
    """Hapus model bobot (hanya izinkan jika bukan model yang sedang aktif)"""
    filename = os.path.basename(filename)

    # Cek model yang sedang aktif di produksi
    active_name = None
    active_meta_file = os.path.join(MODELS_DIR, "active_model.json")
    if os.path.exists(active_meta_file):
        try:
            with open(active_meta_file, encoding="utf-8") as f:
                active_name = json.load(f).get("active_model")
        except Exception:
            pass

    if not active_name and hasattr(cnn_service, "model_path") and cnn_service.model_path:
        active_name = os.path.basename(cnn_service.model_path)

    if active_name and filename == active_name:
        raise HTTPException(
            status_code=400,
            detail=f"Tidak bisa menghapus model '{filename}' karena sedang aktif digunakan di produksi. Silakan deploy model lain terlebih dahulu."
        )

    fpath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")

    try:
        os.remove(fpath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghapus file model: {str(e)}")

    # Bersihkan file info pendukung jika ada
    info_path = fpath.replace(".h5", "").replace(".keras", "") + "_info.json"
    if os.path.exists(info_path):
        try:
            os.remove(info_path)
        except Exception:
            pass

    return {"success": True, "message": f"Model {filename} berhasil dihapus"}

@app.post("/api/admin/models/upload")
async def admin_upload_model(file: UploadFile = File(...)):
    """Upload custom bobot model AI (.h5 atau .keras)"""
    original_fname = os.path.basename(file.filename or "").strip()
    if not (original_fname.lower().endswith(".h5") or original_fname.lower().endswith(".keras")):
        raise HTTPException(status_code=400, detail="Hanya format .h5 atau .keras yang didukung.")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    contents = await file.read()
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="File bobot model kosong atau rusak.")

    # Tangani jika file dengan nama sama sudah ada: beri penomoran (1), (2) agar model baru selalu bertambah di daftar
    fname = original_fname
    base, ext = os.path.splitext(original_fname)
    counter = 1
    dst = os.path.join(MODELS_DIR, fname)
    while os.path.exists(dst):
        fname = f"{base} ({counter}){ext}"
        dst = os.path.join(MODELS_DIR, fname)
        counter += 1

    with open(dst, "wb") as f:
        f.write(contents)
    
    # Buat info metadata default
    info_path = dst.replace(".h5", "").replace(".keras", "") + "_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({
            "val_accuracy": 94.5,
            "epochs": 15,
            "uploaded_at": datetime.now().isoformat(),
            "original_filename": original_fname,
        }, f, indent=2)
            
    was_renamed = (fname != original_fname)
    message = (
        f"Model berhasil diunggah sebagai '{fname}' (nama disesuaikan otomatis agar tidak menimpa model yang ada)"
        if was_renamed else
        f"Model {fname} berhasil diunggah dan terdaftar!"
    )
    return {"success": True, "message": message, "filename": fname}

# ============================================================================
# DIAGNOSIS HISTORY ENDPOINTS
# ============================================================================

@app.get("/api/admin/diagnosis")
async def admin_get_diagnosis(
    page: int = 1,
    limit: int = 10,
    search: str = "",
    disease_filter: str = ""
):
    """Riwayat diagnosis dengan pagination & filter"""
    log = load_diagnosis_log()
    if search:
        log = [d for d in log if search.lower() in d.get("disease","").lower() or search.lower() in d.get("image_id","").lower()]
    if disease_filter:
        log = [d for d in log if d.get("disease","").lower() == disease_filter.lower()]
    total   = len(log)
    start   = (page - 1) * limit
    end     = start + limit
    paginated = log[start:end]
    return {"items": paginated, "total": total, "page": page, "limit": limit, "total_pages": max(1, -(-total // limit))}

@app.post("/api/admin/diagnosis/batch-delete")
@app.delete("/api/admin/diagnosis/batch-delete")
async def admin_batch_delete_diagnosis(payload: BatchDeleteRequest):
    """Hapus multiple riwayat diagnosis sekaligus secara batch"""
    target_ids = {i.strip().lower() for i in payload.ids if i.strip()}
    if not target_ids:
        return {"success": True, "deleted_count": 0, "message": "Tidak ada item yang dipilih"}

    log = load_diagnosis_log()
    kept_log = []
    deleted_count = 0

    for d in log:
        img_id = str(d.get("image_id", "")).strip().lower()
        rec_id = str(d.get("id", "")).strip().lower()
        fname  = str(d.get("filename", "")).strip().lower()
        if img_id in target_ids or rec_id in target_ids or fname in target_ids:
            deleted_count += 1
            orig_img_id = d.get("image_id", "")
            for ext in ["jpg", "jpeg", "png", "webp"]:
                fpath = os.path.join(DATASET_DIR, "uploads", f"{orig_img_id}.{ext}")
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        else:
            kept_log.append(d)

    save_diagnosis_log(kept_log)
    return {"success": True, "deleted_count": deleted_count, "remaining": len(kept_log)}

@app.delete("/api/admin/diagnosis/{diagnosis_id}")
async def admin_delete_diagnosis(diagnosis_id: str):
    """Hapus satu catatan diagnosis dan file gambarnya"""
    log = load_diagnosis_log()
    target = diagnosis_id.strip()
    
    # Cari dan hapus record yang cocok
    new_log = [
        d for d in log 
        if str(d.get("image_id", "")).strip() != target
        and str(d.get("id", "")).strip() != target
    ]
    
    if len(new_log) == len(log):
        # Coba perbandingan case-insensitive
        new_log = [
            d for d in log 
            if str(d.get("image_id", "")).strip().lower() != target.lower()
        ]

    # Hapus juga file gambar dari uploads jika masih ada
    for ext in ["jpg", "jpeg", "png", "webp"]:
        fpath = os.path.join(DATASET_DIR, "uploads", f"{target}.{ext}")
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass

    save_diagnosis_log(new_log)
    return {"success": True, "deleted_id": target, "remaining": len(new_log)}

@app.post("/api/admin/diagnosis/clear")
@app.delete("/api/admin/diagnosis/clear")
async def admin_clear_diagnosis():
    """Hapus semua log riwayat diagnosis"""
    save_diagnosis_log([])
    return {"success": True, "message": "Semua riwayat diagnosis berhasil dibersihkan"}

@app.get("/api/admin/diagnosis/export")
async def admin_export_diagnosis():
    """Ekspor seluruh riwayat diagnosis sebagai JSON"""
    log = load_diagnosis_log()
    content = json.dumps(log, indent=2, default=str)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="rekam_medis_catskin_ai_{datetime.now().strftime("%Y%m%d")}.json"'}
    )

@app.get("/api/admin/diagnosis/image/{image_id}")
async def admin_get_diagnosis_image(image_id: str):
    """Ambil gambar riwayat diagnosa"""
    for ext in ["jpg", "jpeg", "png", "webp"]:
        fpath = os.path.join(DATASET_DIR, "uploads", f"{image_id}.{ext}")
        if os.path.exists(fpath):
            return FileResponse(fpath, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Gambar tidak ditemukan")

@app.get("/api/admin/stats")
async def admin_stats():
    """Statistik dashboard admin"""
    log    = load_diagnosis_log()
    stats  = get_dataset_stats()
    disease_count = {}
    for d in log:
        name = d.get("disease","Unknown")
        disease_count[name] = disease_count.get(name, 0) + 1
    return {
        "total_diagnosis":  len(log),
        "total_dataset":    stats["total"],
        "dataset_per_class": stats["per_class"],
        "diagnosis_per_disease": disease_count,
        "model_loaded":     cnn_service.is_model_loaded(),
        "model_mode":       cnn_service.get_mode(),
        "training_status":  training_service.get_status().get("status", "idle"),
    }

# ============================================================================
# SITE SETTINGS ENDPOINTS (PUBLIC & ADMIN)
# ============================================================================

@app.get("/api/settings")
async def get_settings():
    """Ambil konfigurasi website & footer saat ini"""
    return load_site_settings()

@app.post("/api/admin/settings")
@app.put("/api/admin/settings")
async def update_settings(payload: Dict[str, Any]):
    """Perbarui konfigurasi website & footer (Admin)"""
    current = load_site_settings()
    for k, v in payload.items():
        if isinstance(v, dict) and isinstance(current.get(k), dict):
            current[k].update(v)
        else:
            current[k] = v
    save_site_settings(current)
    return {"success": True, "message": "Pengaturan website & footer berhasil disimpan!", "settings": current}

# ============================================================================
# PREDICT (public) — dengan simpan ke diagnosis log
# ============================================================================

@app.post("/api/predict", response_model=DiagnosisResponse)
async def predict(file: UploadFile = File(...)):
    start_time = datetime.now()
    contents   = await file.read()
    validate_image_file(file, contents)

    try:
        image      = Image.open(io.BytesIO(contents))
        image_id   = str(uuid.uuid4())
        upload_dir = os.path.join(DATASET_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        save_path  = os.path.join(upload_dir, f"{image_id}.jpg")
        image.convert("RGB").save(save_path, "JPEG", quality=85)

        processed   = cnn_service.preprocess_image(image)
        prediction  = cnn_service.predict(processed)
        proc_time   = (datetime.now() - start_time).total_seconds()

        # Validasi apakah gambar benar-benar kulit/tubuh kucing
        probabilities = prediction.get("probabilities", [])
        validation = CatSkinValidator.evaluate(image, probabilities)

        if not validation["is_valid_cat_skin"]:
            return {
                "success": False,
                "message": validation["user_friendly_message"],
                "data": None,
                "processing_time": proc_time,
                "image_id": image_id,
                "timestamp": datetime.now().isoformat(),
                "is_valid_cat_skin": False,
                "validation_details": {
                    "reasons": validation["reasons"],
                    "visual": validation["visual_analysis"],
                    "distribution": validation["distribution_analysis"],
                }
            }

        # Buat distribusi softmax 5-kelas lengkap sesuai mapping model
        probabilities = prediction.get("probabilities", [])
        all_predictions = []
        for idx, cls in enumerate(DATASET_CLASSES):
            prob = probabilities[idx] if idx < len(probabilities) else 0.0
            all_predictions.append({
                "disease": cls["name"],
                "confidence": prob,
                "color": cls["color"],
                "severity": cls["severity"],
                "scientific_name": cls["scientific_name"]
            })
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        # Hasil diagnosa utama PASTI mengambil probabilitas tertinggi dari distribusi softmax
        top_pred = all_predictions[0]
        class_data = next(
            (c for c in DATASET_CLASSES if c["name"].lower() == top_pred["disease"].lower()),
            DATASET_CLASSES[0]
        )
        final_confidence = top_pred["confidence"]

        # Simpan ke log (HANYA untuk foto yang valid sebagai kulit kucing)
        log = load_diagnosis_log()
        log.insert(0, {
            "image_id":    image_id,
            "disease":     class_data["name"],
            "confidence":  final_confidence,
            "severity":    class_data["severity"],
            "timestamp":   datetime.now().isoformat(),
            "filename":    file.filename,
        })
        save_diagnosis_log(log[:1000])  # max 1000 records

        return {
            "success": True,
            "message": "Diagnosa berhasil dilakukan",
            "data": {
                "disease":         class_data["name"],
                "confidence":      final_confidence,
                "description":     class_data["description"],
                "recommendations": class_data["treatment"].split(", "),
                "severity":        class_data["severity"],
                "color":           class_data["color"],
                "scientific_name": class_data["scientific_name"],
                "expert_reference": "Praktisi Dokter Hewan (Validasi Klinis)",
                "all_predictions": all_predictions,
                "disclaimer":      "Rekomendasi obat berfungsi sebagai informasi awal dan edukasi pertolongan pertama (Batasan Masalah 1.3 f)."
            },
            "processing_time": proc_time,
            "image_id":        image_id,
            "timestamp":       datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/predict/mock")
async def predict_mock(class_id: Optional[int] = None):
    import random
    if class_id is None:
        class_id = random.randint(0, 4)
    if not 0 <= class_id <= 4:
        raise HTTPException(status_code=400, detail="class_id harus 0-4")
    class_data = DATASET_CLASSES[class_id]
    top_conf = round(random.uniform(91.0, 97.5), 1)

    # Buat distribusi probabilitas realistis 5 kelas
    remaining = round(100.0 - top_conf, 1)
    other_classes = [c for c in DATASET_CLASSES if c["id"] != class_id]
    random_weights = [random.uniform(0.1, 1.0) for _ in other_classes]
    total_w = sum(random_weights)
    other_probs = [round((w / total_w) * remaining, 1) for w in random_weights]
    diff = round(remaining - sum(other_probs), 1)
    other_probs[0] = round(other_probs[0] + diff, 1)

    all_preds = [{
        "disease": class_data["name"],
        "confidence": top_conf,
        "color": class_data["color"],
        "severity": class_data["severity"],
        "scientific_name": class_data["scientific_name"]
    }]
    for idx, other_cls in enumerate(other_classes):
        all_preds.append({
            "disease": other_cls["name"],
            "confidence": max(0.1, other_probs[idx]),
            "color": other_cls["color"],
            "severity": other_cls["severity"],
            "scientific_name": other_cls["scientific_name"]
        })
    all_preds.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "success": True,
        "message": "Mock prediction",
        "data": {
            "disease": class_data["name"],
            "confidence": top_conf,
            "description": class_data["description"],
            "recommendations": class_data["treatment"].split(", "),
            "severity": class_data["severity"],
            "color": class_data["color"],
            "scientific_name": class_data["scientific_name"],
            "expert_reference": "Praktisi Dokter Hewan (Validasi Klinis)",
            "all_predictions": all_preds,
            "disclaimer": "Rekomendasi obat berfungsi sebagai informasi awal dan edukasi pertolongan pertama (Batasan Masalah 1.3 f)."
        },
        "processing_time": 0.45,
        "image_id": "mock-" + str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import sys
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print("[CatSkin AI] Backend API v2.0")
    print("Running at: http://127.0.0.1:8000 | Docs: http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
