# CatSkin AI - Backend

Backend API untuk Sistem Diagnosa Penyakit Kulit Kucing menggunakan CNN.

## 📁 Struktur Folder

```
backend/
├── app/
│   ├── __init__.py
│   └── cnn_service.py       # CNN Service untuk prediksi
├── training/
│   └── train_cnn.py         # Script training model CNN
├── dataset/
│   └── uploads/             # Folder untuk menyimpan gambar yang diupload
├── models/
│   └── cnn_model.h5         # Model CNN (generate dari training)
├── main.py                  # FastAPI main application
├── requirements.txt         # Dependencies
└── README.md               # Dokumentasi ini
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Jalankan Server

```bash
python main.py
```

Server akan berjalan di: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

## 📊 Dataset 5 Kelas

| ID | Nama | Nama Ilmiah | Severity |
|----|------|-------------|----------|
| 0 | Dermatitis | Feline Dermatitis | Sedang |
| 1 | Flea Allergy | Flea Allergy Dermatitis | Tinggi |
| 2 | Ringworm | Dermatophytosis | Sedang |
| 3 | Scabies | Sarcoptic Mange | Tinggi |
| 4 | Healthy | Normal Skin Condition | Normal |

## 🔌 API Endpoints

### Health Check
```
GET /api/health
```

### Dataset Info
```
GET /api/dataset/info
```

### Predict (Diagnosa)
```
POST /api/predict
Content-Type: multipart/form-data

file: <image_file>
```

### Mock Predict (Testing)
```
POST /api/predict/mock?class_id=0
```

## 🧠 Training Model CNN

### 1. Siapkan Dataset

Struktur folder:
```
dataset/
├── Dermatitis/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Flea Allergy/
├── Ringworm/
├── Scabies/
└── Healthy/
```

Minimal 50-100 gambar per kelas untuk hasil yang baik.

### 2. Jalankan Training

```bash
cd training
python train_cnn.py
```

Model akan disimpan di `models/cnn_model.h5`

### 3. Arsitektur Model

- **Base**: MobileNetV2 (Transfer Learning)
- **Input**: 224x224x3
- **Output**: 5 kelas (Softmax)
- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Categorical Crossentropy

## 🔧 Mode Operasi

### MOCK Mode (Default)
Jika model belum dilatih, sistem akan menggunakan random prediction untuk demo.

### REAL Mode
Setelah model dilatih dan disimpan di `models/cnn_model.h5`, sistem akan menggunakan model CNN untuk prediksi nyata.

## 📝 Contoh Penggunaan

### Python
```python
import requests

url = "http://localhost:8000/api/predict"
files = {"file": open("cat_skin.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Penyakit: {result['data']['disease']}")
print(f"Confidence: {result['data']['confidence']}%")
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@cat_skin.jpg"
```

### JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/api/predict', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => {
  console.log('Hasil:', data.data.disease);
  console.log('Akurasi:', data.data.confidence);
});
```

## 🐳 Docker (Opsional)

### Build Image
```bash
docker build -t catskin-api .
```

### Run Container
```bash
docker run -p 8000:8000 catskin-api
```

## ⚙️ Environment Variables

Buat file `.env`:
```
MODEL_PATH=models/cnn_model.h5
DATASET_PATH=dataset
UPLOAD_PATH=dataset/uploads
```

## 📈 Monitoring

Cek status training:
```
GET /api/training/status
```

## 🆘 Troubleshooting

### Error: "Model file not found"
- Model belum dilatih. Jalankan `python training/train_cnn.py` terlebih dahulu.
- Atau sistem akan otomatis menggunakan MOCK mode.

### Error: "No module named 'tensorflow'"
- Install dependencies: `pip install -r requirements.txt`
- Atau gunakan virtual environment.

### Low Accuracy
- Tambah jumlah gambar per kelas (minimal 100)
- Cek kualitas gambar (resolusi, pencahayaan)
- Tambah epochs training
- Coba fine-tuning dengan unfreeze layer

## 📚 Referensi

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

---

**Catatan**: Untuk production, gunakan model yang sudah dilatih dengan dataset yang representatif.
