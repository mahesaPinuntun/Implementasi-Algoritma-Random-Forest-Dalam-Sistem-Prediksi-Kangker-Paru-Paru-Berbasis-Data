# Implementasi Algoritma Random Forest dalam Sistem Prediksi Kanker Paru-Paru Berbasis Data

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![RandomForest](https://img.shields.io/badge/Model-Random%20Forest-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-94.5%25-brightgreen)

## 📋 Deskripsi
Sistem prediksi tingkat risiko kanker paru-paru berbasis web menggunakan 
algoritma **Random Forest Classifier** dengan akurasi **94.5%**.
Sistem ini mengklasifikasikan pasien ke dalam tiga kategori risiko:
**Risiko Tinggi**, **Risiko Sedang**, dan **Risiko Rendah**.

## 🧠 Model Machine Learning
| Parameter | Nilai |
|---|---|
| Algoritma | Random Forest Classifier |
| Jumlah Pohon | 200 |
| Max Depth | 10 |
| Akurasi Test | 94.5% |
| CV Mean (5-fold) | 93.9% |
| Overfit Gap | 0.25% (sehat) |

## 📊 Dataset
- **Sumber**: Kaggle — Cancer Patient Data Sets
- **Jumlah data**: 1.000 pasien
- **Fitur**: 23 atribut (usia, jenis kelamin, paparan lingkungan, gejala klinis)
- **Target**: Low / Medium / High risk

## 🛠️ Teknologi
- Python 3.x
- Flask (web framework)
- scikit-learn (machine learning)
- Vue.js 3 (frontend interaktif)
- Tailwind CSS (styling)
- joblib (model serialization)

## 🚀 Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/mahesaPinuntun/Implementasi-Algoritma-Random-Forest-Dalam-Sistem-Prediksi-Kangker-Paru-Paru-Berbasis-Data.git
cd Implementasi-Algoritma-Random-Forest-Dalam-Sistem-Prediksi-Kangker-Paru-Paru-Berbasis-Data
```

### 2. Buat virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate    # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train model (generate joblib files)
```bash
python start.py
```

### 5. Jalankan aplikasi
```bash
python app.py
```

### 6. Buka browser