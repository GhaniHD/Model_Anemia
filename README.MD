# Prediksi Anemia (Anemia Prediction)

Model ini dikembangkan untuk memprediksi status anemia berdasarkan beberapa parameter darah yang relevan. Model telah melalui proses *preprocessing* data yang komprehensif, pelatihan menggunakan Neural Network (TensorFlow/Keras), dan diekspor ke format ONNX untuk inferensi yang efisien di lingkungan produksi.

---

## 1.*Frontend*

### Input Data yang Dibutuhkan

Model membutuhkan data dalam format spesifik. Berikut adalah fitur-fitur yang diharapkan beserta tipe datanya:

| Fitur     | Tipe Data Asli       | Contoh       | Keterangan                                      | Cara Input              |
| :-------- | :------------------- | :----------- | :---------------------------------------------- | :---------------------- |
| `HGB`     | Numerik (Float)      | `14.5`       | Kadar Hemoglobin (g/dL).                        | Input angka desimal     |
| `HCT`     | Numerik (Float)      | `42.0`       | Hematokrit (%).                                 | Input angka desimal     |
| `RBC`     | Numerik (Float)      | `5.0`        | Jumlah Sel Darah Merah (juta/µL).               | Input angka desimal     |
| `RDW`     | Numerik (Float)      | `11.5`       | Distribusi Lebar Sel Darah Merah (%).           | Input angka desimal     |
| `MCH`     | Numerik (Float)      | `29.0`       | Kadar Hemoglobin Mean Corpuscular (pg).         | Input angka desimal     |
| `MCHC`    | Numerik (Float)      | `33.0`       | Kadar Hemoglobin Mean Corpuscular Konsentrasi (g/dL). | Input angka desimal |
| `MCV`     | Numerik (Float)      | `89.0`       | Volume Sel Darah Merah Mean Corpuscular (fL).   | Input angka desimal     |
| `SD`      | Numerik (Float)      | `40.0`       | Standar Deviasi (terkait RDW atau parameter lain). | Input angka desimal |
| `TSD`     | Numerik (Float)      | `50.0`       | Total Standar Deviasi (terkait distribusi).     | Input angka desimal     |

---

### Output Prediksi

*Backend* akan mengembalikan hasil prediksi yang bisa berupa:
* **Probabilitas:** Angka antara 0 dan 1 untuk setiap kelas (misalnya, probabilitas tidak terkena anemia, probabilitas terkena anemia).
* **Label Prediksi:** `Tidak Terkena Anemia` atau `Terkena Anemia`.

Disarankan untuk menampilkan probabilitas dan label prediksi untuk memberikan informasi yang lebih lengkap kepada pengguna.

---

## 2. *Backend*

Tim *backend* bertanggung jawab untuk menerima data dari *frontend*, melakukan *preprocessing* sesuai dengan pipeline model, menjalankan inferensi menggunakan model ONNX, dan mengembalikan hasilnya.

### Struktur Proyek Model

Model dan objek yang diperlukan untuk *preprocessing* disimpan dalam direktori `models/`.

```
models/
├── anemia_binary_model.keras     # Model Keras asli (untuk referensi/pengembangan)
├── anemia_binary_model.onnx       # Model dalam format ONNX (digunakan untuk inferensi)
└── encoders/
    └── scaler.pkl                # StandardScaler untuk fitur numerik
```

*Catatan*: Karena data anemia dalam contoh Anda tidak mencakup fitur kategorikal, tidak ada `LabelEncoder` yang disimpan untuk kolom kategorikal.

---

### Dependensi Python

Pastikan lingkungan *backend* Anda memiliki dependensi berikut terinstal:
* `pandas`
* `numpy`
* `scikit-learn`
* `tensorflow` (jika Anda perlu memuat model .keras atau untuk referensi)
* `onnxruntime`
* `tf2onnx` (jika Anda perlu mengkonversi ulang model)

Anda bisa menginstal ini menggunakan `pip`:
```bash
pip install pandas numpy scikit-learn tensorflow onnxruntime tf2onnx
```

---

### Alur Kerja Inferensi di *Backend*

Berikut adalah langkah-langkah yang harus dilakukan oleh *backend* untuk memproses permintaan prediksi:

1. **Muat Model dan Objek Preprocessing:**
   * Muat model ONNX: `onnxruntime.InferenceSession('models/anemia_binary_model.onnx')`
   * Muat `StandardScaler` dari `models/encoders/scaler.pkl`.

   **Contoh loading:**
   ```python
   import pickle
   import onnxruntime as ort

   # Load ONNX model
   ort_session = ort.InferenceSession("models/anemia_binary_model.onnx")

   # Load scaler
   with open('models/encoders/scaler.pkl', 'rb') as f:
       scaler = pickle.load(f)
   ```

2. **Terima Data Mentah dari *Frontend***:
   Data akan diterima biasanya dalam format JSON dari permintaan API.

   **Contoh data yang diterima:**
   ```python
   new_data = {
       'HGB': 14.5, 'HCT': 42.0, 'RBC': 5.0, 'RDW': 11.5,
       'MCH': 29.0, 'MCHC': 33.0, 'MCV': 89.0, 'SD': 40.0, 'TSD': 50.0
   }
   ```

3. **Preprocessing Data Input:**
   * **Buat DataFrame:** Konversi data input menjadi Pandas DataFrame.
   * **Handling Missing Values:** Isi nilai yang hilang dengan median dari data pelatihan (dari `scaler` atau data pelatihan sebelumnya).
   * **Scaling Numerik:** Gunakan `StandardScaler` yang telah dimuat untuk mentransformasi semua fitur numerik.
     ```python
     import pandas as pd
     import numpy as np

     df_new = pd.DataFrame([new_data]) # Pastikan ini adalah list of dicts
     expected_columns = ['HGB', 'HCT', 'RBC', 'RDW', 'MCH', 'MCHC', 'MCV', 'SD', 'TSD']
     df_new = df_new[expected_columns]  # Pastikan urutan kolom sesuai

     # Isi nilai yang hilang dengan median pelatihan (jika ada)
     df_new = df_new.fillna(df_new.median())

     # Scaling
     df_new[expected_columns] = scaler.transform(df_new[expected_columns])
     input_data_processed = df_new[expected_columns].to_numpy().astype(np.float32)
     ```

4. **Jalankan Inferensi dengan ONNX Runtime:**
   ```python
   input_name = ort_session.get_inputs()[0].name
   outputs = ort_session.run(None, {input_name: input_data_processed})
   y_pred_prob = outputs[0] # Probabilitas output
   ```

5. **Proses Output Prediksi:**
   * **Dapatkan label prediksi:** `y_pred_label = np.argmax(y_pred_prob, axis=1)`
   * **Dekode label ke teks:** Gunakan mapping sederhana (misalnya, `0: "Tidak Terkena Anemia"`, `1: "Terkena Anemia"`) karena target adalah biner.
     ```python
     y_pred_label = np.argmax(y_pred_prob, axis=1)
     label_mapping = {0: "Tidak Terkena Anemia", 1: "Terkena Anemia"}
     final_prediction = [label_mapping[label] for label in y_pred_label]
     ```

6. **Kirim Hasil Prediksi ke *Frontend***:
   Kembalikan probabilitas dan label prediksi yang telah didekode ke *frontend* dalam format JSON.

   **Contoh output JSON:**
   ```json
   {
       "prediction_label": "Tidak Terkena Anemia",
       "prediction_probability": [0.95, 0.05] // Probabilitas untuk [Tidak Terkena Anemia, Terkena Anemia]
   }
   ```

---

### Catatan Penting

* **Konsistensi Data:** Pastikan data yang dikirim dari *frontend* dan diproses di *backend* konsisten dengan data yang digunakan untuk melatih model. Perbedaan dalam jumlah atau urutan kolom bisa menyebabkan hasil prediksi yang tidak akurat.
* **Penanganan Error:** Implementasikan penanganan error yang robust untuk kasus di mana data input tidak valid atau ada masalah saat memuat model/scaler.
* **Versi Dependensi:** Usahakan untuk menggunakan versi library yang sama antara lingkungan pelatihan model dan lingkungan produksi untuk menghindari masalah kompatibilitas.
* **Optimasi ONNX:** ONNX Runtime sangat efisien. Anda bisa memanfaatkan *session options* untuk optimasi lebih lanjut jika diperlukan.

---

