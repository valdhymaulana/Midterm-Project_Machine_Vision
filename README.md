# Klasifikasi Karakter Tulisan Tangan (EMNIST Letters) menggunakan HOG dan SVM

Proyek ini adalah implementasi *pipeline Machine Learning* untuk mengklasifikasikan karakter tulisan tangan dari dataset EMNIST (Extended MNIST) khusus huruf (A-Z). Proyek ini dikembangkan sebagai pemenuhan tugas Ujian Tengah Semester mata kuliah Machine Vision oleh Valdhy Maulana Alzikri.

Algoritma yang digunakan berfokus pada ekstraksi fitur menggunakan **Histogram of Oriented Gradients (HOG)** dan klasifikasi menggunakan **Support Vector Machine (SVM)** dengan optimasi parameter.

## 📊 Dataset
Dataset yang digunakan adalah **EMNIST Letters** format CSV. 
- Terdiri dari 26 kelas (Huruf A sampai Z).
- Untuk mencegah bias model, dataset diseimbangkan secara ketat menjadi **2600 sampel** (tepat 100 sampel acak untuk masing-masing kelas).
- Data dibagi dengan proporsi **80% Training** (2080 citra) dan **20% Testing** (520 citra) menggunakan *stratified shuffle split*.

## ⚙️ Alur Kerja (Pipeline)

### 1. Pra-pemrosesan Citra (Image Preprocessing)
Citra mentah pada dataset EMNIST beresolusi 28x28 piksel dan secara *default* tersimpan dalam kondisi terotasi 90 derajat serta terbalik (*flipped*).
- **Koreksi Rotasi:** Setiap larik (*array*) 1D diubah bentuknya (*reshape*) menjadi matriks 28x28, kemudian dilakukan operasi **Transpose (`.T`)** agar citra huruf berdiri tegak sempurna sebelum diekstraksi.

### 2. Ekstraksi Fitur (HOG)
Karena resolusi citra sangat rendah (28x28), parameter HOG di-*tuning* pada tingkat presisi tinggi agar tidak kehilangan detail lengkungan karakter:
- `orientations` = 9
- `pixels_per_cell` = (4, 4)
- `cells_per_block` = (2, 2)
- *Output:* 1296 dimensi fitur per citra.

### 3. Klasifikasi dan Tuning (SVM)
Pencarian parameter bobot terbaik dilakukan menggunakan **GridSearchCV** (dengan *Cross-Validation* = 5 fold).
- Parameter terbaik yang ditemukan: `C=10`, `gamma='scale'`, `kernel='rbf'`.
- Kernel RBF (*Radial Basis Function*) terbukti paling tangguh dalam memisahkan pola data citra non-linear pada ruang dimensi tinggi.

## 📈 Metrik dan Hasil Evaluasi

Model dievaluasi secara ketat menggunakan pengujian validasi saat masa latih dan pengujian pada data yang belum pernah dilihat sebelumnya.

- **Evaluasi LOOCV (Leave-One-Out Cross-Validation) pada Data Latih:**
  - Accuracy: **84.2%**
  - Precision: **84.4%**
  - Recall: **84.2%**
  - F1-Score: **84.2%**

- **Evaluasi Uji Akhir (Testing Data 20%):**
  - Accuracy: **81.7%**
  - Precision: **82.3%**
  - Recall: **81.7%**
  - F1-Score: **81.7%**

Jarak yang sangat rapat antara performa *training* dan *testing* membuktikan bahwa model ini sangat general dan terhindar dari fenomena *overfitting*.
