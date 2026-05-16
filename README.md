# Handwritten Character Classification (EMNIST Letters) using HOG and SVM

This project is an implementation of a Machine Learning pipeline to classify handwritten characters from the EMNIST (Extended MNIST) dataset, specifically focusing on letters (A-Z). This project was developed as a Midterm Project for the Machine Vision course 

The algorithm focuses on feature extraction using **Histogram of Oriented Gradients (HOG)** and classification using **Support Vector Machine (SVM)** with parameter optimization.

## 📊 Dataset
The dataset used is the **EMNIST Letters** in CSV format. 
- Consists of 26 classes (Letters A to Z).
- To prevent model bias, the dataset is strictly balanced to **2600 samples** (exactly 100 random samples for each class).
- The data is split into **80% Training** (2080 images) and **20% Testing** (520 images) using a *stratified shuffle split*.

## ⚙️ Pipeline

### 1. Image Preprocessing
The raw images in the EMNIST dataset have a resolution of 28x28 pixels and are stored rotated by 90 degrees and flipped by default.
- **Rotation Correction:** Each 1D array is reshaped into a 28x28 matrix, followed by a **Transpose (`.T`)** operation to ensure the letter images stand perfectly upright before extraction.

### 2. Feature Extraction (HOG)
Because the image resolution is very low (28x28), the HOG parameters were fine-tuned with high precision so as not to lose the details of the character curves:
- `orientations` = 9
- `pixels_per_cell` = (4, 4)
- `cells_per_block` = (2, 2)
- *Output:* 1296 feature dimensions per image.

### 3. Classification and Tuning (SVM)
The search for the best weight parameters was conducted using **GridSearchCV** (with a 5-fold Cross-Validation).
- The best parameters found: `C=10`, `gamma='scale'`, `kernel='rbf'`.
- The RBF (*Radial Basis Function*) kernel proved to be the most robust in separating non-linear image data patterns in a high-dimensional space.

## 📈 Evaluation Metrics and Results

The model was strictly evaluated using validation testing during the training phase and testing on unseen data.

- **LOOCV (Leave-One-Out Cross-Validation) Evaluation on Training Data:**
  - Accuracy: **84.2%**
  - Precision: **84.4%**
  - Recall: **84.2%**
  - F1-Score: **84.2%**

- **Final Testing Evaluation (20% Unseen Data):**
  - Accuracy: **81.7%**
  - Precision: **82.3%**
  - Recall: **81.7%**
  - F1-Score: **81.7%**

The very close gap between the training and testing performance proves that this model generalizes well and successfully avoids *overfitting*.
