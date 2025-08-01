# 🧠 Face Recognition using Eigenfaces + ANN (with Impostor Detection)

This project implements a face recognition system using the **Eigenfaces technique** (PCA-based) for feature extraction and an **Artificial Neural Network (ANN)** for classification. It also includes **impostor detection** based on confidence thresholding.

---

## 📌 Features

- 🧠 Face recognition using Eigenfaces (PCA)
- 🤖 Classification using ANN (`MLPClassifier`)
- 👥 Supports multiple persons and folder structures
- 🚨 Impostor detection using prediction confidence threshold
- 📈 Accuracy comparison for multiple `k` values (top eigenfaces)
- 📊 Visualizations: mean face, eigenfaces, confidence plots, and accuracy graphs
- 📁 Dataset auto-download from GitHub if not provided

---

## 🧰 Technologies Used

| Library         | Purpose                                  |
|-----------------|------------------------------------------|
| `NumPy`         | Matrix and array operations              |
| `OpenCV`        | Image reading and resizing               |
| `scikit-learn`  | ANN, train-test split, evaluation        |
| `matplotlib`    | Plotting and visualizations              |
| `requests`      | Dataset downloading                      |
| `zipfile`       | Handling zipped datasets                 |
| `logging`       | Logging process steps                    |

---

## 🧠 How It Works

1. **Data Loading**
   - Dataset is downloaded from GitHub or provided manually
   - Supports various folder structures

2. **Preprocessing**
   - Converts images to grayscale
   - Resizes to 100x100
   - Flattens to 1D vectors
   - Computes mean face and performs mean normalization

3. **Feature Extraction**
   - Calculates covariance matrix
   - Extracts eigenfaces using PCA
   - Projects images into eigenface space (signature generation)

4. **Model Training**
   - Trains an ANN (2 hidden layers) on face signatures

5. **Evaluation**
   - Tests on 40% of the data
   - Computes accuracy for multiple `k` values
   - Plots accuracy vs `k`

6. **Impostor Detection**
   - Adds random noise images as impostors
   - Detects them if prediction confidence < 0.6
   - Plots confidence distribution

7. **Prediction**
   - Can predict new images using `predict_identity()` method

---

## 🚀 How to Run

### ▶️ Run the Complete Pipeline

```bash
python abarna.py
```

This will:
- Download and load dataset
- Train the model
- Evaluate performance
- Generate plots and save them

---

## 🖼️ Output Visuals

- `accuracy_vs_k.png` – Accuracy comparison for different k values
- `confidence_distribution.png` – Confidence of genuine vs impostors
- `mean_face.png` – Average face image
- `eigenfaces.png` – Top eigenfaces used

---

## 🧪 Predict a New Image

Once the model is trained:

```python
from abarna import FaceRecognition

face_rec = FaceRecognition()
face_rec.run()  # First, run the training
result = face_rec.predict_identity("path_to_your_image.jpg")
print("Prediction:", result)
```

---

## 📁 Dataset Used

By default, the dataset is downloaded from this link:

👉 [Download Dataset (.zip)](https://github.com/robaita/introduction_to_machine_learning/raw/main/dataset.zip)

You can replace it with your own folder structure:
```
dataset/
    ├── person1/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── person2/
    │   └── ...
```

---

## 📊 Sample Results

- ✅ Best Accuracy: ~**94–98%** depending on the dataset and `k` value
- 🚨 Impostor detection rate: ~**90%+** with thresholding

---



---

## 📜 License

This project is licensed under the MIT License.

