# 🎶 Music Genre Classification (GTZAN Dataset)

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Modeling-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-green.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_NOTEBOOK_LINK)

---

## 📂 Dataset Description
- **Dataset:** [GTZAN Genre Collection](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection)  
- **Size:** 1000 audio tracks (`.wav`, 30 seconds each)  
- **Genres:** 10 categories (100 tracks per genre)  
- **Features Used:** MFCC, Chroma, Tempo, Key, Loudness  

---

## 🔧 Approach
1. **Feature Extraction** using `librosa` (MFCC, Chroma, Tempo).  
2. **Machine Learning Classifiers**:  
   - 📊 Logistic Regression  
   - 🌲 Random Forest  
   - 📈 Support Vector Machine (SVM)  
3. **Deep Learning (CNN)** for raw spectrogram-based classification.  

---

## 📊 Results & Observations
- 🌲 **Random Forest** achieved the **highest accuracy** among classical ML methods.  
- 🎼 **Classical music** was consistently well-classified (19/20 correct).  
- 🎸 **Rock** & 💃 **Disco** were harder to distinguish (≤ 12 correct predictions).  

| Model                | Accuracy | Key Notes |
|-----------------------|----------|-----------|
| Logistic Regression   | 74%      | Baseline |
| Random Forest         | 77.5%      | Best ML model |
| SVM                   | 68%      | Moderate performance |

We can notice that the accuracy is not exactly high, hence we can improve this by utilising CNNs next.
---

## 🚀 Usage
Clone the repo and install dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/music-genre-classification.git
cd music-genre-classification
pip install -r requirements.txt
