# Histopathologic Cancer Detection

<div align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-blue" alt="Deep Learning"/>
  <img src="https://img.shields.io/badge/Medical%20AI-Pathology-green" alt="Medical AI"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Python-3.8+-yellow" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-red" alt="License"/>
</div>

## 🎯 Project Overview

This project develops deep learning models to automatically identify metastatic cancer in histopathologic image patches. Using Convolutional Neural Networks (CNNs), we aim to assist pathologists in cancer detection by automating the analysis of tissue slides, potentially leading to faster and more accurate diagnoses.

### 🔬 Problem Statement

- **Task**: Binary classification of 96x96 pixel histopathology image patches
- **Objective**: Detect presence of metastatic tissue in the central region of image patches
- **Dataset**: PatchCamelyon (PCam) benchmark dataset
- **Evaluation Metric**: Area Under the ROC Curve (AUC)

## 📊 Dataset

The dataset consists of histopathologic scans from the PatchCamelyon benchmark:

- **Training Images**: 220,000+ labeled image patches
- **Test Images**: 57,000+ image patches
- **Image Size**: 96x96 pixels (RGB)
- **Classes**: 
  - `0`: No metastatic tissue (negative)
  - `1`: Contains metastatic tissue (positive)
- **Source**: Camelyon16 and Camelyon17 challenges

### Class Distribution
The dataset exhibits class imbalance typical of medical datasets, with fewer positive (cancerous) samples than negative samples.

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
OpenCV (cv2)
Pillow
tqdm
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/mafiat/Histopathologic-Cancer-Detection.git
cd Histopathologic-Cancer-Detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
```bash
# Using Kaggle API (requires authentication)
kaggle competitions download -c histopathologic-cancer-detection -p data/
```

4. **Run the notebook**
```bash
jupyter notebook Histopathologic_Cancer_Detection.ipynb
```

## 🏗️ Model Architectures

This project implements and compares multiple CNN architectures:

### 1. **Simple Custom CNN**
- 4 convolutional blocks with BatchNormalization
- Progressive filter sizes: 32 → 64 → 128 → 256
- Global Average Pooling
- Dropout regularization
- **Hyperparameter Optimization**: Learning rate, optimizer, batch size, dropout rate

### 2. **Transfer Learning with Pre-trained Models with VGG19 Architecture**
- Pre-trained VGG19 model with ImageNet weights
- Custom classification head for binary cancer detection
- Frozen base layers for feature extraction
- Fine-tuning capabilities for domain adaptation

### 3. **Transfer Learning with Pre-trained Models with ResNet50 (Residual Network)**
- Pre-trained ResNet50 model with ImageNet weights
- Residual connections to mitigate vanishing gradients
- Custom dense layers for histopathology classification
- Transfer learning from natural images to medical imaging

## 🔧 Key Features

### Hyperparameter Optimization
- **Learning Rates**: [0.001, 0.0001]
- **Optimizers**: Adam, RMSprop
- **Batch Sizes**: [16, 32, 64]
- **Dropout Rates**: [0.3, 0.5, 0.7]
- **Early Stopping**: Monitors validation loss with patience=3

### Data Preprocessing
- Image normalization (0-1 scaling)
- Efficient pickle storage for preprocessed images
- Train/validation split with stratification
- GPU optimization for TensorFlow

### Evaluation Metrics
- Accuracy
- AUC-ROC
- Precision & Recall
- Confusion Matrix
- Classification Report

## 📈 Results

The project evaluates multiple architectures with comprehensive hyperparameter tuning:

| Model | Test Accuracy | AUC | Notes |
|-------|---------------|-----|-------|
| Simple Custom CNN (Optimized) | 0.9475 | 0.9859 | With hyperparameter tuning |
| VGG19 (Transfer Learning) | 0.8236 | 0.8984 | Pre-trained on ImageNet |
| ResNet50 (Transfer Learning) | 0.7112 | 0.7713 | Residual network architecture |

*Results will be updated after running the complete optimization pipeline.*

## 🔍 Research Questions

### Exploratory Questions
1. What is the class distribution in the dataset?
2. What are the key visual differences between cancerous and non-cancerous patches?
3. How do color histograms compare between classes?
4. What is the impact of image quality variations?

### Predictive Questions
1. Can CNNs accurately classify metastatic cancer in histopathology patches?
2. Which architecture yields the best performance?
3. How does hyperparameter optimization impact model performance?
4. What level of confidence can the model provide for clinical decision support?

## 📁 Project Structure

```
Histopathologic-Cancer-Detection/
├── README.md
├── Histopathologic_Cancer_Detection.ipynb    # Main notebook
├── data/
│   ├── train/                                # Training images (.tif)
│   ├── test/                                 # Test images (.tif)
│   ├── train_labels.csv                      # Training labels
│   └── sample_submission.csv                 # Submission template
├── images_pickle/
│   ├── X_train_images.pkl                    # Preprocessed training images
│   └── X_val_images.pkl                      # Preprocessed validation images
└── requirements.txt                          # Python dependencies
```

## 🛠️ Technical Implementation

### GPU Configuration
The project includes comprehensive GPU setup for TensorFlow:
- Automatic GPU detection
- Memory growth configuration
- Fallback to CPU if GPU unavailable

### Data Pipeline
- Efficient image loading with OpenCV
- Batch preprocessing with progress tracking
- Pickle serialization for faster subsequent runs
- Strategic memory management

### Model Training
- Early stopping to prevent overfitting
- Comprehensive logging of hyperparameter combinations
- Best model tracking and parameter storage
- Validation monitoring during training

## 📚 Clinical Relevance

This project addresses a critical need in digital pathology:

- **Speed**: Automated analysis reduces diagnostic time
- **Consistency**: Reduces inter-observer variability
- **Scalability**: Can process large volumes of tissue samples
- **Support**: Assists pathologists rather than replacing them
- **Quality**: Potential for improved diagnostic accuracy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle**: Histopathologic Cancer Detection competition
- **PatchCamelyon**: Original PCam benchmark dataset
- **Camelyon16/17**: Source challenges for the dataset
- **Medical AI Community**: For advancing automated pathology

## 📞 Contact

**Author**: Mehdi Afiatpour
**Email**: mehdi@afiatpour.com
**GitHub**: [@mafiat](https://github.com/mafiat)
**Project Link**: [https://github.com/mafiat/Histopathologic-Cancer-Detection](https://github.com/mafiat/Histopathologic-Cancer-Detection)

---

<div align="center">
  <strong>🔬 Advancing Medical AI for Better Healthcare 🔬</strong>
</div>
