# Sampling Techniques - Credit Card Fraud Detection

## 📋 Objective 
The objective of this assignment is to understand the importance of sampling techniques in handling imbalanced datasets and to analyze how different sampling strategies affect the performance of various machine learning models. 

## 🎯 Problem Statement 
You are given a highly imbalanced credit card dataset. In real-world applications, such imbalance can significantly affect model performance. Your task is to balance the dataset, apply different sampling techniques, and evaluate how these techniques influence the accuracy of multiple machine learning models.

## 📊 Dataset Information
- **File:** `Creditcard_data.csv`
- **Total Samples:** 772
- **Features:** 30 (Time, V1-V28, Amount)
- **Target:** Class (0 = Normal, 1 = Fraud)
- **Class Distribution:**
  - Class 0 (Normal): 763 samples
  - Class 1 (Fraud): 9 samples
  - **Imbalance Ratio:** 84.78:1

## 🔬 Sampling Techniques Applied

### Sampling1: SMOTE (Synthetic Minority Over-sampling Technique)
- Creates synthetic samples by interpolating between existing minority class samples
- Balances dataset to 763:763

### Sampling2: Random Over-sampling
- Randomly duplicates minority class samples
- Balances dataset to 763:763

### Sampling3: Random Under-sampling
- Randomly removes majority class samples
- Reduces dataset to 9:9

### Sampling4: ADASYN (Adaptive Synthetic Sampling)
- Similar to SMOTE but focuses on harder-to-learn samples
- Balances dataset to ~765:763

### Sampling5: SMOTETomek
- Combines SMOTE over-sampling with Tomek Links under-sampling
- Cleans borderline samples for better decision boundaries
- Results in 745:745

## 🤖 Machine Learning Models

| Model | Description |
|-------|-------------|
| **M1** | Logistic Regression |
| **M2** | Decision Tree Classifier |
| **M3** | Random Forest Classifier |
| **M4** | Support Vector Machine (SVM) |
| **M5** | K-Nearest Neighbors (KNN) |

## 📈 Results

### Accuracy Table (%)

|            | M1 (Logistic) | M2 (Decision Tree) | M3 (Random Forest) | M4 (SVM) | M5 (KNN) |
|------------|---------------|--------------------|--------------------|----------|----------|
| Sampling1  | 91.70         | 97.82              | 98.91              | 67.47    | 86.03    |
| Sampling2  | 91.70         | 99.13              | **100.00**         | 75.11    | 98.91    |
| Sampling3  | 33.33         | 66.67              | 50.00              | 66.67    | 66.67    |
| Sampling4  | 90.41         | 96.95              | 98.91              | 66.23    | 83.44    |
| Sampling5  | 91.95         | 95.75              | 99.11              | 66.89    | 84.34    |

### 🏆 Best Combinations

**Best Sampling Technique for Each Model:**
- **M1 (Logistic Regression):** Sampling5 (SMOTETomek) - 91.95%
- **M2 (Decision Tree):** Sampling2 (Random Oversampling) - 99.13%
- **M3 (Random Forest):** Sampling2 (Random Oversampling) - **100.00%** 🎯
- **M4 (SVM):** Sampling2 (Random Oversampling) - 75.11%
- **M5 (KNN):** Sampling2 (Random Oversampling) - 98.91%

**Overall Best Combination:**
- **Model:** M3 (Random Forest)
- **Sampling:** Sampling2 (Random Over-sampling)
- **Accuracy:** 100.00%

## 📊 Key Findings

1. **Sampling2 (Random Over-sampling)** consistently performs best across most models
2. **Sampling3 (Random Under-sampling)** performs poorly due to significant data loss
3. **Random Forest (M3)** is the most robust model across all sampling techniques
4. **SVM (M4)** struggles with all sampling techniques, showing lowest accuracies
5. Over-sampling techniques significantly outperform under-sampling for highly imbalanced data

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Complete Assignment
```bash
python sampling_assignment.py
```

### Output Files Generated
1. `sampling_results.csv` - Complete accuracy table
2. `heatmap_sampling_vs_models.png` - Visual heatmap of results
3. `best_combinations.png` - Bar charts showing best combinations
4. `performance_comparison_line.png` - Line plot comparing model performance

## 📁 Project Structure
```
sampling_techniques/
├── Creditcard_data.csv                  # Original dataset
├── sampling_assignment.py               # Main script
├── sampling_results.csv                 # Results table
├── heatmap_sampling_vs_models.png      # Visualization 1
├── best_combinations.png               # Visualization 2
├── performance_comparison_line.png     # Visualization 3
├── requirements.txt                    # Dependencies
├── Sampling_Assignment.pdf             # Assignment document
└── README.md                           # This file
```

## 🛠️ Technologies Used
- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **scikit-learn** - Machine learning models
- **imbalanced-learn** - Sampling techniques
- **matplotlib & seaborn** - Visualizations

## 💡 Conclusions

1. **Random Over-sampling** is the most effective technique for this highly imbalanced dataset
2. **Avoid under-sampling** when minority class has very few samples (only 9 fraud cases)
3. **Random Forest** achieves perfect accuracy with proper sampling
4. **SMOTE and ADASYN** provide good alternatives with synthetic data generation
5. **Class imbalance** significantly impacts model performance - proper sampling is crucial
