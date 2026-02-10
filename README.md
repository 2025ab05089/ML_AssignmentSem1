# Breast Cancer Classification

## Problem Statement

Breast cancer is one of the most common cancers among women worldwide. Early and accurate diagnosis is critical for effective treatment and improved survival rates. Traditional diagnostic methods can be time-consuming and subjective. This project aims to develop and compare multiple machine learning classification models to predict whether a breast mass is malignant (cancerous) or benign (non-cancerous) based on features computed from digitized images of fine needle aspirate (FNA) of breast masses.

The goals of this project are to:
- Build 6 different classification models
- Evaluate their performance using comprehensive metrics
- Deploy an interactive web application for real-time predictions
- Compare model performances to identify the most effective approach for clinical decision support

## Dataset Description

**Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset  
**Source**: UCI Machine Learning Repository  
**Type**: Binary Classification (0 = Malignant, 1 = Benign)

### Dataset Characteristics:
- **Total Instances**: 569 patients
- **Total Features**: 31 (30 input features + 1 target variable)
- **Missing Values**: None
- **Class Distribution**: 
  - Class 0 (Malignant): 212 instances (37.3%)
  - Class 1 (Benign): 357 instances (62.7%)

### Feature Descriptions:

Features are computed from digitized images of fine needle aspirate (FNA) of breast masses. They describe characteristics of cell nuclei present in the image.

**Ten real-valued features are computed for each cell nucleus:**

1. **radius** - mean of distances from center to points on the perimeter
2. **texture** - standard deviation of gray-scale values
3. **perimeter** - perimeter of the nucleus
4. **area** - area of the nucleus
5. **smoothness** - local variation in radius lengths
6. **compactness** - perimeter² / area - 1.0
7. **concavity** - severity of concave portions of the contour
8. **concave points** - number of concave portions of the contour
9. **symmetry** - symmetry of the nucleus
10. **fractal dimension** - "coastline approximation" - 1

**For each of these 10 features, three measurements are provided:**
- **Mean** (features 1-10)
- **Standard Error** (features 11-20)
- **Worst** or largest mean of three worst/largest values (features 21-30)

This results in **30 total input features**.

| Feature Type | Examples | Count |
|-------------|----------|-------|
| Mean values | mean radius, mean texture, mean perimeter, mean area, etc. | 10 |
| Standard Error | radius error, texture error, perimeter error, etc. | 10 |
| Worst values | worst radius, worst texture, worst perimeter, etc. | 10 |
| **Total** | | **30** |

**Target Variable:**
- **0**: Malignant (Cancerous)
- **1**: Benign (Non-cancerous)

### Data Source & Credits:
This is the famous Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository.

**Citation:**
- Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

**Creators:**
- Dr. William H. Wolberg (General Surgery Dept., University of Wisconsin)
- W. Nick Street (Computer Sciences Dept., University of Wisconsin)
- Olvi L. Mangasarian (Computer Sciences Dept., University of Wisconsin)

### Data Preprocessing:
- Feature scaling applied using StandardScaler for models requiring normalized features
- Train-test split: 80% training, 20% testing
- Stratified sampling to maintain class distribution
- No missing values or data cleaning required

## Models Used

### Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9825 | 0.9982 | 0.9861 | 0.9861 | 0.9861 | 0.9619 |
| Decision Tree | 0.9298 | 0.9254 | 0.9310 | 0.9444 | 0.9377 | 0.8500 |
| kNN | 0.9649 | 0.9850 | 0.9577 | 0.9444 | 0.9510 | 0.9220 |
| Naive Bayes | 0.9474 | 0.9909 | 0.9437 | 0.9306 | 0.9371 | 0.8845 |
| Random Forest (Ensemble) | 0.9737 | 0.9960 | 0.9718 | 0.9722 | 0.9720 | 0.9423 |
| XGBoost (Ensemble) | 0.9737 | 0.9958 | 0.9859 | 0.9722 | 0.9790 | 0.9430 |


### Model Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieves outstanding performance with 98.25% accuracy and an exceptional AUC of 0.998, demonstrating near-perfect ability to distinguish between malignant and benign cases. The balanced precision and recall (both 0.986) indicate the model rarely makes classification errors. The high MCC of 0.962 confirms strong correlation between predictions and actual outcomes. This linear model excels on this dataset due to the well-separated feature distributions and strong linear relationships. Recommended for clinical deployment due to its simplicity, interpretability, and excellent performance. Coefficients can be examined to understand which features most influence cancer diagnosis. |
| **Decision Tree** | Shows good performance with 92.98% accuracy but is the weakest performer among all models. While precision (0.931) and recall (0.944) are respectable, the lower AUC of 0.925 suggests less reliable probability estimates. The MCC of 0.850 indicates moderate predictive power. The tree structure makes it highly interpretable - doctors can visualize the exact decision path. However, the max_depth=5 constraint may be limiting its ability to capture complex patterns. May be prone to overfitting if depth is increased. Best suited for educational purposes or when complete interpretability is required over maximum accuracy. |
| **kNN** | Demonstrates strong performance with 96.49% accuracy and excellent AUC of 0.985. The algorithm effectively leverages local similarity patterns in the 30-dimensional feature space. With precision of 0.958 and recall of 0.944, it makes few mistakes in both directions. The MCC of 0.922 indicates very good predictive strength. Performance is highly dependent on proper feature scaling (which was applied) and the k=5 neighbor selection. The model doesn't learn parameters but memorizes training data, making it simple yet effective. May struggle with new data far from training examples. Good choice when interpretability is less important than accuracy. |
| **Naive Bayes** | Achieves solid 94.74% accuracy with an outstanding AUC of 0.991, second only to Logistic Regression in probability calibration. Despite its "naive" assumption of feature independence (which doesn't hold in this dataset - features are correlated), it performs remarkably well. The precision (0.944) and recall (0.931) are well-balanced. The Gaussian assumption fits reasonably well due to continuous features. MCC of 0.885 shows good predictive ability. The model is extremely fast to train and makes predictions instantly, making it ideal for real-time applications. The strong AUC despite violated assumptions demonstrates the robustness of the algorithm. |
| **Random Forest (Ensemble)** | Exhibits excellent performance with 97.37% accuracy and near-perfect AUC of 0.996. The ensemble of 100 decision trees (max_depth=10) effectively combines multiple weak learners to create a powerful predictor. With precision of 0.972 and recall of 0.972, the model makes very few errors. The MCC of 0.942 indicates very strong predictive correlation. Random Forest's ability to capture non-linear relationships and feature interactions makes it robust and accurate. Provides feature importance rankings, helping identify which cell characteristics are most indicative of cancer. The ensemble approach reduces overfitting compared to single decision trees. Excellent choice balancing accuracy, robustness, and interpretability through feature importance. |
| **XGBoost (Ensemble)** | Delivers top-tier performance with 97.37% accuracy and exceptional AUC of 0.996, tied with Random Forest but with superior precision (0.986 vs 0.972). The gradient boosting approach iteratively corrects errors from previous trees, resulting in highly optimized predictions. With recall of 0.972 and F1 of 0.979, it achieves the best balance between precision and recall. The MCC of 0.943 (highest among all models) indicates the strongest correlation between predictions and true outcomes. XGBoost's sophisticated regularization techniques prevent overfitting despite model complexity. Provides excellent probability calibration for risk assessment. While computationally more intensive than simpler models, the performance gain justifies the cost for clinical applications where accuracy is paramount. Recommended as the primary model for deployment. 
