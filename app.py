"""
Breast Cancer Classification - Streamlit Web App
ML Assignment 2
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide"
)

# Title and description
st.title("🎗️ Breast Cancer Classification System")
st.markdown("""
This application demonstrates **6 Machine Learning models** for breast cancer classification using the 
**Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

Upload your test data (CSV format) and select a model to see predictions and evaluation metrics.
""")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Machine Learning Model:",
    ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors (kNN)", 
     "Naive Bayes", "Random Forest", "XGBoost"]
)

# Model mapping
model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors (kNN)": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model_names_short = {
    "Logistic Regression": "Logistic Regression",
    "Decision Tree": "Decision Tree",
    "K-Nearest Neighbors (kNN)": "kNN",
    "Naive Bayes": "Naive Bayes",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost"
}

# Load scaler
@st.cache_resource
def load_scaler():
    with open('model/scaler.pkl', 'rb') as f:
        return pickle.load(f)

# Load model
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# File upload
st.sidebar.header("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

# Information about dataset
with st.expander("ℹ️ About the Dataset"):
    st.write("""
    **Breast Cancer Wisconsin (Diagnostic) Dataset**
    
    This dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses.
    
    **Source**: UCI Machine Learning Repository
    
    **30 Features** (10 real-valued features × 3 measurements):
    
    **Mean values:**
    1. radius - mean of distances from center to points on the perimeter
    2. texture - standard deviation of gray-scale values
    3. perimeter
    4. area
    5. smoothness - local variation in radius lengths
    6. compactness - perimeter² / area - 1.0
    7. concavity - severity of concave portions of the contour
    8. concave points - number of concave portions of the contour
    9. symmetry
    10. fractal dimension - "coastline approximation" - 1
    
    **Standard error** of the above 10 features (11-20)
    
    **Worst** (mean of three largest values) of the above 10 features (21-30)
    
    **Target Variable:**
    - **0**: Malignant (Cancer)
    - **1**: Benign (No Cancer)
    
    **Total Instances**: 569 | **Features**: 31 (30 input + 1 target)
    """)

# Model comparison table
with st.expander("📊 Model Performance Comparison"):
    try:
        comparison_df = pd.read_csv('model/model_comparison.csv', index_col=0)
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))
        
        # Bar chart for comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Model Performance Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.legend(loc='lower right')
        ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning("Model comparison data not available. Please train models first.")

# Main section
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
    
    # Display data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10))
    
    # Check if target column exists
    if 'target' in df.columns:
        X = df.drop('target', axis=1)
        y_true = df['target']
        has_labels = True
    else:
        X = df
        y_true = None
        has_labels = False
        st.warning("⚠️ No 'target' column found. Showing predictions only (no evaluation metrics).")
    
    # Load selected model
    model = load_model(model_files[model_choice])
    scaler = load_scaler()
    
    # Check if model needs scaling
    needs_scaling = model_choice in ["Logistic Regression", "K-Nearest Neighbors (kNN)", "Naive Bayes"]
    
    # Make predictions
    if needs_scaling:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Display predictions
    st.subheader(f"🔮 Predictions using {model_choice}")
    predictions_df = pd.DataFrame({
        'Actual': y_true if has_labels else ['N/A'] * len(y_pred),
        'Predicted': y_pred,
        'Predicted_Label': ['Benign' if p == 1 else 'Malignant' for p in y_pred],
        'Probability_Benign': y_pred_proba
    })
    st.dataframe(predictions_df.head(20))
    
    # Evaluation metrics (only if labels are available)
    if has_labels:
        st.subheader("📈 Evaluation Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = accuracy_score(y_true, y_pred)
            st.metric("Accuracy", f"{accuracy:.4f}")
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            st.metric("Precision", f"{precision:.4f}")
        
        with col2:
            auc = roc_auc_score(y_true, y_pred_proba)
            st.metric("AUC Score", f"{auc:.4f}")
            
            recall = recall_score(y_true, y_pred, zero_division=0)
            st.metric("Recall", f"{recall:.4f}")
        
        with col3:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            st.metric("F1 Score", f"{f1:.4f}")
            
            mcc = matthews_corrcoef(y_true, y_pred)
            st.metric("MCC Score", f"{mcc:.4f}")
        
        # Confusion Matrix
        st.subheader("🎯 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Malignant', 'Benign'],
                    yticklabels=['Malignant', 'Benign'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_choice}')
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("📄 Classification Report")
        report = classification_report(y_true, y_pred, 
                                       target_names=['Malignant', 'Benign'],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"))
    
    # Download predictions
    st.subheader("💾 Download Predictions")
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name=f"predictions_{model_names_short[model_choice]}.csv",
        mime="text/csv"
    )

else:
    # Show instructions
    st.info("👈 Please upload a CSV file from the sidebar to get started.")
    
    st.markdown("""
    ### How to use this app:
    1. **Select a model** from the dropdown in the sidebar
    2. **Upload your test data** (CSV format) using the file uploader
    3. **View predictions** and evaluation metrics
    4. **Compare models** using the comparison table above
    
    ### CSV Format Requirements:
    - Must contain the 30 feature columns
    - Optionally include 'target' column for evaluation (0 = Malignant, 1 = Benign)
    - No missing values
    
    ### Example:
    You can download sample test data from the GitHub repository.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>ML Assignment 2 - Breast Cancer Classification | M.Tech (AIML/DSE)</p>
    <p>Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) | UCI ML Repository</p>
</div>
""", unsafe_allow_html=True)
