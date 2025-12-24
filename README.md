# 🚀 AutoML & Data Analysis Platform

A comprehensive **Streamlit-based AutoML application** that allows users to upload datasets, perform data cleaning, exploratory data analysis (EDA), preprocessing, automated model training, evaluation, and download trained models — **all without writing ML code**.

---

## 🌐 Live Demo & Repository

- 🔗 **Live App:** https://automl-data-analysis-platform.streamlit.app/  
- 📦 **GitHub Repo:** https://github.com/M-Muneebweb/AutoML-Data-Analysis-Platform

---

## ✨ Key Features

### 📁 Data Handling
- Upload datasets in **CSV, Excel, JSON, TSV, Parquet**
- Automatic data profiling (rows, columns, missing values, duplicates)
- Download cleaned & processed datasets

### 🧹 Data Cleaning
- Missing value handling (mean, median, mode, drop)
- Duplicate removal
- Outlier detection using **IQR**
- Preview categorical encoding

### 📊 Exploratory Data Analysis (EDA)
- Histograms & distributions
- Count plots for categorical features
- Boxplots & outlier visualization
- Correlation heatmaps
- Pairplots (with sampling for large data)

### 🎯 AutoML Intelligence
- Automatic **problem type detection** (Classification / Regression)
- Target & feature selection
- Feature scaling (StandardScaler, MinMaxScaler)
- Categorical encoding (Label Encoding, One-Hot Encoding)

### 🤖 Model Training
**Classification Models**
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- KNN
- XGBoost (optional)

**Regression Models**
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- SVR
- XGBoost Regressor (optional)

### ⚙️ Hyperparameter Tuning
- Default parameters
- Grid Search CV
- Randomized Search CV

### 📈 Model Evaluation
- Accuracy, F1-score, ROC-AUC (Classification)
- RMSE, MAE, R² Score (Regression)
- Confusion Matrix / Residual plots
- Feature importance visualization
- Best model auto-selection

### 📥 Downloads
- Trained ML model (`.pkl`)
- Cleaned dataset
- Preprocessed dataset

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Deployment:** Streamlit Cloud

---

## 📦 Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
seaborn>=0.12.0
matplotlib>=3.7.0
openpyxl>=3.1.0
pyarrow>=14.0.0
```

---

## ▶️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/M-Muneebweb/AutoML-Data-Analysis-Platform.git

# Navigate to project directory
cd AutoML-Data-Analysis-Platform

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📌 Use Cases

- Beginners learning Machine Learning
- Freelancers & Data Analysts
- Rapid ML prototyping
- Client demos & proof-of-concepts

---

## 👨‍💻 Author

**Muneeb**  
Python | Data Analysis | Machine Learning  
🌐 GitHub: https://github.com/M-Muneebweb

---

## ⭐ Support

If you like this project:
- ⭐ Star the repository  
- 🍴 Fork it  
- 🐛 Report issues  
- 💡 Suggest new features  

---

> 🤖 *Upload your data, let AutoML do the heavy lifting, and get insights in minutes!*  
