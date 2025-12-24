# 🤖 AutoML Data Analysis Platform

An interactive **Streamlit-based AutoML & Data Analysis platform** that allows users to upload datasets, perform automatic data cleaning, exploratory data analysis (EDA), visualize data, and train multiple machine learning models — all in a single app.

---

## ✨ Features

- 📂 Upload datasets in multiple formats: CSV, Excel, JSON, TSV, Parquet
- 🧹 Automatic **data cleaning**:
  - Detect missing values
  - Remove duplicates
  - Handle categorical features
- 📊 **Exploratory Data Analysis (EDA)** with interactive plots using Seaborn/Matplotlib:
  - Histograms, boxplots, count plots
  - Correlation heatmap, pairplots
- 🎯 Target feature selection & problem type detection (classification/regression)
- ⚙ Feature scaling (StandardScaler / MinMaxScaler) and encoding (OneHot / LabelEncoding)
- 🤖 Train multiple ML models with user-defined or default hyperparameters:
  - Classification: Logistic Regression, Random Forest, XGBoost, SVM, KNN
  - Regression: Linear Regression, Random Forest Regressor, Gradient Boosting, XGBoost, SVR
- 📈 Model evaluation and comparison:
  - Classification: Accuracy, F1-Score, ROC-AUC, Confusion Matrix
  - Regression: RMSE, MAE, R², Residual plots
- 🏆 Highlight best-performing model
- 📥 Download trained model (.pkl) and cleaned dataset
- ⚡ Interactive UI with Streamlit components (sidebar, tabs, buttons, metrics, progress bars)

---

## 🌐 Live App

Try the live app here:  
👉 [AutoML Data Analysis Platform](https://automl-data-analysis-platform.streamlit.app/)

## 📂 GitHub Repository

Check the code:  
👉 [GitHub - AutoML Data Analysis Platform](https://github.com/M-Muneebweb/AutoML-Data-Analysis-Platform)

---

## 🛠️ Tech Stack

- **Python**  
- **Streamlit**  
- **Pandas**  
- **NumPy**  
- **Scikit-Learn**  
- **XGBoost**  
- **Seaborn**  
- **Matplotlib**

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/M-Muneebweb/AutoML-Data-Analysis-Platform.git
cd AutoML-Data-Analysis-Platform```
2️⃣ Install dependencies
```pip install -r requirements.txt
```
Requirements:
```
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
3️⃣ Run the app
```streamlit run app.py
```
🧪 How to Use

Upload your dataset (CSV, Excel, JSON, TSV, Parquet)

Explore dataset summary and column types

Perform data cleaning (missing values, duplicates, encoding)

Generate interactive plots for EDA

Select target feature and problem type (classification/regression)

Choose features, models, and hyperparameters

Train models and evaluate performance metrics

View plots and comparison charts

Download best model and cleaned dataset

📊 Performance & Evaluation Metrics

Classification: Accuracy, F1-Score, ROC-AUC, Confusion Matrix

Regression: RMSE, MAE, R², Residual plots

👨‍💻 Author

Muhammad Muneeb
AI & Data Science Developer
Pakistan

📜 License

This project is licensed under the MIT License.
