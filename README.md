# 🚀 Website Performance Analyzer (Streamlit)

A powerful **Streamlit-based web application** that analyzes website
performance, SEO health, and resource usage.\
It provides **performance grades**, **warnings**, **actionable
optimization tips**, and **downloadable reports**.

------------------------------------------------------------------------

## 🌐 Live Demo

👉 **App:** https://automl-data-analysis-platform.streamlit.app/\
👉 **GitHub:**
https://github.com/M-Muneebweb/AutoML-Data-Analysis-Platform

------------------------------------------------------------------------

## ✨ Features

-   ⏱️ Website load time analysis
-   📦 Page size & request count detection
-   🖼️ Image, CSS & JavaScript resource analysis
-   🧠 SEO checks (Title & Meta Description)
-   🚨 Performance warnings with severity levels
-   🎯 A--F performance grading system
-   💡 Actionable optimization recommendations
-   📥 Downloadable performance report (TXT)
-   📊 Clean & interactive UI using Streamlit

------------------------------------------------------------------------

## 📸 Preview

> Enter one or multiple website URLs and get instant performance
> insights.

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   **Frontend / UI:** Streamlit
-   **Backend:** Python
-   **Libraries:**
    -   requests
    -   BeautifulSoup (bs4)
    -   pandas
    -   plotly
    -   datetime

------------------------------------------------------------------------

## 📦 Installation

### 1️⃣ Clone the Repository

``` bash
git clone https://github.com/M-Muneebweb/AutoML-Data-Analysis-Platform.git
cd AutoML-Data-Analysis-Platform
```

### 2️⃣ Create Virtual Environment (Recommended)

``` bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Requirements

``` bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📄 Requirements

    streamlit>=1.28.0
    pandas>=2.0.0
    numpy>=1.24.0
    scikit-learn>=1.3.0
    xgboost>=2.0.0
    seaborn>=0.12.0
    matplotlib>=3.7.0
    openpyxl>=3.1.0
    pyarrow>=14.0.0

------------------------------------------------------------------------

## 📊 Performance Metrics Explained

  Metric      Description
  ----------- -----------------------------
  Load Time   Time to fetch the main page
  Page Size   Total downloaded data
  Requests    Number of HTTP requests
  SEO         Title & Meta Description
  Grade       A--F score (0--100)

------------------------------------------------------------------------

## ⚠️ Warnings System

-   🚨 **Critical** -- Immediate optimization required\
-   ⚠️ **Medium** -- Should be improved\
-   ✅ **Good** -- No action required

------------------------------------------------------------------------

## 📥 Downloadable Report

The app allows you to **download a detailed TXT performance report**
including: - Grade & score - Load time - Page size - Requests - SEO
warnings

------------------------------------------------------------------------

## 🔮 Future Improvements

-   PDF & CSV report export
-   Core Web Vitals (LCP, CLS, INP)
-   Mobile performance testing
-   Lighthouse API integration
-   Dark mode UI

------------------------------------------------------------------------

## 👨‍💻 Author

**Muneeb**\
💼 Freelancer \| Python & Data Analysis\
🌍 GitHub: https://github.com/M-Muneebweb

------------------------------------------------------------------------

## ⭐ Support

If you like this project: - ⭐ Star the repository\
- 🐛 Report issues\
- 🤝 Contribute with PRs

------------------------------------------------------------------------

## 📜 License

This project is licensed under the **MIT License**.

------------------------------------------------------------------------

🕒 Generated on: 2025-12-24 13:53:29
