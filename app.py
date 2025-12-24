"""
🚀 AutoML & Data Analysis Platform
A comprehensive Streamlit application for automated machine learning and data analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import pickle
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'df': None,
        'df_cleaned': None,
        'df_processed': None,
        'target_column': None,
        'feature_columns': [],
        'problem_type': None,
        'models_trained': {},
        'best_model': None,
        'best_model_name': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'label_encoders': {},
        'scaler': None,
        'data_uploaded': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
@st.cache_data
def load_data(uploaded_file):
    """Load data from various file formats"""
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        elif file_extension == 'tsv':
            df = pd.read_csv(uploaded_file, sep='\t')
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_column_types(df):
    """Categorize columns by data type"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    return numeric_cols, categorical_cols, datetime_cols

def detect_problem_type(y):
    """Auto-detect if classification or regression"""
    unique_values = y.nunique()
    if unique_values <= 20 or y.dtype == 'object':
        return 'classification'
    return 'regression'

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def download_model(model, filename="model.pkl"):
    """Create downloadable model file"""
    buffer = BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    return buffer

def download_dataframe(df, filename="data.csv"):
    """Create downloadable CSV file"""
    return df.to_csv(index=False).encode('utf-8')

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("🤖 AutoML Platform")
    st.markdown("---")

    # Navigation
    st.subheader("📍 Navigation")
    page = st.radio(
        "Go to:",
        ["🏠 Home & Upload", "🧹 Data Cleaning", "📊 EDA", 
         "🎯 Target & Features", "⚙️ Preprocessing", 
         "🤖 Model Training", "📈 Results"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Data Info
    if st.session_state.df is not None:
        st.subheader("📋 Data Info")
        st.metric("Rows", st.session_state.df.shape[0])
        st.metric("Columns", st.session_state.df.shape[1])
        if st.session_state.target_column:
            st.metric("Target", st.session_state.target_column)
        if st.session_state.problem_type:
            st.metric("Problem Type", st.session_state.problem_type.title())

# ============================================================
# PAGE 1: HOME & DATA UPLOAD
# ============================================================
if page == "🏠 Home & Upload":
    st.markdown('<p class="main-header">🚀 AutoML & Data Analysis Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your data and let AI do the heavy lifting!</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📁 **Step 1:** Upload your dataset")
    with col2:
        st.info("🧹 **Step 2:** Clean & preprocess data")
    with col3:
        st.info("🤖 **Step 3:** Train models & get results")

    st.markdown("---")

    # File Upload Section
    st.subheader("📤 Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json', 'tsv', 'parquet'],
        help="Supported formats: CSV, Excel, JSON, TSV, Parquet"
    )

    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)

        if df is not None:
            st.session_state.df = df
            st.session_state.df_cleaned = df.copy()
            st.session_state.data_uploaded = True
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

            # Display data preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Dataset Summary
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Dataset Summary")
                numeric_cols, categorical_cols, datetime_cols = get_column_types(df)

                summary_data = {
                    "Metric": ["Total Rows", "Total Columns", "Numeric Columns", 
                              "Categorical Columns", "Missing Values", "Duplicate Rows"],
                    "Value": [df.shape[0], df.shape[1], len(numeric_cols), 
                             len(categorical_cols), df.isnull().sum().sum(), df.duplicated().sum()]
                }
                st.table(pd.DataFrame(summary_data))

            with col2:
                st.subheader("📋 Column Types")
                dtype_df = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str),
                    "Non-Null": df.count().values,
                    "Null": df.isnull().sum().values
                })
                st.dataframe(dtype_df, use_container_width=True, height=300)

            # Descriptive Statistics
            st.subheader("📈 Descriptive Statistics")
            st.dataframe(df.describe(include='all').T, use_container_width=True)

# ============================================================
# PAGE 2: DATA CLEANING
# ============================================================
elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df_cleaned.copy()

        tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Duplicates", "Outliers", "Encoding"])

        # --- TAB 1: Missing Values ---
        with tab1:
            st.subheader("🔍 Missing Values Analysis")

            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]

            if len(missing_data) == 0:
                st.success("✅ No missing values found!")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    missing_df = pd.DataFrame({
                        "Column": missing_data.index,
                        "Missing Count": missing_data.values,
                        "Percentage": (missing_data.values / len(df) * 100).round(2)
                    })
                    st.dataframe(missing_df, use_container_width=True)

                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    missing_data.plot(kind='barh', ax=ax, color='coral')
                    ax.set_xlabel('Missing Count')
                    ax.set_title('Missing Values by Column')
                    st.pyplot(fig)
                    plt.close()

                st.subheader("🛠️ Handle Missing Values")

                strategy = st.selectbox(
                    "Select strategy:",
                    ["Drop rows with missing values", "Fill with mean (numeric)", 
                     "Fill with median (numeric)", "Fill with mode", "Fill with custom value"]
                )

                if st.button("Apply Missing Value Strategy", key="missing_btn"):
                    if strategy == "Drop rows with missing values":
                        df = df.dropna()
                    elif strategy == "Fill with mean (numeric)":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    elif strategy == "Fill with median (numeric)":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                    elif strategy == "Fill with mode":
                        for col in df.columns:
                            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

                    st.session_state.df_cleaned = df
                    st.success("✅ Missing values handled!")
                    st.rerun()

        # --- TAB 2: Duplicates ---
        with tab2:
            st.subheader("🔄 Duplicate Rows")

            duplicates = df.duplicated().sum()

            if duplicates == 0:
                st.success("✅ No duplicate rows found!")
            else:
                st.warning(f"⚠️ Found {duplicates} duplicate rows ({(duplicates/len(df)*100):.2f}%)")

                if st.button("Remove Duplicates", key="dup_btn"):
                    df = df.drop_duplicates()
                    st.session_state.df_cleaned = df
                    st.success("✅ Duplicates removed!")
                    st.rerun()

        # --- TAB 3: Outliers ---
        with tab3:
            st.subheader("📊 Outlier Detection")

            numeric_cols, _, _ = get_column_types(df)

            if len(numeric_cols) == 0:
                st.info("No numeric columns for outlier detection.")
            else:
                selected_col = st.selectbox("Select column for outlier detection:", numeric_cols)

                if selected_col:
                    outliers, lower, upper = detect_outliers_iqr(df, selected_col)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Outliers Found", len(outliers))
                        st.write(f"Lower Bound: {lower:.2f}")
                        st.write(f"Upper Bound: {upper:.2f}")

                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.boxplot(x=df[selected_col], ax=ax)
                        ax.set_title(f'Boxplot of {selected_col}')
                        st.pyplot(fig)
                        plt.close()

                    if len(outliers) > 0:
                        if st.button("Remove Outliers", key="outlier_btn"):
                            df = df[(df[selected_col] >= lower) & (df[selected_col] <= upper)]
                            st.session_state.df_cleaned = df
                            st.success(f"✅ Removed {len(outliers)} outliers!")
                            st.rerun()

        # --- TAB 4: Auto Encoding Preview ---
        with tab4:
            st.subheader("🔤 Categorical Encoding Preview")

            _, categorical_cols, _ = get_column_types(df)

            if len(categorical_cols) == 0:
                st.info("No categorical columns to encode.")
            else:
                st.write("Categorical columns detected:")
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    st.write(f"- **{col}**: {unique_count} unique values")

                st.info("💡 Encoding will be applied in the Preprocessing step.")

        # Download cleaned data
        st.markdown("---")
        st.subheader("📥 Download Cleaned Data")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Cleaned CSV",
                data=download_dataframe(df),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        with col2:
            st.metric("Rows after cleaning", df.shape[0])
            st.metric("Columns", df.shape[1])

# ============================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS
# ============================================================
elif page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df_cleaned
        numeric_cols, categorical_cols, _ = get_column_types(df)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Histograms", "Count Plots", "Boxplots", "Correlation", "Pairplot"
        ])

        # --- TAB 1: Histograms ---
        with tab1:
            st.subheader("📊 Distribution of Numeric Features")

            if len(numeric_cols) == 0:
                st.info("No numeric columns available.")
            else:
                selected_num_cols = st.multiselect(
                    "Select numeric columns:",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))]
                )

                if selected_num_cols:
                    n_cols = min(2, len(selected_num_cols))
                    n_rows = (len(selected_num_cols) + n_cols - 1) // n_cols

                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
                    axes = np.array(axes).flatten() if len(selected_num_cols) > 1 else [axes]

                    for idx, col in enumerate(selected_num_cols):
                        sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], color='steelblue')
                        axes[idx].set_title(f'Distribution of {col}')

                    for idx in range(len(selected_num_cols), len(axes)):
                        axes[idx].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        # --- TAB 2: Count Plots ---
        with tab2:
            st.subheader("📊 Categorical Feature Counts")

            if len(categorical_cols) == 0:
                st.info("No categorical columns available.")
            else:
                selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)

                if selected_cat_col:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    value_counts = df[selected_cat_col].value_counts()
                    if len(value_counts) > 15:
                        value_counts = value_counts.head(15)
                        st.info("Showing top 15 categories")

                    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette='viridis')
                    ax.set_xlabel('Count')
                    ax.set_ylabel(selected_cat_col)
                    ax.set_title(f'Count Plot of {selected_cat_col}')
                    st.pyplot(fig)
                    plt.close()

        # --- TAB 3: Boxplots ---
        with tab3:
            st.subheader("📦 Boxplots for Outlier Detection")

            if len(numeric_cols) == 0:
                st.info("No numeric columns available.")
            else:
                selected_box_cols = st.multiselect(
                    "Select columns for boxplot:",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))],
                    key="boxplot_cols"
                )

                if selected_box_cols:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    df[selected_box_cols].boxplot(ax=ax)
                    ax.set_title('Boxplots of Selected Features')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        # --- TAB 4: Correlation Heatmap ---
        with tab4:
            st.subheader("🔥 Correlation Heatmap")

            if len(numeric_cols) < 2:
                st.info("Need at least 2 numeric columns for correlation.")
            else:
                corr_cols = st.multiselect(
                    "Select columns for correlation:",
                    numeric_cols,
                    default=numeric_cols[:min(10, len(numeric_cols))],
                    key="corr_cols"
                )

                if len(corr_cols) >= 2:
                    corr_matrix = df[corr_cols].corr()

                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(
                        corr_matrix, 
                        annot=True, 
                        cmap='coolwarm', 
                        center=0,
                        fmt='.2f',
                        ax=ax
                    )
                    ax.set_title('Correlation Heatmap')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        # --- TAB 5: Pairplot ---
        with tab5:
            st.subheader("🔗 Pairplot")

            if len(numeric_cols) < 2:
                st.info("Need at least 2 numeric columns for pairplot.")
            else:
                pair_cols = st.multiselect(
                    "Select columns (max 5 recommended):",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))],
                    key="pair_cols"
                )

                hue_col = None
                if categorical_cols:
                    hue_col = st.selectbox(
                        "Color by (optional):",
                        ["None"] + categorical_cols,
                        key="hue_col"
                    )
                    hue_col = None if hue_col == "None" else hue_col

                if len(pair_cols) >= 2:
                    if st.button("Generate Pairplot", key="pairplot_btn"):
                        with st.spinner("Generating pairplot..."):
                            sample_df = df[pair_cols + ([hue_col] if hue_col else [])].dropna()
                            if len(sample_df) > 1000:
                                sample_df = sample_df.sample(1000, random_state=42)

                            fig = sns.pairplot(sample_df, hue=hue_col, diag_kind='kde')
                            st.pyplot(fig)
                            plt.close()

# ============================================================
# PAGE 4: TARGET & FEATURE SELECTION
# ============================================================
elif page == "🎯 Target & Features":
    st.header("🎯 Target & Feature Selection")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df_cleaned

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Select Target Column")
            target_column = st.selectbox(
                "Target variable (y):",
                df.columns.tolist(),
                index=len(df.columns)-1
            )

            if target_column:
                st.session_state.target_column = target_column

                # Auto-detect problem type
                detected_type = detect_problem_type(df[target_column])

                st.info(f"🔍 Auto-detected: **{detected_type.title()}**")

                problem_type = st.radio(
                    "Confirm or override:",
                    ["classification", "regression"],
                    index=0 if detected_type == "classification" else 1
                )
                st.session_state.problem_type = problem_type

                # Show target distribution
                st.subheader("📊 Target Distribution")

                fig, ax = plt.subplots(figsize=(8, 4))
                if problem_type == "classification":
                    df[target_column].value_counts().plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_ylabel('Count')
                else:
                    sns.histplot(df[target_column].dropna(), kde=True, ax=ax, color='steelblue')
                    ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {target_column}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with col2:
            st.subheader("📋 Select Feature Columns")

            available_features = [col for col in df.columns if col != target_column]

            if st.checkbox("Select All Features", value=True):
                selected_features = available_features
            else:
                selected_features = st.multiselect(
                    "Features (X):",
                    available_features,
                    default=available_features
                )

            st.session_state.feature_columns = selected_features

            st.write(f"**Selected features:** {len(selected_features)}")

            # Feature info
            if selected_features:
                numeric_feats = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
                cat_feats = df[selected_features].select_dtypes(include=['object', 'category']).columns.tolist()

                st.metric("Numeric Features", len(numeric_feats))
                st.metric("Categorical Features", len(cat_feats))

        if st.button("✅ Confirm Selection", type="primary"):
            st.success(f"""
            ✅ Configuration saved!
            - Target: {st.session_state.target_column}
            - Problem Type: {st.session_state.problem_type}
            - Features: {len(st.session_state.feature_columns)}
            """)

# ============================================================
# PAGE 5: PREPROCESSING
# ============================================================
elif page == "⚙️ Preprocessing":
    st.header("⚙️ Feature Scaling & Encoding")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    elif st.session_state.target_column is None:
        st.warning("⚠️ Please select target column first!")
    else:
        df = st.session_state.df_cleaned

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📏 Feature Scaling")
            scaling_method = st.selectbox(
                "Select scaling method:",
                ["None", "StandardScaler", "MinMaxScaler"]
            )

            st.info("""
            **StandardScaler**: Mean=0, Std=1 (good for algorithms assuming normal distribution)

            **MinMaxScaler**: Scales to [0,1] (good for neural networks)
            """)

        with col2:
            st.subheader("🔤 Categorical Encoding")
            encoding_method = st.selectbox(
                "Select encoding method:",
                ["LabelEncoding", "OneHotEncoding"]
            )

            st.info("""
            **LabelEncoding**: Convert categories to integers (good for tree-based models)

            **OneHotEncoding**: Create binary columns (good for linear models)
            """)

        st.markdown("---")

        if st.button("🚀 Apply Preprocessing", type="primary"):
            with st.spinner("Preprocessing data..."):
                progress_bar = st.progress(0)

                # Prepare data
                X = df[st.session_state.feature_columns].copy()
                y = df[st.session_state.target_column].copy()

                progress_bar.progress(20)

                # Handle categorical target
                if st.session_state.problem_type == 'classification' and y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.session_state.label_encoders['target'] = le

                progress_bar.progress(40)

                # Encode categorical features
                cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

                if cat_cols:
                    if encoding_method == "LabelEncoding":
                        for col in cat_cols:
                            le = LabelEncoder()
                            X[col] = X[col].astype(str)
                            X[col] = le.fit_transform(X[col])
                            st.session_state.label_encoders[col] = le
                    else:  # OneHotEncoding
                        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

                progress_bar.progress(60)

                # Handle missing values in features
                X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)

                progress_bar.progress(80)

                # Scale features
                if scaling_method != "None":
                    scaler = StandardScaler() if scaling_method == "StandardScaler" else MinMaxScaler()
                    X_scaled = scaler.fit_transform(X)
                    X = pd.DataFrame(X_scaled, columns=X.columns)
                    st.session_state.scaler = scaler

                progress_bar.progress(100)

                # Store processed data
                st.session_state.df_processed = X

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test

                st.success("✅ Preprocessing complete!")

                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(X))
                with col2:
                    st.metric("Training Samples", len(X_train))
                with col3:
                    st.metric("Test Samples", len(X_test))

                st.subheader("📊 Processed Features Preview")
                st.dataframe(X.head(), use_container_width=True)

# ============================================================
# PAGE 6: MODEL TRAINING
# ============================================================
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")

    if st.session_state.X_train is None:
        st.warning("⚠️ Please complete preprocessing first!")
    else:
        problem_type = st.session_state.problem_type

        # Model selection
        st.subheader("🎛️ Select Models to Train")

        if problem_type == "classification":
            available_models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42),
                "KNN": KNeighborsClassifier()
            }
            if XGBOOST_AVAILABLE:
                available_models["XGBoost"] = XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            available_models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "SVR": SVR()
            }
            if XGBOOST_AVAILABLE:
                available_models["XGBoost"] = XGBRegressor(random_state=42)

        selected_models = st.multiselect(
            "Choose models:",
            list(available_models.keys()),
            default=list(available_models.keys())[:3]
        )

        st.markdown("---")

        # Hyperparameter options
        st.subheader("⚙️ Hyperparameter Tuning")

        tuning_option = st.radio(
            "Tuning method:",
            ["Use Default Parameters", "Grid Search (Slower)", "Random Search"]
        )

        st.markdown("---")

        if st.button("🚀 Train Models", type="primary"):
            if not selected_models:
                st.error("Please select at least one model!")
            else:
                X_train = st.session_state.X_train
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test

                results = {}
                trained_models = {}

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, model_name in enumerate(selected_models):
                    status_text.text(f"Training {model_name}...")

                    model = available_models[model_name]

                    # Apply hyperparameter tuning if selected
                    if tuning_option != "Use Default Parameters":
                        param_grids = {
                            "Random Forest": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
                            "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                            "KNN": {"n_neighbors": [3, 5, 7]},
                            "SVM": {"C": [0.1, 1, 10]},
                            "SVR": {"C": [0.1, 1, 10]},
                            "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
                        }

                        if model_name in param_grids:
                            scoring = 'accuracy' if problem_type == "classification" else 'r2'

                            # FIX: Separate GridSearchCV and RandomizedSearchCV initialization
                            if tuning_option == "Grid Search (Slower)":
                                search = GridSearchCV(
                                    model, 
                                    param_grids[model_name], 
                                    cv=3, 
                                    scoring=scoring
                                )
                            else:  # Random Search
                                search = RandomizedSearchCV(
                                    model, 
                                    param_grids[model_name], 
                                    cv=3, 
                                    scoring=scoring,
                                    n_iter=5,
                                    random_state=42
                                )

                            search.fit(X_train, y_train)
                            model = search.best_estimator_
                        else:
                            model.fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train)

                    # Predictions
                    y_pred = model.predict(X_test)

                    # Calculate metrics
                    if problem_type == "classification":
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        try:
                            if hasattr(model, 'predict_proba'):
                                y_proba = model.predict_proba(X_test)
                                if len(np.unique(y_test)) == 2:
                                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                                else:
                                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                            else:
                                roc_auc = None
                        except:
                            roc_auc = None

                        results[model_name] = {
                            "Accuracy": accuracy,
                            "F1-Score": f1,
                            "ROC-AUC": roc_auc if roc_auc else "N/A"
                        }
                    else:
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        results[model_name] = {
                            "RMSE": rmse,
                            "MAE": mae,
                            "R²": r2
                        }

                    trained_models[model_name] = {
                        "model": model,
                        "predictions": y_pred
                    }

                    progress_bar.progress((idx + 1) / len(selected_models))

                status_text.text("Training complete!")

                # Store results
                st.session_state.models_trained = trained_models

                # Find best model
                if problem_type == "classification":
                    best_model_name = max(results, key=lambda x: results[x]["Accuracy"])
                else:
                    best_model_name = max(results, key=lambda x: results[x]["R²"])

                st.session_state.best_model = trained_models[best_model_name]["model"]
                st.session_state.best_model_name = best_model_name

                # Display results
                st.success("✅ All models trained successfully!")

                st.subheader("📊 Model Performance Comparison")

                results_df = pd.DataFrame(results).T
                results_df.index.name = "Model"

                # Highlight best model
                st.dataframe(
                    results_df.style.highlight_max(axis=0, color='lightgreen'),
                    use_container_width=True
                )

                st.success(f"🏆 **Best Model: {best_model_name}**")

# ============================================================
# PAGE 7: RESULTS & DOWNLOAD
# ============================================================
elif page == "📈 Results":
    st.header("📈 Results & Model Download")

    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
    else:
        problem_type = st.session_state.problem_type
        trained_models = st.session_state.models_trained

        # Best model highlight
        st.subheader("🏆 Best Model")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", st.session_state.best_model_name)
        with col2:
            y_test = st.session_state.y_test
            y_pred = trained_models[st.session_state.best_model_name]["predictions"]

            if problem_type == "classification":
                score = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{score:.4f}")
            else:
                score = r2_score(y_test, y_pred)
                st.metric("R² Score", f"{score:.4f}")
        with col3:
            if problem_type == "classification":
                f1 = f1_score(y_test, y_pred, average='weighted')
                st.metric("F1-Score", f"{f1:.4f}")
            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.metric("RMSE", f"{rmse:.4f}")

        st.markdown("---")

        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix / Residuals", "Feature Importance", "Model Comparison"])

        with tab1:
            model_to_viz = st.selectbox(
                "Select model to visualize:",
                list(trained_models.keys())
            )

            y_pred = trained_models[model_to_viz]["predictions"]

            if problem_type == "classification":
                st.subheader("📊 Confusion Matrix")

                cm = confusion_matrix(st.session_state.y_test, y_pred)

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'Confusion Matrix - {model_to_viz}')
                st.pyplot(fig)
                plt.close()

                # Classification report
                st.subheader("📋 Classification Report")
                report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).T, use_container_width=True)
            else:
                st.subheader("📊 Residual Plot")

                residuals = st.session_state.y_test - y_pred

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Residual vs Predicted
                axes[0].scatter(y_pred, residuals, alpha=0.5)
                axes[0].axhline(y=0, color='r', linestyle='--')
                axes[0].set_xlabel('Predicted Values')
                axes[0].set_ylabel('Residuals')
                axes[0].set_title('Residuals vs Predicted')

                # Residual distribution
                sns.histplot(residuals, kde=True, ax=axes[1])
                axes[1].set_xlabel('Residuals')
                axes[1].set_title('Residual Distribution')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with tab2:
            st.subheader("📊 Feature Importance")

            # Select model with feature importance
            importance_models = [
                name for name in trained_models.keys() 
                if hasattr(trained_models[name]["model"], 'feature_importances_')
            ]

            if importance_models:
                imp_model_name = st.selectbox(
                    "Select model:",
                    importance_models,
                    key="imp_model"
                )

                model = trained_models[imp_model_name]["model"]
                importances = model.feature_importances_

                feature_names = st.session_state.X_train.columns

                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)

                fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))

                # Show top 20 features if too many
                plot_df = importance_df.tail(20)

                ax.barh(plot_df['Feature'], plot_df['Importance'], color='steelblue')
                ax.set_xlabel('Importance')
                ax.set_title(f'Feature Importance - {imp_model_name}')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No models with feature importance available (e.g., tree-based models).")

        with tab3:
            st.subheader("📊 Model Performance Comparison")

            comparison_data = []
            for name, data in trained_models.items():
                y_pred = data["predictions"]
                y_test = st.session_state.y_test

                if problem_type == "classification":
                    comparison_data.append({
                        "Model": name,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "F1-Score": f1_score(y_test, y_pred, average='weighted')
                    })
                else:
                    comparison_data.append({
                        "Model": name,
                        "R²": r2_score(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                    })

            comparison_df = pd.DataFrame(comparison_data)

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(comparison_df))

            if problem_type == "classification":
                width = 0.35
                bars1 = ax.bar(x - width/2, comparison_df['Accuracy'], width, label='Accuracy', color='steelblue')
                bars2 = ax.bar(x + width/2, comparison_df['F1-Score'], width, label='F1-Score', color='coral')
            else:
                ax.bar(x, comparison_df['R²'], color='steelblue')
                ax.set_ylabel('R² Score')

            ax.set_xlabel('Model')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax.legend()
            ax.set_title('Model Comparison')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # Download section
        st.subheader("📥 Download")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="📦 Download Best Model (.pkl)",
                data=download_model(st.session_state.best_model),
                file_name=f"{st.session_state.best_model_name.lower().replace(' ', '_')}_model.pkl",
                mime="application/octet-stream"
            )

        with col2:
            st.download_button(
                label="📊 Download Cleaned Data (CSV)",
                data=download_dataframe(st.session_state.df_cleaned),
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

        with col3:
            if st.session_state.df_processed is not None:
                st.download_button(
                    label="⚙️ Download Processed Data (CSV)",
                    data=download_dataframe(st.session_state.df_processed),
                    file_name="processed_dataset.csv",
                    mime="text/csv"
                )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        <p>🤖 AutoML Platform | Built with Streamlit & Scikit-Learn</p>
        <p>Upload your data, train models, and get predictions in minutes!</p>
    </div>
    """,
    unsafe_allow_html=True
)
