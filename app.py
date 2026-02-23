# ===================== IMPORTS =====================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import streamlit as st

from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="E-commerce Sales Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== DATA LOADING =====================
@st.cache_data
def load_data():
    df = pd.read_csv("Modified_Ecommerce_Sales_Prediction.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["total_sales"] = df["Price"] * df["Units_Sold"]
    return df

raw_df = load_data()

# ===================== FEATURE ENGINEERING =====================
def create_time_features(df):
    df = df.sort_values("Date").copy()
    df["lag_1"] = df["Units_Sold"].shift(1)
    df["lag_7"] = df["Units_Sold"].shift(7)
    df["rolling_7"] = df["Units_Sold"].rolling(7).mean()
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df = df.dropna()
    return df

df = create_time_features(raw_df)

# ===================== SIDEBAR =====================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Data Overview",
        "Exploratory Analysis",
        "Time Series Analysis",
        "Regression Analysis",
        "Forecasting Model 🚀",
        "Price Optimization 💰",
        "30-Day Forecast 🔮",
        "Explainability (SHAP) 🧠",
    ],
)

# ===================== PLOT HELPERS =====================
def plot_distribution(column, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(raw_df[column], bins=30, kde=True, ax=ax, color="blue")
    ax.set_title(title)
    st.pyplot(fig)

def plot_scatter(x, y, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=raw_df[x], y=raw_df[y], alpha=0.6, ax=ax, color="blue")
    ax.set_title(title)
    st.pyplot(fig)

def plot_time_series(data, x, y, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=data, x=x, y=y, marker="o", ax=ax)
    ax.set_title(title)
    ax.grid(True)
    st.pyplot(fig)

# =========================================================
# ===================== DATA OVERVIEW =====================
# =========================================================
if page == "Data Overview":
    st.title("📊 E-commerce Sales Data Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(raw_df))
    c2.metric("Total Revenue", f"{raw_df['total_sales'].sum():,.0f}")
    c3.metric("Avg Units Sold", f"{raw_df['Units_Sold'].mean():.1f}")

    st.subheader("First 5 Rows")
    st.dataframe(raw_df.head())

    st.subheader("Descriptive Statistics")
    st.dataframe(raw_df.describe())

    st.subheader("Columns")
    st.write(raw_df.columns.tolist())

# =========================================================
# ===================== EDA (UNCHANGED CORE) ==============
# =========================================================
elif page == "Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distributions", "Relationships", "Correlations", "Segment Analysis"]
    )

    with tab1:
        numerical_cols = ["Price", "Discount", "Marketing_Spend", "Units_Sold"]
        selected_col = st.selectbox("Select numerical column", numerical_cols)
        plot_distribution(selected_col, f"Distribution of {selected_col}")

        categorical_cols = ["Product_Category", "Customer_Segment"]
        selected_cat = st.selectbox("Select categorical column", categorical_cols)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(x=selected_cat, data=raw_df, palette="deep", ax=ax)
        plt.xticks(rotation=30)
        st.pyplot(fig)

    with tab2:
        scatter_pairs = [
            ("Price", "Units_Sold"),
            ("Marketing_Spend", "Units_Sold"),
            ("Discount", "Units_Sold"),
            ("Price", "Marketing_Spend"),
        ]

        selected_pair = st.selectbox(
            "Select pair",
            scatter_pairs,
            format_func=lambda x: f"{x[0]} vs {x[1]}",
        )
        plot_scatter(selected_pair[0], selected_pair[1], "Relationship")

    with tab3:
        numerical_features = [
            "Price",
            "Discount",
            "Marketing_Spend",
            "Units_Sold",
        ]
        selected_features = st.multiselect(
            "Select features",
            numerical_features,
            default=numerical_features,
        )

        if len(selected_features) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            corr_matrix = raw_df[selected_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="Blues", ax=ax)
            st.pyplot(fig)

    with tab4:
        seg = (
            raw_df.groupby(["Product_Category", "Customer_Segment"])
            .agg(Total_Sales=("total_sales", "sum"))
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x="Product_Category",
            y="Total_Sales",
            hue="Customer_Segment",
            data=seg,
            ax=ax,
        )
        plt.xticks(rotation=30)
        st.pyplot(fig)

# =========================================================
# ===================== TIME SERIES =======================
# =========================================================
elif page == "Time Series Analysis":
    st.title("📈 Time Series Analysis")

    monthly_sales = (
        raw_df.set_index("Date")
        .groupby(pd.Grouper(freq="M"))["total_sales"]
        .sum()
        .reset_index()
    )

    plot_time_series(
        monthly_sales, "Date", "total_sales", "Monthly Total Sales"
    )

# =========================================================
# ===================== REGRESSION ========================
# =========================================================
elif page == "Regression Analysis":
    st.title("📉 Regression Analysis")

    tab1, tab2 = st.tabs(["Single Variable Regression", "Multiple Regression"])

    with tab1:
        independent_vars = ["Price", "Discount", "Marketing_Spend"]
        selected_var = st.selectbox("Select independent variable", independent_vars)

        X = raw_df[[selected_var]]
        y = raw_df["Units_Sold"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)

        # ================= METRICS =================
        st.markdown("### 📊 Model Performance Metrics")

        lr_mae = mean_absolute_error(y_test, y_pred)
        lr_mse = mean_squared_error(y_test, y_pred)
        lr_rmse = np.sqrt(lr_mse)
        lr_r2 = r2_score(y_test, y_pred)
        lr_mape = mean_absolute_percentage_error(y_test, y_pred)

        metrics_df = pd.DataFrame(
            {
                "Metric": ["MAE", "MSE", "RMSE", "R²", "MAPE"],
                "Value": [lr_mae, lr_mse, lr_rmse, lr_r2, lr_mape],
            }
        )

        st.table(metrics_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        # ================= PLOT =================
        st.markdown("### 📉 Regression Fit")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(X_test.values.flatten(), y_test, alpha=0.6, label="Actual")
        ax.plot(
            X_test.values.flatten(),
            y_pred,
            color="red",
            linewidth=2,
            label="Regression Line",
        )

        ax.set_xlabel(selected_var)
        ax.set_ylabel("Units Sold")
        ax.set_title(f"{selected_var} vs Units Sold")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
    with tab2:
        st.subheader("Multiple Linear Regression")

        features = st.multiselect(
            "Select independent variables",
            ["Price", "Discount", "Marketing_Spend"],
            default=["Price", "Discount", "Marketing_Spend"],
        )

        if len(features) >= 1:

            # ================= DATA =================
            X_multi = df[features]
            y_multi = df["Units_Sold"]

            test_size_multi = st.slider(
                "Select test size ratio (Multiple Regression)",
                0.1,
                0.5,
                0.2,
                0.05,
                key="multi_test_size",
            )

            X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
                X_multi, y_multi, test_size=test_size_multi, random_state=42
            )

            # ================= MODEL =================
            multi_model = LinearRegression()
            multi_model.fit(X_train_m, y_train_m)

            y_pred_m = multi_model.predict(X_test_m)

            # ================= EQUATION =================
            equation = f"Units_Sold = {multi_model.intercept_:.2f}"
            for i, col in enumerate(features):
                equation += f" + ({multi_model.coef_[i]:.4f} × {col})"

            st.markdown("### 📌 Regression Equation")
            st.code(equation)

            # ================= METRICS =================
            multi_mae = mean_absolute_error(y_test_m, y_pred_m)
            multi_rmse = np.sqrt(mean_squared_error(y_test_m, y_pred_m))
            multi_r2 = r2_score(y_test_m, y_pred_m)

            metrics_df = pd.DataFrame(
                {
                    "Metric": ["MAE", "RMSE", "R²"],
                    "Value": [multi_mae, multi_rmse, multi_r2],
                }
            )

            st.markdown("### 📊 Model Performance")
            st.table(metrics_df)

            # ================= ACTUAL VS PRED =================
            st.markdown("### 📉 Actual vs Predicted")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y_test_m, y_pred_m, alpha=0.6)
            ax.set_xlabel("Actual Units Sold")
            ax.set_ylabel("Predicted Units Sold")
            ax.set_title("Multiple Regression: Actual vs Predicted")
            ax.grid(True)
            st.pyplot(fig)

        else:
            st.warning("⚠️ Please select at least one feature.")
# =========================================================
# ===================== FORECAST MODEL ====================
# =========================================================
elif page == "Forecasting Model 🚀":
    st.title("🤖 Demand Forecasting — XGBoost")

    features = [
        "Price",
        "Discount",
        "Marketing_Spend",
        "lag_1",
        "lag_7",
        "rolling_7",
        "day_of_week",
        "month",
    ]

    X = df[features]
    y = df["Units_Sold"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    baseline = X_test["lag_1"]

    results = pd.DataFrame(
        {
            "Model": ["Naive Baseline", "XGBoost"],
            "RMSE": [
                np.sqrt(mean_squared_error(y_test, baseline)),
                np.sqrt(mean_squared_error(y_test, pred)),
            ],
        }
    )

    st.table(results)

# =========================================================
# ===================== PRICE OPT =========================
# =========================================================
elif page == "Price Optimization 💰":
    st.title("💰 Revenue Optimization Engine")

    temp = raw_df[(raw_df["Price"] > 0) & (raw_df["Units_Sold"] > 0)].copy()
    temp["log_price"] = np.log(temp["Price"])
    temp["log_demand"] = np.log(temp["Units_Sold"])

    lr = LinearRegression()
    lr.fit(temp[["log_price"]], temp["log_demand"])
    elasticity = lr.coef_[0]

    st.metric("Estimated Price Elasticity", round(elasticity, 3))

# =========================================================
# ===================== 30 DAY ============================
# =========================================================
elif page == "30-Day Forecast 🔮":
    st.title("🔮 30-Day Forecast")

    features = [
        "Price",
        "Discount",
        "Marketing_Spend",
        "lag_1",
        "lag_7",
        "rolling_7",
        "day_of_week",
        "month",
    ]

    model = XGBRegressor().fit(df[features], df["Units_Sold"])

    last_row = df.iloc[-1:].copy()
    future = []

    for _ in range(30):
        p = model.predict(last_row[features])[0]
        future.append(p)
        last_row["lag_1"] = p

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(future, marker="o")
    st.pyplot(fig)

# =========================================================
# ===================== SHAP ==============================
# =========================================================
elif page == "Explainability (SHAP) 🧠":
    st.title("🧠 SHAP Explainability")

    features = [
        "Price",
        "Discount",
        "Marketing_Spend",
        "lag_1",
        "lag_7",
        "rolling_7",
        "day_of_week",
        "month",
    ]

    X = df[features]
    model = XGBRegressor().fit(X, df["Units_Sold"])

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X.iloc[:200])

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)