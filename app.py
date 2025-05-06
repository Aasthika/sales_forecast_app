import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st



st.set_page_config(
    page_title="E-commerce Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_data():
    df = pd.read_csv('C:/Users/Admin/Downloads/Modified_Ecommerce_Sales_Prediction.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['total_sales'] = df['Price'] * df['Units_Sold']
    return df

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Time Series Analysis", "Regression Analysis"])

def plot_distribution(column, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column], bins=30, kde=True, ax=ax, color="blue")
    ax.set_title(title)
    st.pyplot(fig)

def plot_scatter(x, y, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=df[x], y=df[y], alpha=0.6, ax=ax, color="blue")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    st.pyplot(fig)

def plot_time_series(data, x, y, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=data, x=x, y=y, marker="o", linewidth=2.5, ax=ax, color="darkblue")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(y)
    ax.grid(True)
    st.pyplot(fig)


if page == "Data Overview":
    st.title("E-commerce Sales Data Overview")
    
    st.subheader("First 5 Rows of Data")
    st.write(df.head())
    
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    st.subheader("Data Columns")
    st.write(df.columns.tolist())

elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Relationships", "Correlations", "Segment Analysis"])
    
    with tab1:
        st.subheader("Numerical Features Distribution")
        numerical_cols = ["Price", "Discount", "Marketing_Spend", "Units_Sold"]
        selected_col = st.selectbox("Select a numerical column to visualize", numerical_cols)
        plot_distribution(selected_col, f"Distribution of {selected_col}")
        
        st.subheader("Categorical Features Distribution")
        categorical_cols = ["Product_Category", "Customer_Segment"]
        selected_cat = st.selectbox("Select a categorical column to visualize", categorical_cols)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(x=selected_cat, data=df, palette="deep", ax=ax)
        ax.set_title(selected_cat)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Feature Relationships")
        
        scatter_pairs = [
            ("Price", "Units_Sold"),
            ("Marketing_Spend", "Units_Sold"),
            ("Discount", "Units_Sold"),
            ("Price", "Marketing_Spend")
        ]
        
        selected_pair = st.selectbox("Select variable pair to visualize", scatter_pairs, format_func=lambda x: f"{x[0]} vs {x[1]}")
        plot_scatter(selected_pair[0], selected_pair[1], f"{selected_pair[0]} vs {selected_pair[1]}")
        
    with tab3:
        st.subheader("Correlation Analysis")
        
        numerical_features = ["Price", "Discount", "Marketing_Spend", "Units_Sold"]
        selected_features = st.multiselect("Select features for correlation", numerical_features, default=numerical_features)
        
        if len(selected_features) >= 2:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            corr_matrix = df[selected_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)
            
               
    with tab4:
        st.subheader("Segment Analysis")
        
        sales_by_category_segment = df.groupby(["Product_Category", "Customer_Segment"]).agg(
            Total_Sales=("total_sales", "sum"), 
            Total_Units_Sold=("Units_Sold", "sum"),
            Total_Marketing_Spend=("Marketing_Spend", "sum")
        ).reset_index()
        
        metric = st.selectbox("Select metric to visualize", ["Total_Sales", "Total_Units_Sold", "Total_Marketing_Spend"])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Product_Category", y=metric, hue="Customer_Segment", data=sales_by_category_segment)
        plt.title(f"{metric.replace('_', ' ')} by Category and Customer Segment", fontsize=14)
        plt.xlabel("Product Category")
        plt.ylabel(metric.replace('_', ' '))
        plt.legend(title='Customer Segment', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

elif page == "Time Series Analysis":
    st.title("Time Series Analysis")
    
    df_time = df.set_index('Date')
    monthly_sales = df_time.groupby(pd.Grouper(freq='M'))['total_sales'].sum().reset_index()
    monthly_sales.set_index('Date', inplace=True)
   
 
    st.subheader("Monthly Sales Trend")
    plot_time_series(monthly_sales.reset_index(), "Date", "total_sales", "Monthly Total Sales Over Time")
        
    st.subheader("Moving Average Trend")
    window = st.slider("Select moving average window (months)", 2, 12, 3)
    monthly_sales[f'{window}_month_MA'] = monthly_sales['total_sales'].rolling(window=window).mean()
        
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly_sales['total_sales'], marker="o", linewidth=1.5, ax=ax, color="gray", label="Monthly Sales")
    sns.lineplot(data=monthly_sales[f'{window}_month_MA'], marker="o", linewidth=2.5, ax=ax, color="red", 
                    label=f"{window}-Month Moving Avg")
    ax.set_title(f"Sales with {window}-Month Moving Average", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.grid(True)
    st.pyplot(fig)
    
    

elif page == "Regression Analysis":
    st.title("Regression Analysis")
    
    tab1, tab2 = st.tabs(["Single Variable Regression", "Multiple Regression"])
    
    with tab1:
        st.subheader("Single Variable Linear Regression")
        
        independent_vars = ["Price", "Discount", "Marketing_Spend"]
        selected_var = st.selectbox("Select independent variable", independent_vars)
        
        X = df[[selected_var]]
        y = df['Units_Sold']
        
        test_size = st.slider("Select test size ratio", 0.1, 0.5, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        y_pred = lr_model.predict(X_test)
        
        intercept = lr_model.intercept_
        coef = lr_model.coef_[0]
        st.write(f"*Regression Equation:* Units_Sold = {intercept:.2f} + {coef:.4f} * {selected_var}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual Data')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
        plt.title(f'Linear Regression: {selected_var} vs Units Sold')
        plt.xlabel(selected_var)
        plt.ylabel('Units Sold')
        plt.legend()
        plt.grid()
        st.pyplot(fig)
        
        st.subheader("Model Performance Metrics")
        
        lr_mae = mean_absolute_error(y_test, y_pred)
        lr_mse = mean_squared_error(y_test, y_pred)
        lr_rmse = np.sqrt(lr_mse)
        lr_r2 = r2_score(y_test, y_pred)
        
        metrics_df = pd.DataFrame({
            "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", 
                      "Root Mean Squared Error (RMSE)", "R-squared (R²)"],
            "Value": [lr_mae, lr_mse, lr_rmse, lr_r2]
        })
        
        st.table(metrics_df)
    
    with tab2:
        st.subheader("Multiple Linear Regression")
        
        features = st.multiselect("Select independent variables", 
                                ["Price", "Discount", "Marketing_Spend"],
                                default=["Price", "Discount", "Marketing_Spend"])
        
        if len(features) >= 1:
            X_multi = df[features]
            y_multi = df['Units_Sold']
            
            test_size = st.slider("Select test size ratio for multiple regression", 0.1, 0.5, 0.2, 0.05)
            X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, 
                                                                       test_size=test_size, 
                                                                       random_state=42)
            
            multi_model = LinearRegression()
            multi_model.fit(X_train_m, y_train_m)
            
            y_pred_m = multi_model.predict(X_test_m)
            
            equation = f"Units_Sold = {multi_model.intercept_:.2f} + "
            for i, col in enumerate(features):
                equation += f"{multi_model.coef_[i]:.4f} * {col} + "
            equation = equation[:-3]  # Remove the last ' + '
            
            st.write("*Regression Equation:*")
            st.write(equation)
            
            st.subheader("Model Performance Metrics")
            
            multi_mae = mean_absolute_error(y_test_m, y_pred_m)
            multi_mse = mean_squared_error(y_test_m, y_pred_m)
            multi_rmse = np.sqrt(multi_mse)
            multi_r2 = r2_score(y_test_m, y_pred_m)
            
            metrics_df = pd.DataFrame({
                "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", 
                          "Root Mean Squared Error (RMSE)", "R-squared (R²)"],
                "Value": [multi_mae, multi_mse, multi_rmse, multi_r2]
            })
            
            st.table(metrics_df)
            
            
         

