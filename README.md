This project is a user-friendly interactive dashboard built using Streamlit to help understand and predict sales trends in an e-commerce setting. It offers insights through data visualizations, time series trends, and regression-based sales predictions.
The dataset used in this project is `Modified_Ecommerce_Sales_Prediction.csv`, which includes the following columns:

- Date – Date of the transaction  
- Product Category– Type of product sold  
- Customer Segment– Type of customer (e.g., Regular, New, Premium)  
- Price– Selling price of the product  
- Discount – Discount applied on the product  
- Marketing Spend – Amount spent on promoting the product  
- Units Sold – Number of units sold
This dataset is used to uncover patterns and make predictions about future sales.

Features:

1. Data Overview
- Displays the structure and first few rows of the dataset
- Summary statistics of all numeric columns

2. Exploratory Data Analysis (EDA)
- Visualizations like bar charts, box plots, histograms, and correlation heatmaps
- Insights into sales by product categories and customer segments

3. Time Series Analysis
- Monthly trend visualization of sales
- Moving averages to highlight underlying patterns

4. Regression-Based Prediction
- Single Feature Linear Regression: Predict units sold using one chosen variable  
- Multiple Linear Regression: Predict using a combination of features  
- Performance metrics: MAE, MSE, RMSE, R²

Tech Stack Used
- Python
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for visualizations
- scikit-learn, statsmodelsfor machine learning and statistical modeling
- Streamlit for the interactive dashboard

Getting Started

1. Make sure you have Python installed.
2. Clone this repository or download the files.
3. Place the dataset (`Modified_Ecommerce_Sales_Prediction.csv`) in the same folder as the `app.py` file.
4. Install the required libraries
5. run the streamlit app - streamlit run app.py

