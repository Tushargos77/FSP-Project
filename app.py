import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------
# Streamlit Title
# ---------------------------------------
st.title("🌫️ Air Quality Forecasting & Regression App")

# ---------------------------------------
# File Upload
# ---------------------------------------
uploaded_file = st.file_uploader("📤 Upload AirQualityUCI.csv", type="csv")

if uploaded_file:
    # Read dataset safely
    data = pd.read_csv(uploaded_file, sep=';', decimal=',')
    st.subheader("📊 Raw Data Preview")
    st.dataframe(data.head())

    # ---------------------------------------
    # Data Cleaning
    # ---------------------------------------
    data = data.replace(-200, np.nan)
    data = data.fillna(data.select_dtypes(include=np.number).mean())

    # ---------------------------------------
    # Prophet Forecasting Section
    # ---------------------------------------
    st.header("📈 Forecasting with Prophet")

    # Convert Date safely
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')

    # Fix Time column (handle floats/NaN before replacing)
    data['Time'] = data['Time'].apply(lambda x: str(x).replace('.', ':') if pd.notnull(x) else None)

    # Combine Date and Time safely
    data['ds'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), errors='coerce')

    # Drop invalid/missing rows for Prophet
    prophet_df = data[['ds', 'RH']].dropna().rename(columns={'RH': 'y'})

    # Show cleaned data
    st.subheader("🧹 Cleaned Data Preview")
    st.dataframe(prophet_df.head())

    # Train Prophet model
    if not prophet_df.empty:
        model = Prophet()
        model.fit(prophet_df)

        # Make 48-hour (2-day) forecast
        future = model.make_future_dataframe(periods=48, freq='H')
        forecast = model.predict(future)

        st.subheader("🔮 Forecast Preview")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        st.write("### Forecast Plot")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.write("### Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
    else:
        st.warning("⚠️ No valid data available for Prophet forecasting after cleaning.")

    # ---------------------------------------
    # Regression Model Section
    # ---------------------------------------
    st.header("🔢 Regression Model on Air Quality Data")

    # Check if required columns are available
    required_cols = ['CO(GT)', 'NO2(GT)', 'PT08.S5(O3)', 'T', 'AH']
    if all(col in data.columns for col in required_cols):
        X = data[['CO(GT)', 'NO2(GT)', 'PT08.S5(O3)', 'T']]
        y = data['AH']

        # Handle NaN in features or target
        valid_rows = X.notnull().all(axis=1) & y.notnull()
        X = X[valid_rows]
        y = y[valid_rows]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Show metrics
        st.subheader("📏 Model Performance Metrics")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.3f}")
        st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.3f}")
        st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")

        # Plot Actual vs Predicted
        st.write("### Actual vs Predicted Plot")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Linear Regression - Actual vs Predicted")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Dataset missing required columns for regression.")
else:
    st.info("👆 Upload the **AirQualityUCI.csv** file to begin.")
