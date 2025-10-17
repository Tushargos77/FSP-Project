import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("🌫️ Air Quality Forecasting & Regression App")

# File upload
uploaded_file = st.file_uploader("Upload AirQualityUCI.csv", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file, sep=';', decimal=',')
    st.subheader("Raw Data Preview")
    st.dataframe(data.head())

    # Clean missing data
    data = data.replace(-200, np.nan)
    data = data.fillna(data.select_dtypes(include=np.number).mean())

    # Prophet Forecast
    st.header("📈 Forecasting with Prophet")

    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Time'] = data['Time'].apply(lambda x: x.replace('.', ':'))
    data['ds'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
    prophet_df = pd.DataFrame({'ds': data['ds'], 'y': data['RH']})

    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=48, freq='H')
    forecast = model.predict(future)

    st.subheader("Forecast Preview")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    st.write("### Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.write("### Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Regression Model
    st.header("🔢 Regression Model on Air Quality Data")
    if all(col in data.columns for col in ['CO(GT)', 'NO2(GT)', 'PT08.S5(O3)', 'T', 'AH']):
        X = data[['CO(GT)', 'NO2(GT)', 'PT08.S5(O3)', 'T']]
        y = data['AH']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.3f}")
        st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.3f}")
        st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")

        st.write("### Actual vs Predicted Plot")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Linear Regression - Actual vs Predicted")
        st.pyplot(fig)
    else:
        st.warning("Dataset missing required columns for regression.")
else:
    st.info("👆 Upload the AirQualityUCI.csv file to begin.")
