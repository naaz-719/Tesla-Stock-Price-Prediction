import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# Page Config
st.set_page_config(page_title="Tesla Stock Predictor", layout="wide")

st.title("🚗 Tesla (TSLA) Stock Price Prediction")
st.markdown("Predicting stock behavior using a **Long Short-Term Memory (LSTM)** network.")

# Load Assets
@st.cache_resource
def load_assets():
    # Load the model directly
    model = load_model('tesla_model.keras')
    scaler = joblib.load('scaler.pkl')
    # Load your dataset
    df = pd.read_csv('TSLA (1).csv') 
    df['Date'] = pd.to_datetime(df['Date'])
    return model, scaler, df

try:
    model, scaler, df = load_assets()

    # Sidebar - User Input
    st.sidebar.header("Navigation")
    days_to_show = st.sidebar.slider("Historical Days to View", 100, 1000, 500)

    # Main Dashboard
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Historical Close Price")
        plot_data = df.tail(days_to_show)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(plot_data['Date'], plot_data['Close'], color='#E81010', linewidth=1)
        ax.set_facecolor('#f0f2f6')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

    with col2:
        st.subheader("Predict Next Day")
        # Get the last 60 days to predict the next price
        recent_data = df['Close'].tail(60).values.reshape(-1, 1)
        scaled_input = scaler.transform(recent_data)
        input_seq = np.array([scaled_input])
        
        if st.button("Calculate Forecast"):
            prediction_scaled = model.predict(input_seq)
            prediction = scaler.inverse_transform(prediction_scaled)
            
            current_price = df['Close'].iloc[-1]
            predicted_price = prediction[0][0]
            change = predicted_price - current_price
            
            st.metric(label="Forecasted Price", value=f"${predicted_price:.2f}", delta=f"${change:.2f}")
            st.info("Prediction is based on the 60-day window trend.")

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.warning("Please ensure 'tesla_model.keras', 'scaler.pkl', and 'TSLA (1).csv' are in the same folder as this script.")
