import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import requests
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import numpy as np
from keras.models import load_model
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Vision",
    layout="wide",
    page_icon="💹"
)

# --- App Title ---
st.markdown('<h1 style="text-align: center;">Stock Vision 💹</h1>', unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def prepare_data_for_prediction(ticker, _scaler):
    data = yf.download(ticker, period="5y")
    close_data = data[['Close']].values
    scaled_data = _scaler.transform(close_data)
    last_60 = scaled_data[-60:]
    X_input = np.array([last_60])   # shape (1, 60, 1)
    return data, X_input
def get_stock_data(ticker, start_date, end_date):
    """Fetches stock data from Yahoo Finance."""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            return None, f"No data found for ticker '{ticker}'. It might be delisted or an invalid ticker."
        return stock_data, None
    except Exception as e:
        return None, f"An error occurred while fetching data: {e}"

def calculate_sma(data, window):
    """Calculates the Simple Moving Average (SMA)."""
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI) using a robust method (EMA)."""
    close_delta = data['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ema_up = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    middle_band = calculate_sma(data, window)
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return upper_band, middle_band, lower_band

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def load_lottieurl(url: str):
    """Loads a Lottie animation from a URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Sidebar for User Inputs ---
st.sidebar.header("Stock Selection")
stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]
selected_stock = st.sidebar.selectbox("Select a Stock from the list", stocks)
custom_ticker = st.sidebar.text_input("Or Enter a Custom Ticker").upper()

if custom_ticker:
    ticker = custom_ticker
else:
    ticker = selected_stock

# --- Navigation Menu ---
selected_option = option_menu(
    None,
    ["Home", "Visual Analysis", "News", "Prediction"],
    icons=['house', 'graph-up', 'newspaper', 'lightbulb'],
    menu_icon="cast", default_index=0, orientation="horizontal",
)

# =================================================================================================
# HOME
# =================================================================================================
if selected_option == "Home":
    st.subheader(f"Displaying Daily Price Data for {ticker}")
    col1, col2 = st.columns([1, 2])
    
    lottie_finance_url = "https://lottie.host/19ad9b6a-1882-4957-8216-bafa10a2ceaf/vA8lTy3DBm.json"
    lottie_animation = load_lottieurl(lottie_finance_url)
    with col1:
        if lottie_animation:
            st_lottie(lottie_animation, key="lottie_home", height=200, quality="high")
    
    with col2:
        start_date = st.date_input("Start Date", date.today() - timedelta(days=365))
        end_date = st.date_input("End Date", date.today())
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
        else:
            data, error_message = get_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if error_message:
                st.error(error_message)
            else:
                st.dataframe(data.style.highlight_max(axis=0), height=400)

# =================================================================================================
# VISUAL ANALYSIS
# =================================================================================================
if selected_option == "Visual Analysis":
    st.subheader(f"📊 Visual Analysis for {ticker}")
    
    end_date_visual = date.today()
    start_date_visual = end_date_visual - timedelta(days=730)
    data, error_message = get_stock_data(ticker, start_date_visual, end_date_visual)

    if error_message:
        st.error(error_message)
    else:
        

        # --- Technical Indicators Calculation ---
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data)
        data['RSI_14'] = calculate_rsi(data, 14)
        data['MACD_Line'], data['Signal_Line'], data['MACD_Hist'] = calculate_macd(data)

        # --- Price Chart with Bollinger Bands ---
        st.subheader("Price and Bollinger Bands")
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='Upper Band', line=dict(color='cyan', width=1)))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], mode='lines', name='Middle Band (SMA 20)', line=dict(color='orange', width=1.5, dash='dash')))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='Lower Band', line=dict(color='cyan', width=1)))
        fig_price.update_layout(height=500, template='plotly_dark', xaxis_rangeslider_visible=False, yaxis_title='Price (USD)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_price, use_container_width=True)

        # --- Volume Chart ---
        st.subheader("Trading Volume")
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'))
        fig_volume.update_layout(height=300, template='plotly_dark', yaxis_title='Volume')
        st.plotly_chart(fig_volume, use_container_width=True)

        # --- MACD Chart ---
        st.subheader("Moving Average Convergence Divergence (MACD)")
        st.markdown("- When the **MACD line (blue)** crosses above the **Signal line (orange)**, it's a bullish signal.\n- When it crosses below, it's a bearish signal.")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Line'], name='MACD Line', line=dict(color='blue', width=2)))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], name='Signal Line', line=dict(color='orange', width=2)))
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
        fig_macd.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='Histogram', marker_color=colors))
        fig_macd.update_layout(template='plotly_dark', yaxis_title='Value', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # --- RSI Chart ---
        st.subheader("Relative Strength Index (RSI)")
        st.markdown("- An RSI > 70 is often considered **overbought**.\n- An RSI < 30 is often considered **oversold**.")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI_14'], mode='lines', name='RSI', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(yaxis_title='RSI Value', xaxis_title='Date', template='plotly_dark', yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

# =================================================================================================
# NEWS
# =================================================================================================
if selected_option == "News":
    st.subheader(f"📰 Latest News for {ticker}")
    # NOTE: This uses a hardcoded API key. It's recommended to replace it with your own.
    url = "https://yahoo-finance166.p.rapidapi.com/api/news/list-by-symbol"
    querystring = {"s": ticker, "region": "US", "snippetCount": "10"}
    headers = {
        "x-rapidapi-key": "3407938e63msh859e40926fd9fbbp1fb4b3jsnb030759e322d",
        "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            news_data = response.json()
            articles = news_data.get("data", {}).get("main", {}).get("stream", [])
            if articles:
                for article in articles[:10]:
                    content = article.get("content", {})
                    title = content.get("title", "No Title")
                    pub_date_str = content.get("pubDate", "")
                    link = (content.get("clickThroughUrl") or {}).get("url", "#")
                    provider = (content.get("provider") or {}).get("displayName", "Unknown Source")
                    thumbnail_url = next((res.get("url") for res in (content.get("thumbnail") or {}).get("resolutions", []) if res.get("url")), None)

                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if thumbnail_url:
                            st.image(thumbnail_url, width=150)
                    with col2:
                        st.markdown(f"#### [{title}]({link})")
                        st.caption(f"📰 {provider} | 📅 {pub_date_str}")
                    st.write("---")
            else:
                st.warning("No news articles found.")
        else:
            st.error(f"Failed to fetch news. Status code: {response.status_code}. Check your API key.")
    except Exception as e:
        st.error(f"An error occurred while fetching news: {e}")

# =================================================================================================
# PREDICTION
# =================================================================================================
if selected_option == "Prediction":
    st.subheader(f"📈 Prediction for {selected_stock}")
    try:
        model_files = {"AAPL": ("stock_appl_model_new.keras", "scaler_appl_new.pkl"),
                      "GOOGL":("stock_lstm_model googl.keras","scaler googl.pkl"),
                      "MSFT":("stock_lstm_model msft.keras","scaler msft.pkl"),
                      "AMZN":("stock_lstm_model amzn.keras","scaler msft.pkl"),
                      "TSLA":("stock_lstm_model tsla.keras","scaler tsla.pkl"),
                      "NVDA":("stock_lstm_model nvda.keras","scaler nvda.pkl")}
        if selected_stock not in model_files:
            st.error(f"No trained model found for {selected_stock}.")
        else:
            model_file, scaler_file = model_files[selected_stock]
            model = load_model(model_file)
            scaler = joblib.load(scaler_file)
            data, X_input = prepare_data_for_prediction(selected_stock, scaler)

            def forecast_days(X_input, days):
                preds, current_input = [], X_input.copy()
                for _ in range(days):
                    pred_scaled = model.predict(current_input, verbose=0)
                    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
                    preds.append(pred[0][0])
                    new_scaled = scaler.transform(pred)
                    current_input = np.append(current_input[:, 1:, :], [[new_scaled[0]]], axis=1)
                return preds

            horizons = {"Today & Tomorrow": 2, "Next 7 Days": 7, "Next 30 Days": 30}
            for label, days in horizons.items():
                preds = forecast_days(X_input, days)
                future_dates = pd.date_range(start=date.today(), periods=days)
                df_forecast = pd.DataFrame({"Date": future_dates, "Predicted Close Price": preds})
                st.markdown(f"### 📅 Predicted Stock Prices - {label}")
                st.dataframe(df_forecast)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Predicted Close Price"],
                                         mode="lines+markers", name="Prediction"))
                fig.update_layout(title=f"{selected_stock} - {label}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Could not load model/scaler. Error: {e}")
