import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import base64
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Stock Data Visualization",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("Stock Data Visualization")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
date_range = st.sidebar.selectbox("Select Date Range", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
prediction_days = st.sidebar.slider("Prediction Days", min_value=7, max_value=365, value=30, step=1)
model_choice = st.sidebar.selectbox("Select Prediction Model", ["Linear Regression", "ARIMA", "LSTM"])

# Main content
st.title(f"Stock Data for {stock_symbol}")

@st.cache_data
def get_stock_data(symbol, period):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df, stock.info

def linear_regression_prediction(df, prediction_days):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)

    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=prediction_days)
    future_X = np.arange(len(df), len(df) + prediction_days).reshape(-1, 1)
    future_prices = model.predict(future_X)

    return future_dates, future_prices

def arima_prediction(df, prediction_days):
    model = ARIMA(df['Close'], order=(1, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=prediction_days)
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=prediction_days)
    return future_dates, forecast

def lstm_prediction(df, prediction_days):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)

    inputs = df['Close'].values[-60:].reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    future_prices = []
    current_batch = inputs[-60:].reshape((1, 60, 1))
    for i in range(prediction_days):
        current_pred = model.predict(current_batch)[0]
        future_prices.append(current_pred[0])
        current_batch = np.roll(current_batch, -1, axis=1)
        current_batch[0, -1, 0] = current_pred[0]
    
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=prediction_days)
    return future_dates, future_prices.flatten()

try:
    df, info = get_stock_data(stock_symbol, date_range)
    
    # Summary table
    st.subheader("Summary")
    summary_data = {
        "Open": df['Open'].iloc[-1],
        "High": df['High'].iloc[-1],
        "Low": df['Low'].iloc[-1],
        "Close": df['Close'].iloc[-1],
        "Volume": df['Volume'].iloc[-1],
        "Market Cap": info.get('marketCap', 'N/A'),
        "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
        "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
    }
    summary_df = pd.DataFrame([summary_data])
    st.table(summary_df)

    # Download CSV
    csv = summary_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{stock_symbol}_summary.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Price prediction
    if model_choice == "Linear Regression":
        future_dates, future_prices = linear_regression_prediction(df, prediction_days)
    elif model_choice == "ARIMA":
        future_dates, future_prices = arima_prediction(df, prediction_days)
    else:  # LSTM
        future_dates, future_prices = lstm_prediction(df, prediction_days)

    # Plotting
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('Stock Price', 'Volume'), row_width=[0.2, 0.7])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Stock Price"), row=1, col=1)

    # Prediction line
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name=f'{model_choice} Prediction', line=dict(color='red', dash='dash')), row=1, col=1)

    # Volume chart
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume"), row=2, col=1)

    fig.update_layout(
        title=f"{stock_symbol} Stock Data",
        xaxis_title="Date",
        yaxis_title="Price",
        height=800,
        width=1200,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check if the stock symbol is correct and try again.")

# App description
st.markdown("""
## About this app
This Stock Data Visualization app allows you to:
- View key financial information for a given stock
- Visualize historical price data with an interactive chart
- Download the summary data as a CSV file
- See price predictions using different models (Linear Regression, ARIMA, and LSTM)

Enter a stock symbol in the sidebar to get started!
""")

st.markdown('<p style="text-align:center;font-size:0.8em;">Made with ❤️ by TheBestDeveloper</p>', unsafe_allow_html=True)
