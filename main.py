import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Data Comparison", layout="wide")

st.sidebar.title("Stock Data Comparison")
stock_symbols = st.sidebar.text_input("Enter Stock Symbols (comma-separated, e.g., AAPL,GOOGL)", value="AAPL,GOOGL")
date_range = st.sidebar.selectbox("Select Date Range", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])

st.title("Stock Data Comparison")

@st.cache_data
def get_stock_data(symbols, period):
    data = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        data[symbol] = {'df': df, 'info': stock.info}
    return data

try:
    symbols = [symbol.strip() for symbol in stock_symbols.split(',')]
    stock_data = get_stock_data(symbols, date_range)
    
    for symbol, data in stock_data.items():
        df = data['df']
        info = data['info']
        
        st.subheader(f"Summary for {symbol}")
        summary_data = {
            "Open": df['Open'].iloc[-1],
            "High": df['High'].iloc[-1],
            "Low": df['Low'].iloc[-1],
            "Close": df['Close'].iloc[-1],
            "Volume": df['Volume'].iloc[-1],
            "Market Cap": info.get('marketCap', 'N/A'),
        }
        st.table(pd.DataFrame([summary_data]))

    fig = go.Figure()
    for symbol, data in stock_data.items():
        df = data['df']
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=f"{symbol} Close Price"))

    fig.update_layout(
        title="Stock Price Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check if the stock symbols are correct and try again.")

st.markdown("""
## About this app
This Stock Data Comparison app allows you to:
- Compare multiple stocks side by side
- View key financial information for given stocks
- Visualize historical price data with interactive charts

Enter comma-separated stock symbols in the sidebar to get started!
""")

st.markdown('<p style="text-align:center;font-size:0.8em;">Made with ❤️ by TheBestDeveloper</p>', unsafe_allow_html=True)
