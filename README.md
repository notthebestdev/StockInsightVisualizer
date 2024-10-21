# Stock Data Visualization 📈

![Project Icon](generated-icon.png)

A comprehensive stock data visualization app using Streamlit with advanced prediction models and interactive charts.

## Features 🚀

- Real-time stock data fetching
- Interactive candlestick charts
- Multiple prediction models:
  - Linear Regression
  - ARIMA
  - LSTM (Long Short-Term Memory)
- Customizable date range selection
- Downloadable summary data in CSV format
- User-friendly interface with sidebar controls

## Installation and Setup 🛠️

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-data-visualization.git
   cd stock-data-visualization
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the Streamlit configuration:
   Create a file `.streamlit/config.toml` with the following content:
   ```toml
   [server]
   headless = true
   address = "0.0.0.0"
   port = 5000
   ```

## Usage Instructions 📊

1. Run the Streamlit app:
   ```
   streamlit run main.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter a stock symbol (e.g., AAPL for Apple Inc.) in the sidebar

4. Select the desired date range and prediction model

5. Explore the interactive chart and summary data

6. Download the CSV file for offline analysis

## Technologies Used 💻

- Python 3.11
- Streamlit
- yfinance
- Pandas
- NumPy
- Plotly
- scikit-learn
- statsmodels
- TensorFlow (Keras)

## Note on GPU Acceleration
Currently, this app runs on CPU only. GPU acceleration is not available in the current environment. This may affect the performance of machine learning models, especially for large datasets or complex predictions.

## Contributing 🤝

We welcome contributions to improve the Stock Data Visualization project. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by TheBestDeveloper
