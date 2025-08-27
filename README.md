# 📈 Penny Stock Price Prediction System

A comprehensive machine learning-powered web application for analyzing and predicting penny stock prices using historical data and technical indicators.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

<img width="1676" height="902" alt="Screenshot 2025-08-27 at 10 35 48 PM" src="https://github.com/user-attachments/assets/a58edd4a-4b9e-477a-972e-89042e434f90" />

## 🌟 Features

- **Real-time Data Fetching**: Automatically retrieves historical stock data using Yahoo Finance API
- **Multiple ML Models**: Choose between Random Forest and Linear Regression algorithms
- **Technical Indicators**: Implements SMA, RSI, volatility, and lag features
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **Feature Importance Analysis**: Understand which factors most influence predictions
- **Secure Access**: Password-protected application
- **Responsive Design**: Clean, intuitive user interface

## 🎯 Supported Penny Stocks

- **SNDL** - Sundial Growers Inc.
- **MULN** - Mullen Automotive Inc.
- **ZOM** - Zomedica Corp.
- **CENN** - Cenntro Electric Group Ltd.
- **GEVO** - Gevo Inc.
- **PLUG** - Plug Power Inc.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager



## 📦 Dependencies

```
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.11.0
yfinance>=0.2.0
scikit-learn>=1.2.0
plotly>=5.15.0
```

## 🔧 Usage

### Basic Workflow

1. **Authentication**: Enter the password to access the application
2. **Stock Selection**: Choose from available penny stocks in the sidebar
3. **Date Range**: Set your analysis period (minimum 50 days recommended)
4. **Model Selection**: Pick between Random Forest or Linear Regression
5. **Analysis**: Click "Run Analysis" to fetch data and train the model
6. **Results**: View predictions, metrics, and visualizations

### Key Components

#### Technical Indicators
- **SMA (5, 10, 20)**: Simple Moving Averages
- **RSI**: Relative Strength Index (14-period)
- **Volatility**: 10-day rolling standard deviation
- **Lag Features**: Previous day price and volume data

#### Model Evaluation Metrics
- **MSE**: Mean Squared Error
- **R² Score**: Coefficient of determination
- **Feature Importance**: Ranking of input variables (Random Forest only)

## 📊 Model Architecture

### Feature Engineering
The application creates 13 key features from raw OHLCV data:

- Basic OHLCV (Open, High, Low, Close, Volume)
- Technical indicators (SMA_5, SMA_10, SMA_20, RSI)
- Market dynamics (Price_Change, Volatility)
- Temporal features (Close_Lag1, Close_Lag2, Volume_Lag1)

### Machine Learning Models

#### Random Forest Regressor
- **Ensemble method** with 100 decision trees
- **Handles non-linear relationships** effectively
- **Feature importance analysis** included
- **Robust to outliers** and missing data

#### Linear Regression
- **Simple baseline model** for comparison
- **Fast training and prediction**
- **Interpretable coefficients**
- **Good for linear relationships**

## 🎨 User Interface

### Dashboard Components
- **Configuration Panel**: Stock selection, date range, model choice
- **Dataset Overview**: Key statistics and metrics
- **Price Visualization**: Interactive time series charts
- **Model Performance**: Evaluation metrics display
- **Prediction Results**: Future price forecast with analysis
- **Feature Analysis**: Importance rankings (Random Forest)

## ⚠️ Disclaimer

**Important**: This application is designed for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.

## 🔒 Security Notes

- Default password is set to `password123` for demonstration
- In production, use environment variables or Streamlit secrets for authentication
- Consider implementing more robust authentication for real-world deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Known Issues

- Model performance may vary significantly with market volatility
- Limited to selected penny stocks (expandable)
- Requires minimum 50 days of historical data for reliable predictions

## 🔮 Future Enhancements

- [ ] Support for additional stock symbols
- [ ] Advanced ML models (LSTM, XGBoost)
- [ ] Real-time price streaming
- [ ] Portfolio optimization features
- [ ] Alert system for price targets
- [ ] Export functionality for predictions
- [ ] Backtesting capabilities




## 🙏 Acknowledgments

- [Yahoo Finance API](https://pypi.org/project/yfinance/) for providing stock data
- [Streamlit](https://streamlit.io/) for the web framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms

---

**⭐ Star this repository if you found it helpful!**
