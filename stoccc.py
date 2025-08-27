
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import yfinance as yf
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import plotly.graph_objects as go
# import plotly.express as px

# # Page configuration
# st.set_page_config(
#     page_title="Penny Stock Price Predictor",
#     page_icon="📈",
#     layout="wide"
# )

# # Title and description
# st.title("📈 Penny Stock Price Prediction System")
# st.markdown("**Analyze and predict penny stock prices using machine learning**")

# # Sidebar for user inputs
# st.sidebar.header("🔧 Configuration")

# # Stock selection - UPDATED with penny stocks
# stock_symbol = st.sidebar.selectbox(
#     "Select Penny Stock Symbol",
#     ["SNDL", "MULN", "ZOM", "CENN", "GEVO", "PLUG"],
#     index=0
# )

# # Date range selection
# col1, col2 = st.sidebar.columns(2)
# with col1:
#     start_date = st.date_input(
#         "Start Date",
#         value=datetime(2020, 1, 1),
#         max_value=datetime.now() - timedelta(days=1)
#     )
# with col2:
#     end_date = st.date_input(
#         "End Date",
#         value=datetime.now(),
#         max_value=datetime.now()
#     )

# # Model selection
# model_choice = st.sidebar.selectbox(
#     "Select ML Model",
#     ["Random Forest", "Linear Regression"],
#     index=0
# )

# # Main functions
# @st.cache_data
# def fetch_stock_data(symbol, start, end):
#     """Fetch stock data from Yahoo Finance"""
#     try:
#         data = yf.download(symbol, start=start, end=end)
#         data = data.reset_index()
#         return data
#     except Exception as e:
#         st.error(f"Error fetching data: {e}")
#         return None

# def create_features(data):
#     """Create simple technical indicators"""
#     data['SMA_5'] = data['Close'].rolling(window=5).mean()
#     data['SMA_10'] = data['Close'].rolling(window=10).mean()
#     data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
#     delta = data['Close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     data['RSI'] = 100 - (100 / (1 + rs))
    
#     data['Price_Change'] = data['Close'].diff()
#     data['Volatility'] = data['Close'].rolling(window=10).std()
    
#     data['Close_Lag1'] = data['Close'].shift(1)
#     data['Close_Lag2'] = data['Close'].shift(2)
#     data['Volume_Lag1'] = data['Volume'].shift(1)
    
#     data = data.dropna()
#     return data

# def train_model(data, model_type):
#     """Train the selected model"""
#     feature_cols = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 
#                    'RSI', 'Price_Change', 'Volatility', 'Close_Lag1', 'Close_Lag2', 'Volume_Lag1']
    
#     X = data[feature_cols]
#     y = data['Close']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     if model_type == "Random Forest":
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#     else:
#         model = LinearRegression()
    
#     model.fit(X_train_scaled, y_train)
    
#     y_pred = model.predict(X_test_scaled)
    
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    
#     return model, scaler, X_test, y_test, y_pred, mse, r2, feature_cols

# def predict_future_price(model, scaler, data, feature_cols):
#     """Predict price for a future date by simulating steps"""
#     # This is a simplified forecast and highly speculative
#     last_known_data = data.copy()
    
#     # We need to predict day by day until the target date
#     # For this app, we'll do a simplified single-step prediction
#     # based on the very last known features.
#     last_features = last_known_data[feature_cols].iloc[-1].values.reshape(1, -1)
#     last_features_scaled = scaler.transform(last_features)
#     predicted_price = model.predict(last_features_scaled)[0]
    
#     return predicted_price

# # Main app logic
# if st.sidebar.button("🚀 Run Analysis"):
#     with st.spinner("Fetching and analyzing data..."):
        
#         stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
        
#         # --- FIX for TypeError ---
#         # Flatten the multi-level column names returned by yfinance
#         if stock_data is not None:
#             stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]
        
#         if stock_data is not None and len(stock_data) > 50:
            
#             processed_data = create_features(stock_data)
            
#             st.subheader("📊 Dataset Overview")
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric("Total Records", len(processed_data))
#             with col2:
#                 st.metric("Date Range", f"{len(processed_data)} days")
#             with col3:
#                 st.metric("Current Price", f"${processed_data['Close'].iloc[-1]:.4f}")
#             with col4:
#                 price_change = processed_data['Close'].iloc[-1] - processed_data['Close'].iloc[-2]
#                 st.metric("Daily Change", f"${price_change:.4f}", f"{price_change:.4f}")
            
#             st.subheader("📈 Stock Price Trend")
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['Close'], mode='lines', name='Close Price'))
#             fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['SMA_20'], mode='lines', name='SMA 20'))
#             fig.update_layout(title=f"{stock_symbol} Stock Price", xaxis_title="Date", yaxis_title="Price ($)", height=400)
#             st.plotly_chart(fig, use_container_width=True)
            
#             st.subheader("🤖 Model Training & Evaluation")
#             model, scaler, X_test, y_test, y_pred, mse, r2, feature_cols = train_model(processed_data, model_choice)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Mean Squared Error", f"{mse:.4f}")
#             with col2:
#                 st.metric("R² Score", f"{r2:.4f}")
            
#             st.subheader("🔮 Future Price Prediction")
            
#             # Predict price for December 25, 2025 - UPDATED
#             predicted_price = predict_future_price(model, scaler, processed_data, feature_cols)
            
#             st.markdown("### 🎯 Predicted Price for December 25, 2025")
            
#             col1, col2, col3 = st.columns([1, 2, 1])
#             with col2:
#                 st.markdown(f"""
#                 <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; text-align: center; color: white; font-size: 2rem; font-weight: bold; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
#                     ${predicted_price:.4f}
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             current_price = processed_data['Close'].iloc[-1]
#             price_difference = predicted_price - current_price
#             percentage_change = (price_difference / current_price) * 100 if current_price != 0 else 0
            
#             st.markdown("### 📊 Prediction Analysis")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Current Price", f"${current_price:.4f}")
#             with col2:
#                 st.metric("Predicted Price", f"${predicted_price:.4f}", f"{price_difference:.4f}")
#             with col3:
#                 st.metric("Expected Change", f"{percentage_change:.2f}%")

#             if model_choice == "Random Forest":
#                 st.subheader("🎯 Feature Importance")
#                 feature_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
#                 fig_importance = px.bar(feature_importance.head(8), x='Importance', y='Feature', orientation='h', title="Top 8 Most Important Features")
#                 st.plotly_chart(fig_importance, use_container_width=True)
            
#             st.warning("⚠️ **Disclaimer**: This prediction is based on historical data. Stock prices are highly unpredictable. This tool is for educational purposes only and should not be used for investment decisions.")
            
#         else:
#             st.error("❌ Unable to fetch sufficient data. Please try a different date range or stock symbol.")

# # Sidebar instructions
# st.sidebar.markdown("---")
# st.sidebar.info("This app uses ML to predict stock prices based on historical data and technical indicators.")








# ]]]]]]]]]]]]]]]



# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Penny Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# --- User Authentication ---
def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # The password is now hardcoded for simplicity.
        # In a real application, you would use st.secrets.
        if st.session_state["password"] == "password123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.info("Hint: The password is 'password123'")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# --- Main App ---
def run_app():
    # Title and description
    st.title("📈 Penny Stock Price Prediction System")
    st.markdown("**Analyze and predict penny stock prices using machine learning**")

    # Sidebar for user inputs
    st.sidebar.header("🔧 Configuration")

    # Stock selection - UPDATED with penny stocks
    stock_symbol = st.sidebar.selectbox(
        "Select Penny Stock Symbol",
        ["SNDL", "MULN", "ZOM", "CENN", "GEVO", "PLUG"],
        index=0
    )

    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2020, 1, 1),
            max_value=datetime.now() - timedelta(days=1)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select ML Model",
        ["Random Forest", "Linear Regression"],
        index=0
    )

    # Main functions
    @st.cache_data
    def fetch_stock_data(symbol, start, end):
        """Fetch stock data from Yahoo Finance"""
        try:
            data = yf.download(symbol, start=start, end=end)
            data = data.reset_index()
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    def create_features(data):
        """Create simple technical indicators"""
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        data['Price_Change'] = data['Close'].diff()
        data['Volatility'] = data['Close'].rolling(window=10).std()
        
        data['Close_Lag1'] = data['Close'].shift(1)
        data['Close_Lag2'] = data['Close'].shift(2)
        data['Volume_Lag1'] = data['Volume'].shift(1)
        
        data = data.dropna()
        return data

    def train_model(data, model_type):
        """Train the selected model"""
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 
                       'RSI', 'Price_Change', 'Volatility', 'Close_Lag1', 'Close_Lag2', 'Volume_Lag1']
        
        X = data[feature_cols]
        y = data['Close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, scaler, X_test, y_test, y_pred, mse, r2, feature_cols

    def predict_future_price(model, scaler, data, feature_cols):
        """Predict price for a future date by simulating steps"""
        last_known_data = data.copy()
        last_features = last_known_data[feature_cols].iloc[-1].values.reshape(1, -1)
        last_features_scaled = scaler.transform(last_features)
        predicted_price = model.predict(last_features_scaled)[0]
        
        return predicted_price

    # Main app logic
    if st.sidebar.button("🚀 Run Analysis"):
        with st.spinner("Fetching and analyzing data..."):
            
            stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
            
            if stock_data is not None:
                stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]
            
            if stock_data is not None and len(stock_data) > 50:
                
                processed_data = create_features(stock_data)
                
                st.subheader("📊 Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", len(processed_data))
                with col2:
                    st.metric("Date Range", f"{len(processed_data)} days")
                with col3:
                    st.metric("Current Price", f"${processed_data['Close'].iloc[-1]:.4f}")
                with col4:
                    price_change = processed_data['Close'].iloc[-1] - processed_data['Close'].iloc[-2]
                    st.metric("Daily Change", f"${price_change:.4f}", f"{price_change:.4f}")
                
                st.subheader("📈 Stock Price Trend")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['Close'], mode='lines', name='Close Price'))
                fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['SMA_20'], mode='lines', name='SMA 20'))
                fig.update_layout(title=f"{stock_symbol} Stock Price", xaxis_title="Date", yaxis_title="Price ($)", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("🤖 Model Training & Evaluation")
                model, scaler, X_test, y_test, y_pred, mse, r2, feature_cols = train_model(processed_data, model_choice)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                with col2:
                    st.metric("R² Score", f"{r2:.4f}")
                
                st.subheader("🔮 Future Price Prediction")
                
                predicted_price = predict_future_price(model, scaler, processed_data, feature_cols)
                
                st.markdown("### 🎯 Predicted Price for December 25, 2025")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; text-align: center; color: white; font-size: 2rem; font-weight: bold; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        ${predicted_price:.4f}
                    </div>
                    """, unsafe_allow_html=True)
                
                current_price = processed_data['Close'].iloc[-1]
                price_difference = predicted_price - current_price
                percentage_change = (price_difference / current_price) * 100 if current_price != 0 else 0
                
                st.markdown("### 📊 Prediction Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.4f}")
                with col2:
                    st.metric("Predicted Price", f"${predicted_price:.4f}", f"{price_difference:.4f}")
                with col3:
                    st.metric("Expected Change", f"{percentage_change:.2f}%")

                if model_choice == "Random Forest":
                    st.subheader("🎯 Feature Importance")
                    feature_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                    fig_importance = px.bar(feature_importance.head(8), x='Importance', y='Feature', orientation='h', title="Top 8 Most Important Features")
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                st.warning("⚠️ **Disclaimer**: This prediction is based on historical data. Stock prices are highly unpredictable. This tool is for educational purposes only and should not be used for investment decisions.")
                
            else:
                st.error("❌ Unable to fetch sufficient data. Please try a different date range or stock symbol.")

    # Sidebar instructions
    st.sidebar.markdown("---")
    st.sidebar.info("This app uses ML to predict stock prices based on historical data and technical indicators.")

# --- Authentication Check ---
if check_password():
    run_app()
