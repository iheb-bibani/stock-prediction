import streamlit as st 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.markdown("""
    <style>
        @keyframes move {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .moving-text {
            animation: move 20s linear infinite;
            text-align: center;
            color: red;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 class="moving-text">STOCK PRICE PREDICTION</h1>
""", unsafe_allow_html=True)

st.subheader('**🗃 Data**')

# Define the directory containing the stock data files
stocks_directory = "stocks"

# Get the list of files in the directory
files = os.listdir(stocks_directory)

# Extract symbols from filenames
symbols = [filename.split('.')[0] for filename in files]

# Create a dictionary to store the dataframes
stock_data = {}

# Read the datasets and store them in the dictionary
for symbol in symbols:
    stock_data[symbol] = pd.read_csv(os.path.join(stocks_directory, f"{symbol}.csv"), index_col=0)

# Example usage
Stock = st.sidebar.selectbox("Select a stock symbol:", symbols)

df_Stock = stock_data[Stock]
df_Stock = df_Stock.rename(columns={'Close(t)':'Close'})
st.write(df_Stock.head())

df_Stock = df_Stock.drop(columns='Date_col')

def create_train_test_set(df_Stock):
    
    st.subheader('Feature Selection')
    selected_features = st.multiselect("Select features:", [col for col in df_Stock.columns if col != 'Close_forcast'], default=['EMA50'])
    
    # Filter the DataFrame to include only the selected features
    features = df_Stock[selected_features]
    target = df_Stock['Close_forcast']

    # Splitting features and target into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = create_train_test_set(df_Stock)

preprocessing_method = st.sidebar.selectbox("Select a preprocessing method:", ['None', 'MinMaxScaler', 'StandardScaler'])

if preprocessing_method == 'MinMaxScaler':
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
elif preprocessing_method == 'StandardScaler':
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    X_train_scaled = X_train
    X_test_scaled = X_test

model = st.sidebar.selectbox("Select a Machine Learning/Deep Learning model:", ['Linear Regression','Random Forest'])

if model == 'Linear Regression':
    st.sidebar.markdown('**Hyperparameters Tuning**')
    fit_intercept = st.sidebar.checkbox('Fit Intercept', value=True)
    normalize = st.sidebar.checkbox('Normalize', value=False)
    alpha = st.sidebar.slider('Alpha', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    
    lr = LinearRegression(fit_intercept=fit_intercept)
    lr.fit(X_train_scaled, Y_train)
    predictions = lr.predict(X_test_scaled)

    # Display feature importance
    st.subheader('Feature Importance')
    coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': lr.coef_})
    coef_df['Absolute Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Absolute Coefficient', ascending=False)
    st.write(coef_df)

    # Create bar chart for feature importance
    fig = px.bar(coef_df, x='Feature', y='Absolute Coefficient', title='Feature Importance', labels={'Feature': 'Feature', 'Absolute Coefficient': 'Absolute Coefficient'})
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)

    # Calculate evaluation metrics
    mse = mean_squared_error(Y_test, predictions)
    mae = mean_absolute_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)

    # Display evaluation metrics
    st.subheader('Evaluation')
    st.success(f'Mean Squared Error (MSE): {round(mse, 2)}')
    st.success(f'Mean Absolute Error (MAE): {round(mae, 2)}')
    st.success(f'R^2 Score: {round(r2, 2)}')

    st.subheader('📈 Chart')
    # Plot predictions vs actual using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(X_train)), y=Y_train.values, mode='lines', name='Training Data', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=np.arange(len(X_train), len(X_train) + len(X_test)), y=Y_test.values, mode='lines', name='Test Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=np.arange(len(X_train), len(X_train) + len(X_test)), y=predictions, mode='lines', name='Predictions', line=dict(color='red')))
    fig.update_xaxes(title_text='Index')
    fig.update_yaxes(title_text='Close Price')
    fig.update_layout(title='Linear Regression Predictions vs Actual')
    st.plotly_chart(fig)
    
elif model == 'Random Forest':
    n_estimators = st.sidebar.slider('Number of Estimators', min_value=10, max_value=200, value=100, step=10)
    max_depth = st.sidebar.slider('Max Depth', min_value=1, max_value=20, value=10)
    min_samples_split = st.sidebar.slider('Min Samples Split', min_value=2, max_value=10, value=2)
    min_samples_leaf = st.sidebar.slider('Min Samples Leaf', min_value=1, max_value=10, value=1)

    rf_model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=42)
    rf_model.fit(X_train_scaled, Y_train)

    rf_predictions = rf_model.predict(X_test_scaled)

    st.subheader('Feature Importance')
    feature_importance = rf_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    st.write(feature_importance_df)

    fig = px.bar(feature_importance_df, x='Feature', y='Importance', title='Feature Importance', labels={'Feature': 'Feature', 'Importance': 'Importance'})
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
    
    st.subheader("Evaluation")
    mse_rf = mean_squared_error(Y_test, rf_predictions)
    mae_rf = mean_absolute_error(Y_test, rf_predictions)
    r2_rf = r2_score(Y_test, rf_predictions)
    st.success(f'Mean Squared Error (MSE): {round(mse_rf, 2)}')
    st.success(f'Mean Absolute Error (MAE): {round(mae_rf, 2)}')
    st.success(f'R^2 Score: {round(r2_rf, 2)}')

    st.subheader('📈 Chart')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(X_train)), y=Y_train.values, mode='lines', name='Training Data', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=np.arange(len(X_train), len(X_train) + len(X_test)), y=Y_test.values, mode='lines', name='Test Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=np.arange(len(X_train), len(X_train) + len(X_test)), y=rf_predictions, mode='lines', name='Random Forest Predictions', line=dict(color='red')))
    fig.update_xaxes(title_text='Index')
    fig.update_yaxes(title_text='Close Price')
    fig.update_layout(title='Random Forest Predictions vs Actual')
    st.plotly_chart(fig)
