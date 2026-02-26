import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Diabetes Prediction using Linear Regression")
# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared Score: {r2:.2f}")

# Create interactive plots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("True vs Predicted",
                                    "BMI vs Predicted"))
# True vs Predicted
fig.add_trace(
    go.Scatter(x=y_test, y=y_pred,
               mode='markers',
               marker=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=[y_test.min(), y_test.max()],
               y=[y_test.min(), y_test.max()],
               mode='lines',
               line=dict(dash='dash', color='black')),
    row=1, col=1
)
# BMI vs Predicted
fig.add_trace(
    go.Scatter(x=X_test[:, 2], y=y_pred,
               mode='markers',
               marker=dict(color='green')),
    row=1, col=2
)

fig.update_layout(height=500, width=1000)

st.plotly_chart(fig)
