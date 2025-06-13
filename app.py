import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px

# Title
st.title("📊 Student Performance Prediction Dashboard")

# Sidebar
st.sidebar.header("Settings")
n = st.sidebar.slider("Number of Students in Dataset",
                      min_value=200, max_value=2000, value=1000, step=100)

# Set random seed for reproducibility
np.random.seed(42)

# Create dataset with realistic-looking data
data = pd.DataFrame({
    'Study Hours': np.random.normal(5, 2, n).clip(0, 10),
    'Sleep Hours': np.random.normal(6.5, 1.5, n).clip(3, 10),
    'Attendance %': np.random.randint(50, 100, n),
    'Participation': np.random.randint(0, 2, n)
})

# Generate target variable using a weighted scoring system
data['Performance Score'] = (
    data['Study Hours'] * 3 +
    data['Sleep Hours'] * 2 +
    data['Attendance %'] * 0.5 +
    data['Participation'] * 5
)

# Threshold for passing
data['Result'] = (data['Performance Score'] > 80).astype(int)
data.drop('Performance Score', axis=1, inplace=True)

# Preview
st.subheader("🔍 Data Preview")
st.dataframe(data.head())

# EDA (Exploratory Data Analysis)
st.subheader("📈 Pairplot of Features vs Result")
fig_pair = sns.pairplot(data, hue='Result')
plt.suptitle("Pairplot of Features vs Result", y=1.02)

# Convert matplotlib figure to Streamlit
st.pyplot(fig_pair)

#  Prepare Data & Train-Test Split
X = data.drop('Result', axis=1)
y = data['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Train the random forest model
# Model setup
model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Model Evaluation
st.subheader("✅ Model Evaluation")
st.write(f"**Accuracy Score:** {accuracy_score(y_test, y_pred):.2f}")

# Classification Report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

# Optional: Show actual vs predicted in a table
result_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
st.subheader("🧾 Actual vs Predicted (Test Set)")
st.dataframe(result_df.head(15))

# 7. Feature Importance/Extraction with Plotly
# Extract feature
importance_scores = model.feature_importances_
feature_labels = X.columns

# Create a bar chart using Plotly
bar_chart = go.Figure([go.Bar(x=feature_labels, y=importance_scores)])

# Customize the chart
bar_chart.update_layout(
    title="🔍 Feature Importance - What Affects Result the Most?",
    xaxis_title="Feature",
    yaxis_title="Importance Score",
    template="plotly_dark"
)

st.subheader("🔍 Feature Importance (Interactive Plot)")
st.plotly_chart(bar_chart, use_container_width=True)
