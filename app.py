# app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Badminton Play Prediction App")

# Step 1: Load dataset from Kaggle input path
DATA_PATH = '/kaggle/input/badminton/badminton_dataset.csv'
df = pd.read_csv(DATA_PATH)

st.subheader("Original Dataset")
st.dataframe(df)

# Step 2: Check for missing values
st.subheader("Missing Values in Dataset")
st.dataframe(df.isnull().sum())

# Step 3: Encode categorical variables
le = LabelEncoder()
df_encoded = df.copy()
for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':
        df_encoded[column] = le.fit_transform(df_encoded[column])

st.subheader("Encoded Dataset")
st.dataframe(df_encoded)

# Step 4: Train Decision Tree model
X = df_encoded.drop('Play_Badminton', axis=1)
y = df_encoded['Play_Badminton']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.text(classification_report(y_test, y_pred))

# Step 5: User input for prediction
st.subheader("Predict Play Badminton for New Data")
st.write("Enter feature values:")

# Create dropdowns for categorical columns
outlook = st.selectbox("Outlook", df['Outlook'].unique())
temperature = st.selectbox("Temperature", df['Temperature'].unique())
humidity = st.selectbox("Humidity", df['Humidity'].unique())
wind = st.selectbox("Wind", df['Wind'].unique())

if st.button("Predict"):
    new_data = pd.DataFrame({
        'Outlook': [outlook],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Wind': [wind]
    })

    # Encode new data using the same LabelEncoder
    for column in new_data.columns:
        new_data[column] = le.fit_transform(new_data[column])

    prediction = model.predict(new_data)
    st.success(f"Predicted Play Badminton: {'Yes' if prediction[0] == 1 else 'No'}")

