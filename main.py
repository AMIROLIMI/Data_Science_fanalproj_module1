import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

st.title("Классификация ожирения")
@st.cache_data
data = pd.read_csv("https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/blob/main/Obesity%20prediction.csv")
st.write("### Данные до изменения")
st.dataframe(data.head())

data = pd.read_csv("https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/blob/main/Encoded%20Standardized%20Obesity%20prediction.csv")
st.write("### Данные после изменения")
st.dataframe(data.head())

X = data.drop(columns=["Obesity"])
y = data["Obesity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
