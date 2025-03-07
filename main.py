import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import urllib.request

st.title("Классификация ожирения")

# Функция для загрузки данных с кэшированием
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

# Загрузка данных до обработки
data_raw = load_data("https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/raw/main/Obesity%20prediction.csv")
st.write("### Данные до изменения")
st.dataframe(data_raw.head())

# Загрузка обработанных данных
data = load_data("https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/raw/main/Encoded%20Standardized%20Obesity%20prediction.csv")
data = data.drop(columns=["Unnamed: 0"])
st.write("### Данные после изменения")
st.dataframe(data.head())

# Разделение данных
X = data.drop(columns=["Obesity"])
y = data["Obesity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Функция для загрузки модели
@st.cache_resource
def load_model():
    model_url = "https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/raw/main/knn_model.pkl"
    model_path = "knn_model.pkl"
    urllib.request.urlretrieve(model_url, model_path)
    return joblib.load(model_path)

knn = load_model()
