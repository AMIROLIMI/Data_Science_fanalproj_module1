import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Заголовок
st.title("Анализ данных и классификация ожирения")

# Загрузка данных
@st.cache_data
def load_data():
    url = "https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/raw/main/Obesity%20prediction.csv"
    data = pd.read_csv(url)
    data.drop(columns=["Weight", "SMOKE"], inplace=True)
    col = ["family_history", "FAVC", "SCC"]
    data[col] = data[col].replace({'yes': 1, 'no': 0})
    data["Gender"] = data["Gender"].replace({'Male': 1, 'Female': 0})
    data["Obesity"] = data["Obesity"].replace({
        'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3,
        'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6
    })
    return data

data = load_data()
st.write("### Первые 5 строк данных:")
st.dataframe(data.head())

# Разделение данных
X = data.drop(columns=["Obesity"])
y = data["Obesity"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Загрузка обученной модели
@st.cache_resource
def load_model():
    model_url = "https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/raw/main/knn_model.pkl"
    return joblib.load(model_url)

knn = load_model()

# Визуализация корреляционной матрицы
st.write("### Корреляционная матрица")
fig, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
st.pyplot(fig)

# Анализ важности признаков
st.write("### Важность признаков для модели KNN")
result = permutation_importance(knn, X_scaled, y, n_repeats=10, random_state=42, n_jobs=-1)
importance = result.importances_mean
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
ax.set_xlabel('Важность признаков')
ax.set_ylabel('Признаки')
ax.set_title('Важность признаков для модели KNN')
st.pyplot(fig)

# Форма для пользовательского ввода
st.write("### Предсказание класса ожирения")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))

if st.button("Предсказать"):
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    prediction = knn.predict(user_scaled)
    st.success(f"Предсказанный класс ожирения: {prediction[0]}")
