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
data = pd.read_csv("https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/blob/main/Encoded%20Standardized%20Obesity%20prediction.csv")
st.write("### Первые 5 строк данных:")
st.dataframe(data.head())

X = data.drop(columns=["Obesity"])
y = data["Obesity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
@st.cache_resource
knn = joblib.load("https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/raw/main/knn_model.pkl")

# Визуализация корреляционной матрицы
st.write("### Корреляционная матрица")
fig, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
st.pyplot(fig)

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
