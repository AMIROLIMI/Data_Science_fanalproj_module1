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
st.markdown("""
## О наборе данных
Этот набор данных включает данные для оценки уровня ожирения у людей из стран Мексики, Перу и Колумбии на основе их привычек питания и физического состояния.
Данные содержат 17 атрибутов и 2111 записей, записи помечены переменной класса NObesity (уровень ожирения), что позволяет классифицировать данные с использованием значений: 
Недостаточный вес (0 - Insufficient Weight), Нормальный вес (1 - Normal Weight), Избыточный вес I уровня (2 - Overweight Level I), Избыточный вес II уровня (3 - Overweight Level II), Ожирение I типа (4 - Obesity Type I), Ожирение II типа (5 - Obesity Type II) и Ожирение III типа (6 - Obesity Type III).

### Подробности данных:
- Gender: Пол
- Age: Возраст
- Height: в метрах
- Weight: в кг
- family_history : Страдал ли кто-либо из членов семьи от избыточного веса?
- FAVC: Часто ли вы едите высококалорийную пищу?
- FCVC: Вы обычно едите овощи?
- NCP: Сколько основных приемов пищи у вас в день?
- CAEC: Вы едите что-нибудь между приемами пищи?
- SMOKE: Вы курите?
- CH2O: Сколько воды вы пьете ежедневно?
- SCC: Вы следите за количеством потребляемых ежедневно калорий?
- FAF: Как часто вы занимаетесь физической активностью?
- TUE: Сколько времени вы используете технологические устройства, такие как мобильный телефон, видеоигры, телевизор, компьютер и т. д.?
- CALC: Как часто вы употребляете алкоголь?
- MTRANS: Какой транспорт вы обычно используете?
- Obesity_level (Целевой столбец): Уровень ожирения """)
# Функция для загрузки данных с кэшированием
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

data_raw = load_data("https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/raw/main/Obesity%20prediction.csv")
st.write("### Данные до изменения")
st.dataframe(data_raw.head())
st.write("### Матрица корреляции до обработки данных")
st.image("https://raw.githubusercontent.com/AMIROLIMI/Data_Science_fanalproj_module1/main/CORR_before.png", 
         use_container_width=True)

data = load_data("https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/raw/main/Encoded%20Standardized%20Obesity%20prediction.csv")
data = data.drop(columns=["Unnamed: 0"])
st.write("### Данные после изменения")
st.dataframe(data.head())
st.write("### Матрица корреляции после обработки данных")
st.image("https://raw.githubusercontent.com/AMIROLIMI/Data_Science_fanalproj_module1/main/CORR_after.png", 
         use_container_width=True)

X = data.drop(columns=["Obesity"])
y = data["Obesity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

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

# Важность признаков
st.write("### Важность признаков для модели KNN")
st.image("https://raw.githubusercontent.com/AMIROLIMI/Data_Science_fanalproj_module1/main/feature_imp.png", 
         use_container_width=True)

# Матрица ошибок
st.write("### Матрица ошибок")
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
ax.set_xlabel("Предсказанный класс")
ax.set_ylabel("Истинный класс")
ax.set_title("Матрица ошибок")
st.pyplot(fig)

# Отчёт классификации
st.write("### Отчёт классификации")
st.text(classification_report(y_test, y_pred))

# Форма для пользовательского ввода
st.write("### Введите данные для предсказания")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))

if st.button("Предсказать"):
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    prediction = knn.predict(user_scaled)
    st.success(f"Предсказанный класс ожирения: {prediction[0]}")
