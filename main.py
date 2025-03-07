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
