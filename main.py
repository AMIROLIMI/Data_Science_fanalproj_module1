import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report
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
result = permutation_importance(knn, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importance = result.importances_mean
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
ax.set_xlabel('Важность признаков')
ax.set_ylabel('Признаки')
ax.set_title('Важность признаков для модели KNN')
st.pyplot(fig)

## Человеческие названия классов
class_labels = {
    0: "Недостаточный вес",
    1: "Нормальный вес",
    2: "Избыточный вес I уровня",
    3: "Избыточный вес II уровня",
    4: "Ожирение I типа",
    5: "Ожирение II типа",
    6: "Ожирение III типа"
}

# Ввод данных пользователем
st.write("### Введите данные для предсказания")

user_input = {}

# Пол
gender_map = {"Мужчина": 1, "Женщина": 0}
gender_choice = st.radio("Пол", list(gender_map.keys()))
user_input["Gender"] = gender_map[gender_choice]

# Возраст
user_input["Age"] = st.slider("Возраст", min_value=10, max_value=80, value=30)

# Рост
user_input["Height"] = st.slider("Рост (м)", min_value=1.2, max_value=2.2, value=1.7)

# Семейная история ожирения (Да/Нет)
binary_map = {"Да": 1, "Нет": 0}
user_input["family_history"] = binary_map[st.radio("Есть ли у семьи история ожирения?", list(binary_map.keys()))]

# Часто ли вы едите высококалорийную пищу? (Да/Нет)
user_input["FAVC"] = binary_map[st.radio("Часто ли едите высококалорийную пищу?", list(binary_map.keys()))]

# Употребление овощей (0-3)
user_input["FCVC"] = st.slider("Как часто вы едите овощи?", 0.0, 3.0, 1.5)

# Основные приемы пищи в день (1-4)
user_input["NCP"] = st.slider("Сколько основных приемов пищи у вас в день?", 1.0, 4.0, 3.0)

# Еда между приемами пищи (Категории)
caec_options = {"Никогда": 0, "Редко": 1, "Иногда": 2, "Часто": 3}
user_input["CAEC"] = caec_options[st.selectbox("Едите ли вы между приемами пищи?", list(caec_options.keys()))]

# Сколько воды вы пьете ежедневно? (0-3)
user_input["CH2O"] = st.slider("Сколько воды пьете ежедневно? (литры)", 0.0, 3.0, 1.5)

# Контроль калорий (Да/Нет)
user_input["SCC"] = binary_map[st.radio("Контролируете ли калории?", list(binary_map.keys()))]

# Физическая активность (0-3)
user_input["FAF"] = st.slider("Как часто занимаетесь физической активностью?", 0.0, 3.0, 1.0)

# Время перед экранами (0-2)
user_input["TUE"] = st.slider("Сколько времени проводите за гаджетами (часы)?", 0.0, 2.0, 1.0)

# Употребление алкоголя (Категории)
calc_options = {"Никогда": 0, "Редко": 1, "Иногда": 2, "Часто": 3}
user_input["CALC"] = calc_options[st.selectbox("Как часто употребляете алкоголь?", list(calc_options.keys()))]

# Тип транспорта (Категории)
mtrans_options = {"Авто": 0, "Мотоцикл": 1, "Байк": 2, "Пешком": 3, "Общественный транспорт": 4}
user_input["MTRANS"] = mtrans_options[st.selectbox("Какой транспорт используете чаще всего?", list(mtrans_options.keys()))]

# Кнопка предсказания
if st.button("Предсказать"):
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    prediction = knn.predict(user_scaled)[0]

    st.success(f"**Предсказанный класс ожирения: {class_labels[prediction]}**")
