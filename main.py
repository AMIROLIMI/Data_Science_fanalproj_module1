import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from itertools import cycle
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

data= load_data("https://github.com/AMIROLIMI/Data_Science_fanalproj_module1/raw/main/Obesity%20prediction.csv")
st.write("### Данные до изменения")
st.dataframe(data.head())

import streamlit as st
import pandas as pd
import io

# Основная информация о датасете
st.write("## Основная информация о датасете")

buffer = io.StringIO()
data.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

# Вывод формы (размерности) данных
st.write(f"### Размерность данных: {data.shape[0]} строк, {data.shape[1]} колонок")

# Подсчёт пропущенных значений
st.write("### Количество пропущенных значений в каждом столбце")
st.dataframe(data.isna().sum().reset_index().rename(columns={"index": "Столбец", 0: "Пропущенные значения"}))

# Уникальные значения в каждом столбце
st.write("### Количество уникальных значений в каждом столбце")
st.dataframe(data.nunique().reset_index().rename(columns={"index": "Столбец", 0: "Уникальные значения"}))

# Уникальные значения в колонке "Obesity"
st.write("### Уникальные классы в 'Obesity'")
st.write(", ".join(map(str, data["Obesity"].unique())))

# Описание статистики данных
st.write("### Описательная статистика данных")
st.dataframe(data.describe())

# Список колонок
st.write("### Список колонок в датасете")
st.write(", ".join(data.columns))



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

metrics_model = pd.DataFrame({'Metric': ['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 'Train Precision', 'Test Precision', 'Train Recall', 'Test Recall', 'Train F1 Score', 'Test F1 Score', 'Cross Validation Mean']})

def evaluate_metrics(model, model_name):
    cv_scores = cross_val_score(model, X, y, cv=50, scoring='accuracy')

    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr')
    train_precision = precision_score(y_train, y_train_pred, average='macro')
    train_recall = recall_score(y_train, y_train_pred, average='macro')
    train_f1 = f1_score(y_train, y_train_pred, average='macro')

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr')
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    metrics_model[model_name] = [train_accuracy, test_accuracy, train_roc_auc, test_roc_auc, train_precision, test_precision, 
                                 train_recall, test_recall, train_f1, test_f1, cv_scores.mean()]

evaluate_metrics(knn, "KNN")
st.write("### Метрики модели KNN")
st.dataframe(metrics_model.head(11))

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

from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
import numpy as np

# Определение меток классов
class_labels = {
    0: "Недостаточный вес",
    1: "Нормальный вес",
    2: "Избыточный вес I уровня",
    3: "Избыточный вес II уровня",
    4: "Ожирение I типа",
    5: "Ожирение II типа",
    6: "Ожирение III типа"
}

st.write("### ROC-кривые для каждой категории ожирения")

# Получение вероятностей предсказаний
y_score = knn.predict_proba(X_test)

# Определение уникальных классов
n_classes = 7
colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "purple", "brown"])

fig, ax = plt.subplots(figsize=(10, 6))

# Построение ROC-кривых и AUC для каждого класса
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test == i, y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{class_labels.get(i, i)} (AUC = {roc_auc:.2f})')

# Расчет усредненного ROC AUC
average_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
st.write(f'### Усредненный ROC AUC: {average_auc:.2f}')

# Добавление диагональной линии (случайный предсказатель)
ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC-кривые')
ax.legend(loc="lower right")

st.pyplot(fig)




# Ввод данных пользователем
st.write("### Введите данные для предсказания")
user_input = {}
gender_map = {"Мужчина": 1, "Женщина": 0}
gender_choice = st.radio("Пол (Gender)", list(gender_map.keys()))
user_input["Gender"] = gender_map[gender_choice]
user_input["Age"] = st.slider("Возраст (Age)", min_value=10, max_value=80, value=30)
user_input["Height"] = st.slider("Рост (Height)", min_value=1.2, max_value=2.2, value=1.7)
binary_map = {"Да": 1, "Нет": 0}
user_input["family_history"] = binary_map[st.radio("Есть ли у семьи история ожирения? (family_history)", list(binary_map.keys()))]

user_input["FAVC"] = binary_map[st.radio("Часто ли едите высококалорийную пищу? (FAVC)", list(binary_map.keys()))]
user_input["FCVC"] = st.slider("Как часто вы едите овощи? (FCVC)", 0.0, 3.0, 1.5)
user_input["NCP"] = st.slider("Сколько основных приемов пищи у вас в день? (NCP)", 1.0, 4.0, 3.0)
caec_options = {"Никогда": 0, "Редко": 1, "Иногда": 2, "Часто": 3}

user_input["CAEC"] = caec_options[st.selectbox("Едите ли вы между приемами пищи? (CAEC)", list(caec_options.keys()))]
user_input["CH2O"] = st.slider("Сколько воды пьете ежедневно? (литры) (CH2O)", 0.0, 3.0, 1.5)
user_input["SCC"] = binary_map[st.radio("Контролируете ли калории? (SCC)", list(binary_map.keys()))]
user_input["FAF"] = st.slider("Как часто занимаетесь физической активностью? (FAF)", 0.0, 3.0, 1.0)
user_input["TUE"] = st.slider("Сколько времени проводите за гаджетами (часы)? (TUE)", 0.0, 2.0, 1.0)

calc_options = {"Никогда": 0, "Редко": 1, "Иногда": 2, "Часто": 3}
user_input["CALC"] = calc_options[st.selectbox("Как часто употребляете алкоголь? (CALC)", list(calc_options.keys()))]

mtrans_options = {"Авто": 0, "Мотоцикл": 1, "Байк": 2, "Пешком": 3, "Общественный транспорт": 4}
user_input["MTRANS"] = mtrans_options[st.selectbox("Какой транспорт используете чаще всего? (MTRANS)", list(mtrans_options.keys()))]

if st.button("Предсказать"):
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    prediction = knn.predict(user_scaled)[0]

    st.success(f"**Предсказанный класс ожирения: {class_labels[prediction]}**")
