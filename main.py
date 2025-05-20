import streamlit as st
from menu import menu
import dspy

if "dspy_initialized" not in st.session_state:
    lm = dspy.LM("openai/gpt-4o")
    dspy.configure(lm=lm)
    st.session_state["dspy_initialized"] = True

options = ["Стартовая страница", "Запросы по видео", "Запросы по файлу с транскрипцией"]

if "role" not in st.session_state:
    st.session_state.role = options[0]

current_value = st.session_state.role
index = options.index(current_value) if current_value in options else 0

def set_role():
    selected = st.session_state._role
    if selected == options[0]:
        st.session_state.role = options[0]
    else:
        st.session_state.role = selected

st.selectbox(
    "Выберите страницу:",
    options,
    key="_role",
    index=index,
    on_change=set_role,
)

if st.session_state.role == options[0]:
    st.markdown("---")
    st.markdown("### Добро пожаловать!")
    st.markdown("""
    Здесь вы можете выбрать один из двух режимов работы с видеолекциями:

    #### Режим 1: Запросы по видео
    - Загрузите видеолекцию.
    - Подождите, пока она обработается (в зависимости от длины это может занять некоторое время).
    - После завершения обработки вы сможете задавать **неограниченное количество вопросов**.
    - В ответ вы получите **3 временных промежутка**, где встречается нужная информация.

    #### Режим 2: Запросы по файлу с транскрипцией
    - Загрузите **готовую транскрипцию** лекции.
    - Сразу начинайте задавать вопросы.
    - Ответ будет основан на тексте без привязки ко времени.

    Этот режим больше предназначен для **тестирования**, но если есть транскрипция нужного формата, то можно использовать.
    """)
    st.markdown("---")

menu()
