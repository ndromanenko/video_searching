import streamlit as st
from src.QA import QA
from src.retrieval import Retrieval
import json
import dspy
from dotenv import load_dotenv
import openai
import os
import tempfile
from menu import menu_with_redirect
import pandas as pd

menu_with_redirect()

st.markdown("""
###  Анализ видео-лекции по транскрипции

На этой странице вы можете загрузить `.json` файл с транскрипцией видео. Файл должен состоять из пар `"time"` и `"text"`, где:

- `time` — временной промежуток в формате `"секунды_начала:секунды_конца"` (например, `4145:4205`);
- `text` — фрагмент речи, соответствующий указанному промежутку времени.

После загрузки файла вы можете задавать любое количество вопросов по содержанию лекции. В ответ вы получите **три временных промежутка**, в которых с высокой вероятностью встречается информация, связанная с вашим запросом.

- Первый интервал — наиболее релевантный;
- Второй и третий — возможные дополнения, с чуть меньшей точностью;
- Возможна погрешность до **2 минут** относительно точного упоминания.

Этот режим особенно удобен для **тестирования** или работы с уже расшифрованными видео.
""")

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_ENDPOINT")
os.getenv("LANGSMITH_API_KEY")
os.getenv("LANGSMITH_PROJECT")

default_session_state_dict = {"uploaded_file": None, "user_query": None, "transcription": None}

for key, value in default_session_state_dict.items():
    if key not in st.session_state:
        st.session_state[key] = value

button_style = """
        <style>
        .stButton > button {
            color: black;
            background: white;
            width: 200px;
            height: 50px;
        }
        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    lm = dspy.LM("openai/gpt-4o")
    dspy.configure(lm=lm) 
    program = QA()
    program.load("/Users/nikitaromanenko/video_course/video_searching/data/optimized_program/mipro_optimized_v6_paraphrase_acc@3.json")
    return program

def create_retriever():
    return Retrieval(k=15)

@st.cache_resource
def create_temp_directory():
    return tempfile.mkdtemp()

if "temp_dir_path" not in st.session_state:
    st.session_state["temp_dir_path"] = create_temp_directory()

temp_dir_path = st.session_state["temp_dir_path"]

if "retriever" not in st.session_state:
    st.session_state["retriever"] = create_retriever()

retriever = st.session_state["retriever"]
pipeline = load_pipeline()

st.header("Загрузите файл для работы")

uploaded_file = st.file_uploader("Выберите файл с транскрипцией", type=["json"])


if uploaded_file:
    content = uploaded_file.read()

    if not content.strip():
        st.error("Файл пустой или повреждён. Пожалуйста, загрузите корректный JSON.")
        st.stop()

    st.session_state.uploaded_file = uploaded_file

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        with open(temp_file_path, "r") as file:
            transcription = json.load(file)
        st.session_state.transcription = transcription

    except json.JSONDecodeError:
        st.error("Ошибка при чтении JSON. Проверьте корректность структуры файла.")
        st.session_state.uploaded_file = None
        st.stop()

    texts = [item["text"] for item in st.session_state.transcription]
    metadatas = [{"time": item["time"]} for item in st.session_state.transcription]
    
    st.session_state["retriever"].add_texts(texts=texts, metadata=metadatas)

    if st.session_state.transcription:

        user_query = st.text_input("Введите интересующий вас запрос:")

        if user_query and user_query != st.session_state.get("user_query", ""):
            st.session_state.user_query = user_query

            try:
                mapping = pipeline.forward(user_query, retrieval=st.session_state["retriever"])

                st.subheader("Найденные временные промежутки:")

                for i, time_str in enumerate(mapping, 1):
                    st.markdown(f"**{i}.** {time_str}")

            except (IndexError, ValueError):
                st.error("Произошла ошибка при ранжировании. Попробуйте еще раз.")