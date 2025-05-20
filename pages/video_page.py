import hashlib
import os
import tempfile

import openai
import torch
import streamlit as st
from dotenv import load_dotenv

from menu import menu_with_redirect
from src.QA import QA
from src.audio_processor import AudioProcessor
from src.retrieval import Retrieval
from src.stt_model import STTModel
from src.video_processor import VideoProcessor

menu_with_redirect()

st.markdown("""
### Помощник по видео

Загрузите видео лекцию или любой файл в формате `.mp4`.

Система автоматически создаст текстовую расшифровку. После этого вы сможете задавать сколько угодно вопросов по содержанию видео.

В ответ вы получите до **трёх временных промежутков**, где, скорее всего, встречается нужная информация. Первый вариант — самый точный, следующие — менее уверенные.

Возможна погрешность до **двух минут** относительно точного момента в видео.

Длительность обработки видео **около 5 минут**
""")


load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_ENDPOINT")
os.getenv("LANGSMITH_API_KEY")
os.getenv("LANGSMITH_PROJECT")

video_session_state_dict = {"uploaded_video": None, "user_query": "", "retrieval": None, "mapping": None, 
                                                          "temp_dir_path": None, "temp_video_path": None,
                                                          "transcription": None, "video_hash": None, "video_processed": False}

for key, value in video_session_state_dict.items():
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

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

@st.cache_resource(show_spinner="Подгрузка модели")
def load_model(device):
    return STTModel("./ctc_model_config.yaml", "./ctc_model_weights.ckpt", device=device)

def load_pipeline_main(path):
    program = QA()
    program.load(path)
    return program

def create_temp_directory():
    return tempfile.mkdtemp()

def create_retriever_main():
    return Retrieval(k=15)

def create_processors(_video_path: str, _model: STTModel, _directory: str):
    return VideoProcessor(video_path=_video_path, directory=_directory), AudioProcessor(model=_model, directory=_directory)

pipeline = load_pipeline_main("./data/optimized_program/mipro_optimized_v6_paraphrase_acc@3.json")

model = load_model(device)

st.header("Загрузите видеофайл лекции")

uploaded_video = st.file_uploader("Выберите видеофайл", type=["mp4"])

if uploaded_video:
    st.session_state.uploaded_video = uploaded_video

if st.session_state.uploaded_video:
    content = st.session_state.uploaded_video.read()
    video_hash = hashlib.md5(content).hexdigest()

    if not content.strip():
        st.error("Файл пустой или повреждён. Пожалуйста, загрузите корректный MP4.")
        st.stop()
    
    st.session_state.mapping = None
    if st.session_state.video_hash != video_hash:
        st.session_state.video_hash = video_hash
        st.session_state.transcription = None
        st.session_state.retrieval = None
        st.session_state.video_processed = False
        st.session_state.user_query = ""
        st.session_state.temp_video_path = None
        st.session_state.temp_dir_path = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(content)
        st.session_state.temp_video_path = temp_file.name


    if not st.session_state.video_processed:
        try:
            st.session_state["temp_dir_path"] = create_temp_directory()
            video_processor, audio_processor = create_processors(st.session_state.temp_video_path, model, st.session_state.temp_dir_path)
            video_processor.process_video()

            transcription = audio_processor.transcribe_files()
            st.session_state.transcription = transcription

            st.session_state.retrieval = create_retriever_main()

            texts = [item["text"] for item in st.session_state.transcription]
            metadatas = [{"time": item["time"]} for item in  st.session_state.transcription]
            st.session_state["retrieval"].add_texts(texts=texts, metadata=metadatas)

            st.session_state.video_processed = True

        except Exception as e:
            st.error(f"Произошла ошибка при обработке видео: {e}")
            st.stop()

    if st.session_state.transcription and st.session_state.video_processed:
        user_query = st.text_input("Введите интересующий вас запрос по видео:", key="user_query")

        if user_query:

            try:
                mapping = pipeline.forward(user_query, retrieval=st.session_state["retrieval"])
                st.session_state["mapping"] = mapping

            except (IndexError, ValueError):
                st.error("Произошла ошибка при ранжировании. Попробуйте еще раз.")
                st.session_state["mapping"] = None

            if st.session_state.mapping:
                st.subheader("Найденные временные промежутки:")
                for i, time_str in enumerate(st.session_state["mapping"], 1):
                    st.markdown(f"**{i}.** {time_str}")