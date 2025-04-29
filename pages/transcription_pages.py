import openai
import os
import streamlit as st
import torch 
from src.stt_model import STTModel
from src.QA import QA
import tempfile
from src.retrieval import Retrieval
from src.audio_processor import AudioProcessor
from src.video_processor import VideoProcessor
from dotenv import load_dotenv
import json
import dspy
from menu import menu_with_redirect


menu_with_redirect()

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

default_session_state_dict = {"uploaded_video": None, "user_query": None, "transcription": None}

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

if not torch.backends.mps.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("mps")

@st.cache_resource
def load_model():
    # return STTModel(model="ctc", fp16_encoder=True, device="cpu")
    return STTModel("./ctc_model_config.yaml", "./ctc_model_weights.ckpt", device).load_stt_model()

@st.cache_resource
def load_pipeline():
    lm = dspy.LM("openai/gpt-4o")
    dspy.configure(lm=lm)
    return QA()

@st.cache_resource
def create_temp_directory():
    return tempfile.mkdtemp()

if "temp_dir_path" not in st.session_state:
    st.session_state["temp_dir_path"] = create_temp_directory()

temp_dir_path = st.session_state["temp_dir_path"]

def create_retriever():
    return Retrieval(k=10)

@st.cache_resource
def create_processors(_video_path: str, _model: STTModel, _directory: str):
    return VideoProcessor(video_path=_video_path, directory=_directory), AudioProcessor(model=_model, directory=_directory)

if "retriever" not in st.session_state:
    st.session_state["retriever"] = create_retriever()

retriever = st.session_state["retriever"]
model = load_model()
pipeline = load_pipeline()

st.title("Assistant for Video Lections")

# st.markdown("[Перейти на страницу для теста](pages/test_page.py)")

st.header("Upload your video")

uploaded_video = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_video:
    st.session_state.uploaded_video = uploaded_video

if st.session_state.uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(st.session_state["uploaded_video"].read())
        temp_video_path = temp_file.name

    video_processor, audio_processor = create_processors(temp_video_path, model, temp_dir_path)

    if st.button("Process Video"):
        video_processor.process_video()
        st.session_state.files = os.listdir(temp_dir_path)
        st.success("Chunks processed successfully!")

        if st.session_state.transcription is None:
            transcription = audio_processor.transcribe_files()
            st.session_state.transcription = transcription
            st.success("Video trascribation successfully!")

            # print(transcription)
            
            texts = [item["text"] for item in st.session_state.transcription]
            metadatas = [{"time": item["time"]} for item in st.session_state.transcription]

            # print(texts)
            # print(metadatas)

            st.session_state["retriever"].add_texts(texts=texts, metadata=metadatas)

        # print(f"Video path: {temp_video_path}")
        

        # print(f"Processed files: {st.session_state.get('files', [])}")


    if st.session_state.transcription:

        # ДЛЯ ТЕСТА, ЕСЛИ НОВАЯ ЛЕКЦИЯ, ТРАНСКРИПЦИИ КОТОРОЙ У МЕНЯ НЕТ
        json_data = json.dumps(st.session_state.transcription, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download JSON file",
            data=json_data,
            file_name="transcription.json",
            mime="application/json"
        )

        user_query = st.text_input("Enter your query about the video:")

        if user_query:
            st.session_state.user_query = user_query 

        if st.session_state.user_query:

            # for_ranking, mapping, context = search(user_query, retriever, 10)
            # ranking_response = chat.ranker_function(user_query, context)
            context = retriever.search(user_query)

            try:
                # answer_index = int(ranking_response.split('\n')[0].split('. ')[1]) - 1
                # answer = mapping[for_ranking[answer_index]]
                st.subheader("Answer:")
                st.write(context)
            except (IndexError, ValueError):
                st.error("An error occurred during ranking. Please try again.")