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


menu_with_redirect()

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
    return QA()

def create_retriever():
    return Retrieval(k=10)

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

st.header("Upload file for testing")

uploaded_file = st.file_uploader("Choose a json file with transcription", type=["json"])


if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file.write(st.session_state["uploaded_file"].read())
        temp_file_path = temp_file.name

    with open(temp_file_path, "r") as file:
        transcription = json.load(file)

    st.session_state.transcription = transcription
    st.success("Transcription uploaded successfully!")

    texts = [item["text"] for item in st.session_state.transcription]
    metadatas = [{"time": item["time"]} for item in st.session_state.transcription]
    
    st.session_state["retriever"].add_texts(texts=texts, metadata=metadatas)

    if st.session_state.transcription:

        user_query = st.text_input("Enter your query about the video:")

        if user_query:
            st.session_state.user_query = user_query 

        if st.session_state.user_query:

            try:
                answer = pipeline.forward(user_query, st.session_state["retriever"])
                st.subheader("Answer:")
                st.write(answer)
            except (IndexError, ValueError):
                st.error("An error occurred during ranking. Please try again.")