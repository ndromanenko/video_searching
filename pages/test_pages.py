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

CSV_FILE = "data/queries.csv"
dataframe = pd.read_csv(CSV_FILE)


menu_with_redirect()

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_ENDPOINT")
os.getenv("LANGSMITH_API_KEY")
os.getenv("LANGSMITH_PROJECT")

default_session_state_dict = {"uploaded_file": None, "user_query": None, "transcription": None, "answer_ready": False, "mapping": None}

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
    # st.success("Transcription uploaded successfully!")

    texts = [item["text"] for item in st.session_state.transcription]
    metadatas = [{"time": item["time"]} for item in st.session_state.transcription]
    
    st.session_state["retriever"].add_texts(texts=texts, metadata=metadatas)

    if st.session_state.transcription and not st.session_state.answer_ready:

        user_query = st.text_input("Enter your query about the video:")

        if user_query:
            st.session_state.user_query = user_query
            st.session_state.answer_ready = True
            st.rerun() 

    elif st.session_state.transcription and st.session_state.answer_ready:
        st.subheader("Your query:")
        st.write(st.session_state.user_query)

        try:
            mapping = pipeline.forward_test(st.session_state.user_query, st.session_state["retriever"])
            if not st.session_state.mapping:
                st.session_state.mapping = mapping

            correct_answer = st.text_input("Enter the correct answer from those suggested in the table:")
            if correct_answer and st.session_state.mapping and st.button("Add"):
                new_row = {"Query": st.session_state.user_query, "Correct_timestamp": list(st.session_state.mapping.values())[int(correct_answer)]}
                dataframe = pd.concat([dataframe, pd.DataFrame([new_row])], ignore_index=True)
                dataframe.to_csv(CSV_FILE, index=False)
                st.success("Added!")

                st.session_state.user_query = None
                st.session_state.answer_ready = False
                st.session_state.mapping = None
                st.rerun()

        except (IndexError, ValueError):
            st.error("An error occurred during ranking. Please try again.")