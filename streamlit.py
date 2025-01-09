import streamlit as st
import tempfile
from src.utils import *
from src.EncDecoder import *
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore import InMemoryDocstore
import faiss
from src.Chat import *
from src.model_loader import load_ctc_model
import os

os.environ["OPENAI_API_KEY"] = "api_key"

if not torch.backends.mps.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("mps")

try:
    opt_model = load_ctc_model("./ctc_model_config.yaml", "./ctc_model_weights.ckpt", device)
    print("CTC Model loaded successfully!")
except Exception as e:
    print(f"Error loading CTC Model: {e}")

@st.cache_resource
def create_temp_directory():
    temp_dir = tempfile.mkdtemp()
    return temp_dir

if "temp_dir_path" not in st.session_state:
    st.session_state["temp_dir_path"] = create_temp_directory()

temp_dir_path = st.session_state["temp_dir_path"]

@st.cache_resource
def create_retriever():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    retriever = FAISS(
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
        embedding_function=embeddings.embed_query
    )
    chat = ChatBot()
    return retriever, chat

retriever, chat = create_retriever()

st.title("Assistant for Video Lections")
st.header("Upload your video")

if "uploaded_video" not in st.session_state:
    st.session_state["uploaded_video"] = None

uploaded_video = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_video:
    st.session_state["uploaded_video"] = uploaded_video

if st.session_state["uploaded_video"]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(st.session_state["uploaded_video"].read())
        temp_video_path = temp_file.name

    print(f"Video path: {temp_video_path}")

    if st.button("Process Video"):
        process_video(temp_video_path, temp_dir_path)
        st.session_state["files"] = os.listdir(temp_dir_path)

    st.success("Chunks processed successfully!")

    print(f"Processed files: {st.session_state.get('files', [])}")

    transcribation = transcribe(opt_model, temp_dir_path)
    st.session_state["transcription"] = transcribation
    st.success("Video trascribation successfully!")

    add_data_with_metadata(data=st.session_state["transcription"], retriever=retriever)

    user_query = st.text_input("Enter your query about the video:")

    if user_query:

        for_ranking, mapping, context = search(user_query, retriever, 10)
        ranking_response = chat.ranker_function(user_query, context)

        try:
            answer_index = int(ranking_response.split('\n')[0].split('. ')[1]) - 1
            answer = mapping[for_ranking[answer_index]]
            st.subheader("Answer:")
            st.write(answer)
        except (IndexError, ValueError):
            st.error("An error occurred during ranking. Please try again.")
