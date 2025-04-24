import streamlit as st

def authenticated_menu():
    st.sidebar.page_link("main.py", label="Switch mode")
    if st.session_state.role == "test":
        st.sidebar.page_link("pages/test_pages.py", label="Testing application")
    if st.session_state.role == "main":
        st.sidebar.page_link("pages/transcription_pages.py", label="Transcripting of the lecture")


def unauthenticated_menu():
    st.sidebar.page_link("main.py", label="Select mode")


def menu():
    if "role" not in st.session_state or st.session_state.role is None:
        unauthenticated_menu()
        return
    authenticated_menu()


def menu_with_redirect():
    if "role" not in st.session_state or st.session_state.role is None:
        st.switch_page("main.py")
    menu()