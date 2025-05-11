import streamlit as st

def authenticated_menu():
    st.sidebar.page_link("main.py", label="Сменить страницу")
    if st.session_state.role == "Запросы по файлу с транскрипцией":
        st.sidebar.page_link("pages/test_pages.py", label="Запрос по транскрипции")
    if st.session_state.role == "Запросы по видео":
        st.sidebar.page_link("pages/transcription_pages.py", label="Запросы по видео-лекции")


def unauthenticated_menu():
    st.sidebar.page_link("main.py", label="Выбрать страницу")


def menu():
    if "role" not in st.session_state or st.session_state.role is None:
        unauthenticated_menu()
        return
    authenticated_menu()


def menu_with_redirect():
    if "role" not in st.session_state or st.session_state.role is None:
        st.switch_page("main.py")
    menu()