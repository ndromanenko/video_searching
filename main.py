import streamlit as st
from menu import menu

if "role" not in st.session_state:
    st.session_state.role = None

st.session_state._role = st.session_state.role

def set_role():
    st.session_state.role = st.session_state._role


st.selectbox(
    "Select your role:",
    [None, "main", "test"],
    key="_role",
    on_change=set_role,
)
menu()