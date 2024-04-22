import streamlit as st
from model import genai_engine

import streamlit as st
from model import genai_engine

def main():
    st.title("Emoji Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.text_area(f"You {idx+1}:", value=message["content"], disabled=True, key=f"user_{idx}")
        else:
            st.text_area(f"Chatbot {idx+1}:", value=message["content"], disabled=True, key=f"assistant_{idx}")

    prompt = st.text_input("You:", key="input_text")
    if st.button("Send"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = genai_engine(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
