import streamlit as st
import requests
import json

st.set_page_config(
    page_title="SLM Agent Chat",
    page_icon="💬",
    layout="centered"
)

st.title("💬 SLM Agent Chat")
st.markdown("""
This is a free, local chatbot using an ONNX-optimized language model.
All processing happens on your machine - no data is sent to external services.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI response from FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/generate",
                    json={"text": prompt},
                    timeout=30
                )
                response_json = response.json()
                
                if response.status_code == 200:
                    if "generated_text" in response_json:
                        ai_response = response_json["generated_text"]
                        st.markdown(ai_response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": ai_response}
                        )
                    else:
                        st.error("Unexpected response format from server")
                else:
                    error_message = response_json.get("error", f"Error: Status code {response.status_code}")
                    st.error(f"⚠️ {error_message}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to the backend server. Make sure to run 'uvicorn app:app --reload' first!")
            except Exception as e:
                st.error(f"⚠️ An error occurred: {str(e)}")

# Sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    ### How to use
    1. Make sure the backend is running:
       ```
       uvicorn app:app --reload
       ```
    2. Type your message and press Enter
    3. Wait for the AI to respond

    ### Model Information
    - Using local ONNX-optimized model
    - Fast inference with ONNX Runtime
    - 100% free and private
    """)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
