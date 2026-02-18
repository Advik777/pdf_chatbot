# app.py

import streamlit as st
from ollama_client import stream_chat
from rag_chroma import add_pdf_to_chroma, retrieve_context, list_files, delete_file

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------
st.set_page_config(
    page_title="Advik's AI Assistant",
    layout="wide"
)

# ---------------------------------
# DARK THEME CSS (Permanent)
# ---------------------------------
st.markdown("""
<style>
    /* App Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    /* Sticky Header */
    .sticky-header {
        position: sticky;
        top: 0;
        background-color: #0E1117;
        padding: 15px;
        z-index: 999;
        border-bottom: 1px solid #2A2A2A;
    }

    /* Chat spacing */
    .block-container {
        padding-top: 1rem;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1F2937;
        color: white;
        border-radius: 8px;
        border: 1px solid #374151;
    }

    .stButton>button:hover {
        background-color: #374151;
        color: white;
    }

    /* File uploader */
    .stFileUploader {
        background-color: #1F2937;
        padding: 10px;
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# STICKY HEADER
# ---------------------------------
st.markdown("""
<div class="sticky-header">
    <h1 style="text-align:center; color:white;">
        🦁Sheru - Advik's AI Assistant
    </h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# SIDEBAR (Upload + File Manager)
# ---------------------------------
with st.sidebar:
    st.header("📂 Document Manager")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing and storing in vector DB..."):
            add_pdf_to_chroma(uploaded_file)
        st.success("PDF stored successfully!")

    st.divider()

    # Stored Files
    st.subheader("Stored Files")

    files = list_files()

    if files:
        for file in files:
            col1, col2 = st.columns([3, 1])

            col1.write(file)

            # Delete Button
            if col2.button("❌", key=file):
                delete_file(file)
                st.success(f"{file} deleted.")
                st.rerun()
    else:
        st.caption("No documents stored yet.")

    st.divider()

    # Clear Chat
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Vector DB: Chroma (Persistent)")
    st.caption("Model: llama3.1:8b (Local via Ollama)")

# ---------------------------------
# CHAT SESSION STATE
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------
# USER INPUT
# ---------------------------------
if prompt := st.chat_input("Ask something..."):

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---------------------------------
    # RETRIEVE CONTEXT FROM CHROMA
    # ---------------------------------
    relevant_chunks = retrieve_context(prompt)
    context = "\n\n".join(relevant_chunks)

    augmented_prompt = f"""
Use the following context to answer the question:

{context}

Question: {prompt}
"""

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": augmented_prompt}
    ]

    # ---------------------------------
    # STREAM RESPONSE FROM OLLAMA
    # ---------------------------------
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in stream_chat(messages):
            full_response += chunk
            response_placeholder.markdown(full_response)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })