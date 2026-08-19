import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from htmlTemplates import css, bot_template, user_template


MODEL_NAME = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_pdf_text(pdf_docs):
    text_parts = []

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""

            if page_text.strip():
                text_parts.append(page_text)

    return "\n".join(text_parts)


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    return text_splitter.split_text(text)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )


def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError(
            "No readable text was found in the uploaded PDFs."
        )

    embeddings = get_embeddings()

    return FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )


def get_nvidia_client():
    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is missing."
        )

    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key
    )


def ask_nvidia(question, context, chat_history):
    client = get_nvidia_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are Chitti, an AI assistant for answering "
                "questions about uploaded PDF documents. "
                "Use the supplied document context as your primary "
                "source of truth. "
                "Do not invent facts. "
                "If the answer cannot be found in the document "
                "context, clearly say that the information could "
                "not be found in the uploaded documents. "
                "Answer clearly and directly."
            )
        }
    ]

    for message in chat_history[-6:]:
        messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )

    current_prompt = f"""
Document context:

{context}

Current user question:

{question}

Answer the question using the document context.

If the document context does not contain enough information,
say that the information could not be found in the uploaded
documents.
"""

    messages.append(
        {
            "role": "user",
            "content": current_prompt
        }
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        top_p=0.7,
        max_tokens=4096,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True
            },
            "thinking_token_budget": 2048
        },
        stream=False
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError(
            "NVIDIA returned an empty response."
        )

    return answer.strip()


def handle_userinput(user_question):
    vectorstore = st.session_state.vectorstore

    if vectorstore is None:
        st.warning(
            "Please upload and process your PDFs first."
        )
        return

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )

    documents = retriever.invoke(user_question)

    if not documents:
        answer = (
            "I couldn't find relevant information "
            "in the uploaded documents."
        )
    else:
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        previous_history = list(
            st.session_state.chat_history
        )

        answer = ask_nvidia(
            question=user_question,
            context=context,
            chat_history=previous_history
        )

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


def render_chat_history():
    for message in st.session_state.chat_history:

        if message["role"] == "user":
            st.markdown(
                user_template.replace(
                    "{{MSG}}",
                    message["content"]
                ),
                unsafe_allow_html=True
            )

        elif message["role"] == "assistant":
            st.markdown(
                bot_template.replace(
                    "{{MSG}}",
                    message["content"]
                ),
                unsafe_allow_html=True
            )


def main():
    load_dotenv()

    st.set_page_config(
        page_title="AI Powered Chitti",
        page_icon="🤖",
        layout="wide"
    )

    st.markdown(
        css,
        unsafe_allow_html=True
    )

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:

        st.subheader("Upload Your PDFs")

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button(
            "Process PDFs",
            use_container_width=True
        ):

            if not pdf_docs:
                st.warning(
                    "Please upload at least one PDF."
                )

            else:

                try:

                    with st.spinner(
                        "Processing PDFs..."
                    ):

                        raw_text = get_pdf_text(
                            pdf_docs
                        )

                        if not raw_text.strip():
                            raise ValueError(
                                "No readable text was extracted "
                                "from the uploaded PDFs."
                            )

                        text_chunks = get_text_chunks(
                            raw_text
                        )

                        vectorstore = get_vectorstore(
                            text_chunks
                        )

                        st.session_state.vectorstore = (
                            vectorstore
                        )

                        st.session_state.chat_history = []

                    st.success(
                        "PDFs processed successfully!"
                    )

                except Exception as error:

                    st.error(
                        f"PDF processing failed: {error}"
                    )

    st.title(
        "📄 Multiple PDFs AI Powered Chitti 🤖"
    )

    st.subheader(
        "Ask questions about your uploaded documents"
    )

    if st.session_state.vectorstore is None:

        st.markdown(
            bot_template.replace(
                "{{MSG}}",
                "Hello! Upload PDFs and ask me anything."
            ),
            unsafe_allow_html=True
        )

    else:

        render_chat_history()

    user_question = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if user_question:

        with st.spinner(
            "Chitti is thinking..."
        ):

            try:

                handle_userinput(
                    user_question
                )

            except Exception as error:

                st.error(
                    f"Unable to generate an answer: {error}"
                )

        st.rerun()


if __name__ == "__main__":
    main()
