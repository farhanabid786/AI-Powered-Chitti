# import streamlit as st
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from htmlTemplates import css , bot_template , user_template
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms.base import LLM
# from langchain.schema import LLMResult
# from typing import Optional, List
# from pydantic import BaseModel
# from PIL import Image, ImageDraw, ImageFont
# import requests
# import io

# class NvidiaLlamaMaverickLLM(LLM, BaseModel):
#     api_key: str

#     class Config:
#         arbitrary_types_allowed = True

#     @property
#     def _llm_type(self) -> str:
#         return "nvidia_llama_maverick"

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Accept": "application/json"
#         }

#         payload = {
#             "model": "meta/llama-4-maverick-17b-128e-instruct",
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": 512,
#             "temperature": 1.0,
#             "top_p": 1.0,
#             "frequency_penalty": 0.0,
#             "presence_penalty": 0.0,
#             "stream": False
#         }

#         response = requests.post(
#             "https://integrate.api.nvidia.com/v1/chat/completions",
#             headers=headers,
#             json=payload
#         )
#         response.raise_for_status()
#         result = response.json()
#         return result['choices'][0]['message']['content']

#     def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
#         generations = []
#         for prompt in prompts:
#             text = self._call(prompt, stop=stop)
#             generations.append([{"text": text}])
#         return LLMResult(generations=generations)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vectorstore(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# def get_conversation_chain(vectorstore):
#     llm = NvidiaLlamaMaverickLLM(api_key=os.getenv("NVIDIA_API_KEY"))
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# def create_question_image(question_text):
#     width, height = 600, 150
#     background_color = (240, 240, 240)
#     text_color = (0, 0, 0)
#     font = ImageFont.load_default()

#     image = Image.new("RGB", (width, height), background_color)
#     draw = ImageDraw.Draw(image)
#     draw.text((10, 10), question_text, fill=text_color, font=font)

#     buffer = io.BytesIO()
#     image.save(buffer, format="PNG")
#     buffer.seek(0)
#     return buffer

# def handle_userinput(user_question):
#     if st.session_state.conversation is not None:
#         response = st.session_state.conversation({'question': user_question})

#         # Display user question as image
#         img_buffer = create_question_image(user_question)
#         st.image(img_buffer, caption="User Question", use_column_width=True)

#         # Display bot response
#         st.write(bot_template.replace("{{MSG}}", response['answer']), unsafe_allow_html=True)
#     else:
#         st.warning("Please upload and process your PDFs first.")

# def main():
#     load_dotenv()
#     st.set_page_config(page_title="AI Powered Multiple PDFs Chatbot", page_icon=":robot:")
#     st.write(css, unsafe_allow_html=True)
    
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None

#     st.header("Chat with multiple PDFs :robot:")
#     user_question = st.text_input("Ask me a question related to your documents :")

#     if user_question:
#         if st.session_state.conversation is not None:
#             handle_userinput(user_question)
#         else:
#             st.warning("Upload and process your PDFs before asking questions.")

#     st.write(user_template.replace("{{MSG}}", "Hello Chiiti 👋"), unsafe_allow_html=True)
#     st.write(bot_template.replace("{{MSG}}", "Hello Farhan 😀"), unsafe_allow_html=True)

#     with st.sidebar:
#         st.subheader("Your Documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here and click on Process", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 vectorstore = get_vectorstore(text_chunks)
#                 st.session_state.conversation = get_conversation_chain(vectorstore)

# if __name__ == '__main__':
#     main()
# import streamlit as st
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms.base import LLM
# from langchain.schema import LLMResult
# from typing import Optional, List
# from pydantic import BaseModel
# import requests

# from htmlTemplates import css, bot_template, user_template  # Your HTML templates

# class NvidiaLlamaMaverickLLM(LLM, BaseModel):
#     api_key: str

#     class Config:
#         arbitrary_types_allowed = True

#     @property
#     def _llm_type(self) -> str:
#         return "nvidia_llama_maverick"

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Accept": "application/json"
#         }

#         payload = {
#             "model": "meta/llama-4-maverick-17b-128e-instruct",
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": 512,
#             "temperature": 1.0,
#             "top_p": 1.0,
#             "frequency_penalty": 0.0,
#             "presence_penalty": 0.0,
#             "stream": False
#         }

#         response = requests.post(
#             "https://integrate.api.nvidia.com/v1/chat/completions",
#             headers=headers,
#             json=payload
#         )
#         response.raise_for_status()
#         result = response.json()
#         return result['choices'][0]['message']['content']

#     def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
#         generations = []
#         for prompt in prompts:
#             text = self._call(prompt, stop=stop)
#             generations.append([{"text": text}])
#         return LLMResult(generations=generations)


# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     return text_splitter.split_text(text)


# def get_vectorstore(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# def get_conversation_chain(vectorstore):
#     llm = NvidiaLlamaMaverickLLM(api_key=os.getenv("NVIDIA_API_KEY"))
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )


# def handle_userinput(user_question):
#     if st.session_state.conversation is not None:
#         response = st.session_state.conversation({'question': user_question})
#         st.session_state.chat_history.append({
#             "user": user_question,
#             "bot": response['answer']
#         })


# def render_chat_history():
#     # First static greeting at the top
#     st.write(user_template.replace("{{MSG}}", "Hello Chiiti 👋"), unsafe_allow_html=True)
#     st.write(bot_template.replace("{{MSG}}", "Hello Farhan 😀"), unsafe_allow_html=True)

#     # Then dynamic chat history
#     for chat in st.session_state.chat_history:
#         st.write(user_template.replace("{{MSG}}", chat['user']), unsafe_allow_html=True)
#         st.write(bot_template.replace("{{MSG}}", chat['bot']), unsafe_allow_html=True)


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="AI Powered Multiple PDFs Chatbot", page_icon=":robot:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     st.header("Chat with multiple PDFs :robot:")
#     user_question = st.text_input("Ask me a question related to your documents :")

#     if user_question:
#         if st.session_state.conversation is not None:
#             handle_userinput(user_question)
#         else:
#             st.warning("Upload and process your PDFs before asking questions.")

#     render_chat_history()

#     with st.sidebar:
#         st.subheader("Your Documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on Process", accept_multiple_files=True
#         )
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 vectorstore = get_vectorstore(text_chunks)
#                 st.session_state.conversation = get_conversation_chain(vectorstore)


# if __name__ == '__main__':
#     main()

import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.schema import LLMResult
from typing import Optional, List
from pydantic import BaseModel
import requests

from htmlTemplates import css, bot_template, user_template

class NvidiaLlamaMaverickLLM(LLM, BaseModel):
    api_key: str

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "nvidia_llama_maverick"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        payload = {
            "model": "meta/llama-4-maverick-17b-128e-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 1.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False
        }
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([{"text": text}])
        return LLMResult(generations=generations)

# ---------------- PDF Processing ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = NvidiaLlamaMaverickLLM(api_key=os.getenv("NVIDIA_API_KEY"))
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# ---------------- Chat Handlers ----------------
def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history.append({
            "user": user_question,
            "bot": response['answer']
        })

def render_chat_history():
    # Scrollable chat container
    st.markdown('<div style="max-height: 500px; overflow-y: auto; padding:10px;">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        st.markdown(user_template.replace("{{MSG}}", chat['user']), unsafe_allow_html=True)
        st.markdown(bot_template.replace("{{MSG}}", chat['bot']), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Main App ----------------
def main():
    load_dotenv()
    st.set_page_config(page_title="AI Powered Chitti", page_icon=":robot:", layout="wide")
    st.markdown(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Layout: Sidebar + Main
    with st.sidebar:
        st.subheader("Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload at least one PDF.")

    # Main chat interface
    st.title("📄Multiple PDFs AI Powered Chitti 🤖")
    st.subheader("Ask questions about your uploaded documents")
    user_question = st.text_input("Type your question here:")

    if user_question and st.session_state.conversation is not None:
        handle_userinput(user_question)

    if not st.session_state.chat_history:
        st.markdown(bot_template.replace("{{MSG}}", "Hello! Upload PDFs and ask me anything."), unsafe_allow_html=True)
    else:
        render_chat_history()

if __name__ == '__main__':
    main()


#  ### IMPORTANT
# import streamlit as st
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms.base import LLM
# from langchain.schema import LLMResult
# from typing import Optional, List
# from pydantic import BaseModel
# import requests

# from htmlTemplates import css, bot_template, user_template

# class NvidiaLlamaMaverickLLM(LLM, BaseModel):
#     api_key: str

#     class Config:
#         arbitrary_types_allowed = True

#     @property
#     def _llm_type(self) -> str:
#         return "nvidia_llama_maverick"

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
#         payload = {
#             "model": "meta/llama-4-maverick-17b-128e-instruct",
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": 512,
#             "temperature": 1.0,
#             "top_p": 1.0,
#             "frequency_penalty": 0.0,
#             "presence_penalty": 0.0,
#             "stream": False
#         }
#         response = requests.post(
#             "https://integrate.api.nvidia.com/v1/chat/completions",
#             headers=headers,
#             json=payload
#         )
#         response.raise_for_status()
#         result = response.json()
#         return result['choices'][0]['message']['content']

#     def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
#         generations = []
#         for prompt in prompts:
#             text = self._call(prompt, stop=stop)
#             generations.append([{"text": text}])
#         return LLMResult(generations=generations)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() + "\n"
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     return text_splitter.split_text(text)

# def get_vectorstore(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# def get_conversation_chain(vectorstore):
#     llm = NvidiaLlamaMaverickLLM(api_key=os.getenv("NVIDIA_API_KEY"))
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )

# def handle_userinput(user_question):
#     if st.session_state.conversation is not None:
#         result = st.session_state.conversation({'question': user_question})
#         response = result['answer']
#         st.session_state.chat_history.append({
#             "user": user_question,
#             "bot": response
#         })

# def render_chat_history():
#     st.markdown('<div style="max-height: 500px; overflow-y: auto; padding:10px;">', unsafe_allow_html=True)
#     for chat in st.session_state.chat_history:
#         st.markdown(user_template.replace("{{MSG}}", chat['user']), unsafe_allow_html=True)
#         st.markdown(bot_template.replace("{{MSG}}", chat['bot']), unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# def main():
#     load_dotenv()
#     st.set_page_config(page_title="AI Powered Chitti", page_icon=":robot:", layout="wide")
#     st.markdown(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     with st.sidebar:
#         st.subheader("Upload Your PDFs")
#         pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
#         if st.button("Process PDFs"):
#             if pdf_docs:
#                 with st.spinner("Processing PDFs..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     vectorstore = get_vectorstore(text_chunks)
#                     st.session_state.conversation = get_conversation_chain(vectorstore)
#                     st.success("PDFs processed successfully!")
#             else:
#                 st.warning("Please upload at least one PDF.")

#     st.title("📄 Multiple PDFs AI Powered Chitti 🤖")
#     st.subheader("Ask questions about your uploaded documents")

#     user_question = st.text_input("Type your question here:")
#     if user_question and st.session_state.conversation is not None:
#         handle_userinput(user_question)

#     if not st.session_state.chat_history:
#         st.markdown(bot_template.replace("{{MSG}}", "Hello! Upload PDFs and ask me anything."), unsafe_allow_html=True)
#     else:
#         render_chat_history()

# if __name__ == '__main__':
#     main()

# import streamlit as st
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import HuggingFaceHub
# from htmlTemplates import css, bot_template, user_template
# from datetime import datetime


# # Load environment variables
# load_dotenv()


# # --- PDF Processing ---
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#     return text


# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     return splitter.split_text(text)


# # def get_vectorstore(text_chunks):
# #     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# #     return FAISS.from_texts(text_chunks, embedding=embeddings)
# def get_vectorstore(text_chunks):
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"}   # 👈 Force CPU
#     )
#     return FAISS.from_texts(text_chunks, embedding=embeddings)



# def get_conversation_chain(vectorstore):
#     llm = HuggingFaceHub(
#         repo_id="google/flan-t5-base",
#         model_kwargs={"temperature": 0.5, "max_length": 512}
#     )
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )


# # --- Chat Handling ---
# def handle_userinput(user_question):
#     if st.session_state.conversation is None:
#         st.warning("⚠️ Please upload and process PDFs first.")
#         return

#     response = st.session_state.conversation({"question": user_question})
#     chat_history = response["chat_history"]

#     # store formatted history with timestamp
#     st.session_state.chat_history = []
#     for i, message in enumerate(chat_history):
#         timestamp = datetime.now().strftime("%H:%M:%S")
#         if i % 2 == 0:
#             st.session_state.chat_history.append(
#                 {"role": "user", "content": message.content, "time": timestamp}
#             )
#         else:
#             st.session_state.chat_history.append(
#                 {"role": "bot", "content": message.content, "time": timestamp}
#             )


# def render_chat_history():
#     st.markdown(
#         '<div style="max-height: 500px; overflow-y: auto; padding:10px; background:#1e1e2f; border-radius:10px;">',
#         unsafe_allow_html=True
#     )
#     for chat in st.session_state.chat_history:
#         template = user_template if chat["role"] == "user" else bot_template
#         rendered = template.replace("{{MSG}}", chat["content"]).replace("{{TIME}}", chat["time"])
#         st.write(rendered, unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)


# # --- Main App ---
# def main():
#     st.set_page_config(page_title="AI Powered Chitti", page_icon="🤖", layout="wide")
#     st.markdown(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     # Sidebar for PDFs
#     with st.sidebar:
#         st.markdown(
#             "<div style='padding:15px; background:#2b313e; border-radius:10px; color:white;'>"
#             "<h3 style='text-align:center;'>📂 Upload Your PDFs</h3>",
#             unsafe_allow_html=True
#         )
#         pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
#         if st.button("Process PDFs"):
#             if pdf_docs:
#                 with st.spinner("Processing PDFs..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     vectorstore = get_vectorstore(text_chunks)
#                     st.session_state.conversation = get_conversation_chain(vectorstore)
#                     st.success("✅ PDFs processed successfully! Ask me anything.")
#             else:
#                 st.warning("⚠️ Please upload at least one PDF.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     # Header
#     st.markdown("<h1 style='color:white; text-align:center;'>📄 Multiple PDFs AI Powered Chitti 🤖</h1>", unsafe_allow_html=True)
#     st.subheader("💬 Ask questions about your uploaded documents")

#     # Chat input
#     col1, col2 = st.columns([8, 1])
#     with col1:
#         user_question = st.text_input("Type your question here:", key="user_input")
#     with col2:
#         send = st.button("🚀 Send")

#     if (user_question or send) and st.session_state.conversation is not None:
#         handle_userinput(user_question)
#         st.session_state.user_input = ""

#     # Render chat
#     if not st.session_state.chat_history:
#         st.markdown(bot_template.replace("{{MSG}}", "👋 Hello! Upload PDFs and ask me anything.").replace("{{TIME}}", datetime.now().strftime("%H:%M:%S")), unsafe_allow_html=True)
#     else:
#         render_chat_history()


# if __name__ == "__main__":
#     main()
# import streamlit as st
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms.base import LLM
# from langchain.schema import LLMResult
# from typing import Optional, List
# from pydantic import BaseModel
# import requests

# from htmlTemplates import css, bot_template, user_template

# # -------------------------------
# # Set Hugging Face token from Streamlit secrets
# # -------------------------------
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACE_API_TOKEN"]

# # -------------------------------
# # Cache embeddings to avoid repeated downloads
# # -------------------------------
# @st.cache_resource
# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # -------------------------------
# # Custom LLM for Nvidia Llama
# # -------------------------------
# class NvidiaLlamaMaverickLLM(LLM, BaseModel):
#     api_key: str

#     class Config:
#         arbitrary_types_allowed = True

#     @property
#     def _llm_type(self) -> str:
#         return "nvidia_llama_maverick"

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
#         payload = {
#             "model": "meta/llama-4-maverick-17b-128e-instruct",
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": 512,
#             "temperature": 1.0,
#             "top_p": 1.0,
#             "frequency_penalty": 0.0,
#             "presence_penalty": 0.0,
#             "stream": False
#         }
#         response = requests.post(
#             "https://integrate.api.nvidia.com/v1/chat/completions",
#             headers=headers,
#             json=payload
#         )
#         response.raise_for_status()
#         result = response.json()
#         return result['choices'][0]['message']['content']

#     def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
#         generations = []
#         for prompt in prompts:
#             text = self._call(prompt, stop=stop)
#             generations.append([{"text": text}])
#         return LLMResult(generations=generations)

# # -------------------------------
# # PDF processing
# # -------------------------------
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() + "\n"
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     return text_splitter.split_text(text)

# # -------------------------------
# # Vectorstore using cached embeddings
# # -------------------------------
# def get_vectorstore(text_chunks):
#     embeddings = get_embeddings()  # cached embeddings
#     return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# # -------------------------------
# # Conversation chain
# # -------------------------------
# def get_conversation_chain(vectorstore):
#     llm = NvidiaLlamaMaverickLLM(api_key=os.getenv("NVIDIA_API_KEY"))
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )

# # -------------------------------
# # Chat handlers
# # -------------------------------
# def handle_userinput(user_question):
#     if st.session_state.conversation is not None:
#         result = st.session_state.conversation({'question': user_question})
#         response = result['answer']
#         st.session_state.chat_history.append({
#             "user": user_question,
#             "bot": response
#         })

# def render_chat_history():
#     st.markdown('<div style="max-height: 500px; overflow-y: auto; padding:10px;">', unsafe_allow_html=True)
#     for chat in st.session_state.chat_history:
#         st.markdown(user_template.replace("{{MSG}}", chat['user']), unsafe_allow_html=True)
#         st.markdown(bot_template.replace("{{MSG}}", chat['bot']), unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# # -------------------------------
# # Main Streamlit app
# # -------------------------------
# def main():
#     load_dotenv()
#     st.set_page_config(page_title="AI Powered Chitti", page_icon=":robot:", layout="wide")
#     st.markdown(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     with st.sidebar:
#         st.subheader("Upload Your PDFs")
#         pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
#         if st.button("Process PDFs"):
#             if pdf_docs:
#                 with st.spinner("Processing PDFs..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     vectorstore = get_vectorstore(text_chunks)
#                     st.session_state.conversation = get_conversation_chain(vectorstore)
#                     st.success("PDFs processed successfully!")
#             else:
#                 st.warning("Please upload at least one PDF.")

#     st.title("📄 Multiple PDFs AI Powered Chitti 🤖")
#     st.subheader("Ask questions about your uploaded documents")

#     user_question = st.text_input("Type your question here:")
#     if user_question and st.session_state.conversation is not None:
#         handle_userinput(user_question)

#     if not st.session_state.chat_history:
#         st.markdown(bot_template.replace("{{MSG}}", "Hello! Upload PDFs and ask me anything."), unsafe_allow_html=True)
#     else:
#         render_chat_history()

# if __name__ == '__main__':
#     main()





