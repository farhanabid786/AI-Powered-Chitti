# 📚 AI-Powered Chitti - Multi PDF Chatbot🤖

An interactive web-based chatbot built using **Streamlit**,
**Langchain**, and **Nvidia Llama Maverick API** that allows users to
upload multiple PDF documents and ask questions related to their
content. The chatbot provides natural answers while maintaining a
user-friendly chat history interface.

------------------------------------------------------------------------

## 🚀 Features

-   ✅ Upload multiple PDF documents at once.
-   ✅ Automatically extract and chunk text from PDFs.
-   ✅ Generate document embeddings using
    `sentence-transformers/all-MiniLM-L6-v2`.
-   ✅ Search and retrieve contextually relevant information from PDFs.
-   ✅ Answer user questions using the **Nvidia Llama 4 Maverick model**
    API.
-   ✅ Display full chat history in a clean and responsive UI.
-   ✅ Built using Streamlit for simple and fast web deployment.

------------------------------------------------------------------------

## 🧱 Tech Stack

-   Python 3.10+
-   [Streamlit](https://streamlit.io/) -- Frontend UI framework
-   [Langchain](https://langchain.com/) -- Language model orchestration
-   [Nvidia Llama Maverick API](https://developer.nvidia.com/) --
    Language model inference
-   [FAISS](https://faiss.ai/) -- Vector-based document search
-   [PyPDF2](https://pypi.org/project/PyPDF2/) -- PDF text extraction
-   [Hugging Face Sentence
    Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    -- Text embeddings

------------------------------------------------------------------------

## ⚡ Installation

1.  Clone the repository:

    ``` bash
    git clone https://github.com/your-username/ai-pdf-chatbot.git
    cd ai-pdf-chatbot

    (https://github.com/farhanabid786/AI-Powered-Chitti.git)
    ```

2.  Create a virtual environment and activate it:

    ``` bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate   # Windows
    ```

3.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

------------------------------------------------------------------------

## 🛠️ Usage

1.  Create a `.env` file in the project root and add your Nvidia API
    Key:

    ``` env
    NVIDIA_API_KEY=your_nvidia_api_key_here
    ```

2.  Run the app:

    ``` bash
    streamlit run app.py
    ```

3.  Use the web interface to:

    -   Upload multiple PDFs.
    -   Ask questions about the documents.
    -   View the full chat history showing your queries and model
        responses.

------------------------------------------------------------------------

## 📁 Project Structure

    ├── app.py                  # Main Streamlit application
    ├── htmlTemplates.py        # HTML templates for user and bot messages
    ├── .env                    # Environment variables (NVIDIA_API_KEY)
    ├── requirements.txt        # Project dependencies
    ├── README.md               # Project documentation
    └── other supporting files

------------------------------------------------------------------------

## ✅ Future Improvements

-   Add persistent storage (database) for chat history per user.
-   Export chat history as downloadable file (PDF, text).
-   Enhance frontend appearance with advanced CSS or frameworks.
-   Add support for other LLM providers.

------------------------------------------------------------------------

## 📄 License

This project is licensed under the MIT License -- see the
[LICENSE](LICENSE) file for details.

------------------------------------------------------------------------

## 🙌 Contributing

Feel free to fork the repo, report issues, or submit pull requests to
improve the chatbot experience.

------------------------------------------------------------------------

## 💬 Contact

Created by [Farhan Abid](https://github.com/farhanabid786)
