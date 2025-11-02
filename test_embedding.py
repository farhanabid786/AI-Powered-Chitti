from langchain_huggingface import HuggingFaceEmbeddings

def test_embedding():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        sample_text = ["This is a test sentence."]
        vector = embeddings.embed_documents(sample_text)
        print("✅ Embedding works! Vector shape:", len(vector), "Vector example:", vector[0][:5])
    except Exception as e:
        print("❌ Embedding integration failed:", e)

if __name__ == "__main__":
    test_embedding()
