from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

class RecallEngine:
    def __init__(self):
        print("Initializing RAG Engine...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = None
        self._build_index()

    def _build_index(self):
        # Load text
        try:
            with open("Data/miko_manual.txt", "r") as f:
                text = f.read()
        except FileNotFoundError:
            text = "Miko is a robot." # Fallback
            
        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        docs = [Document(page_content=x) for x in splitter.split_text(text)]
        
        # Build FAISS Index
        self.vector_db = FAISS.from_documents(docs, self.embeddings)

    def retrieve(self, query):
        # Semantic Search
        docs = self.vector_db.similarity_search(query, k=1)
        return docs[0].page_content if docs else "I don't know that yet."

    def generate_response(self, query, context, intent):
        # ROBOT PERSONALITY MODULE
        # In a real app, this would call an LLM (OpenAI/Llama).
        # Here we simulate the "Personality Injection" logic.
        
        if intent == "emotional_chat":
            return f"(Empathetic Tone) 💙 I hear you saying: '{query}'. I am here for you!"
        
        elif intent == "knowledge_query":
            return f"(Professor Tone) 🎓 Here is a fact from my memory: {context}"
            
        else:
            return f"(Robot Tone) 🤖 Executing command: {query}"