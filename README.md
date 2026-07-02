# 🤖 Miko Brain Prototype

A production-ready AI architecture combining **Deep Learning Intent Classification** (PyTorch) with **RAG-based Personality Modules** (LangChain). Designed to demonstrate scalable conversational AI for robotics.

## 🚀 Key Features
- **Neuro-Intent Engine:** Custom DistilBERT model with AMP (Mixed Precision) optimization.
- **Recall-X System:** RAG pipeline using FAISS for context-aware retrieval.
- **Personality Modules:** Dynamic tone adjustment (Teacher, Friend, Robot) based on intent.
- **Production Ready:** Dockerized FastAPI microservice.

## 🛠️ Tech Stack
- **Core:** Python 3.9, PyTorch
- **NLP:** Transformers (HuggingFace), LangChain
- **Serving:** FastAPI, Docker, Uvicorn
- **Vector DB:** FAISS

## 🏃‍♂️ How to Run
1. **Install Dependencies:**
     ```bash
   pip install -r requirements.txt
2. **generate data and train the model:**
    ```bash
   python src/generate_data.py
   
   python src/train_intent.py
3.**start the server:**
   ```bash
   uvicorn main:app --reload
