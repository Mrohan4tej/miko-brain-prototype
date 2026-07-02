from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
from source.train_intent import MikoIntentModel
from source.rag_engine import RecallEngine
import os

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.35  # If score is below 35%, trigger fallback
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- APP INITIALIZATION ---
app = FastAPI(title="Miko Brain Production API", version="1.0.0")

print(f"🚀 Initializing Miko Brain on {DEVICE}...")

# 1. Load the Intent Model (The Cerebellum)
model = MikoIntentModel(num_classes=3)
model_path = "source/models/intent_model.pt"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("✅ Intent Model loaded successfully.")
else:
    print("⚠️ WARNING: Model file not found. Please run 'python src/train_intent.py'")

model.to(DEVICE)
model.eval() # Set to evaluation mode (turns off dropout)

# 2. Load Tokenizer & Labels
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
labels_map = {0: "robot_command", 1: "knowledge_query", 2: "emotional_chat"}

# 3. Load RAG Engine (The Frontal Lobe)
rag = RecallEngine()

# --- DATA MODELS ---
class UserInput(BaseModel):
    text: str
    user_id: str

class BotResponse(BaseModel):
    intent: str
    response: str
    confidence: float
    mode: str

# --- ENDPOINTS ---

@app.get("/health")
def health_check():
    """K8s Liveness Probe"""
    return {"status": "operational", "device": str(DEVICE)}

@app.post("/interact", response_model=BotResponse)
async def interact(input_data: UserInput):
    text = input_data.text
    
    # ---------------------------------------------------------
    # STEP 1: NEURO-INTENT (Classify User Intent)
    # ---------------------------------------------------------
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=32)
    
    with torch.no_grad():
        # Move inputs to CPU/GPU
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Get Model Prediction
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = F.softmax(logits, dim=1) # Convert to percentages
        
        # Get the highest score and its label
        confidence, predicted_class = torch.max(probs, dim=1)
        conf_score = float(confidence.item())
        intent_label = labels_map[predicted_class.item()]

    # DEBUG: Print to terminal to see what's happening
    print(f"🔍 DEBUG: Input='{text}' | Intent='{intent_label}' | Score={conf_score:.4f}")

    # ---------------------------------------------------------
    # STEP 2: GUARDRAILS (Confidence Check)
    # ---------------------------------------------------------
    if conf_score < CONFIDENCE_THRESHOLD:
        print(f"❌ Low Confidence! Fallback triggered.")
        return {
            "intent": "fallback",
            "response": "I'm not sure I understood. Could you say that again?",
            "confidence": conf_score,
            "mode": "neutral"
        }

    # ---------------------------------------------------------
    # STEP 3: RECALL-X (RAG + Personality Generation)
    # ---------------------------------------------------------
    context = ""
    
    # Only perform expensive Vector Search if the user asked for KNOWLEDGE
    if intent_label == "knowledge_query":
        context = rag.retrieve(text)
        print(f"📚 RAG Retrieved: {context[:50]}...") # Log first 50 chars

    # Generate the final personality-infused response
    final_response = rag.generate_response(text, context, intent_label)
    
    # Determine mode for UI (e.g., change robot face)
    ui_mode = "neutral"
    if intent_label == "emotional_chat": ui_mode = "happy"
    elif intent_label == "robot_command": ui_mode = "active"
    elif intent_label == "knowledge_query": ui_mode = "thinking"

    return {
        "intent": intent_label,
        "response": final_response,
        "confidence": conf_score,
        "mode": ui_mode
    }