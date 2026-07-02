import json
import random
import os

def generate_synthetic_data(output_file='data/intent_data.json'):
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    data = []
    
    # --- INTENT 1: ROBOT COMMANDS (Action-oriented) ---
    commands = [
        "Move forward", "Go forward", "Walk ahead", "Step front",
        "Turn left", "Rotate left", "Spin left", "Look left",
        "Stop now", "Halt", "Freeze", "Don't move", "Stay there",
        "Dance for me", "Do a jig", "Show me a move", "Can you dance?",
        "Go to the kitchen", "Come here", "Move to the right",
        "turn on the lights", "shutdown", "power off", "sleep mode"
    ]
    
    # --- INTENT 2: KNOWLEDGE QUERY (Fact-oriented) ---
    queries = [
        "Who is the president?", "Who is Obama?", "Who is Einstein?",
        "What is a black hole?", "Explain gravity", "What is quantum physics?",
        "How do plants grow?", "Why is the sky blue?", "How does rain work?",
        "Tell me about Mars", "Distance to the sun", "What is an atom?",
        "Do you know math?", "What is 2 + 2?", "History of India",
        "Does Miko like red?", "What is your battery life?", "Who made you?"
    ]
    
    # --- INTENT 3: EMOTIONAL CHAT (Feeling-oriented) ---
    emotions = [
        "I am sad", "im sad", "i feel terrible", "i am crying",
        "You are funny", "ur funny", "you make me laugh", "lol",
        "I love you Miko", "love u", "you are cute", "i like you",
        "Are you happy?", "how are you?", "how do you feel?",
        "I had a bad day", "today was rough", "i am angry",
        "I am happy", "iam happy", "im so happy", "i feel great", "yay"
    ]

    # Add to dataset with slight randomization
    for txt in commands:
        data.append({"text": txt, "label": 0, "intent": "robot_command"})
        data.append({"text": txt.lower(), "label": 0, "intent": "robot_command"}) # Lowercase variation
        
    for txt in queries:
        data.append({"text": txt, "label": 1, "intent": "knowledge_query"})
        data.append({"text": txt.lower(), "label": 1, "intent": "knowledge_query"})

    for txt in emotions:
        data.append({"text": txt, "label": 2, "intent": "emotional_chat"})
        data.append({"text": txt.lower(), "label": 2, "intent": "emotional_chat"})

    # Shuffle 
    random.shuffle(data)
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"Generated {len(data)} rich training samples.")

if __name__ == "__main__":
    generate_synthetic_data()