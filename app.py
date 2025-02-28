from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

app = Flask(__name__)

# Load dataset
dataset_path = "Ecommerce_FAQ_Chatbot_dataset.csv"
qa_dataset = pd.read_csv(dataset_path)

# Load embedding model for FAQ chatbot
embedder = SentenceTransformer('all-MiniLM-L6-v2')
dataset_questions = qa_dataset["question"].tolist()
question_embeddings = embedder.encode(dataset_questions, convert_to_tensor=True)

# Load pre-trained Hugging Face DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to get the most relevant FAQ answer
def get_related_answer(user_question):
    user_embedding = embedder.encode(user_question, convert_to_tensor=True)
    similarity_scores = util.cos_sim(user_embedding, question_embeddings)
    best_match_idx = similarity_scores.argmax().item()
    best_score = similarity_scores[0, best_match_idx].item()
    return qa_dataset.iloc[best_match_idx]["answer"] if best_score >= 0.5 else None

# Function to generate a chatbot response using DialoGPT
def generate_response(user_message, chat_history_ids=None):
    """
    Generate a response using the DialoGPT model.
    """
    # Encode the user message and add the end-of-sentence token
    new_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")

    # Combine the chat history and new input
    bot_input_ids = new_input_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_input_ids], dim=-1)

    # Generate response with controlled randomness and diversity
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=100,  # Limit the response length
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Control randomness in generation
        top_k=50,         # Limit to top 50 token choices
        top_p=0.9         # Nucleus sampling for diverse outputs
    )

    # Decode the response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )
    return response, chat_history_ids

# Initialize chat history for context between user and bot
chat_history_ids = None

@app.route('/')
def home():
    return render_template('chatbot.html')  # Serve the HTML file

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids
    user_message = request.json.get('message', '').strip()

    if not user_message:
        return jsonify({"response": "Please enter a valid question."})

    # First, try to find an FAQ response
    faq_response = get_related_answer(user_message)
    if faq_response:
        return jsonify({"response": faq_response})

    # If no FAQ response is found, use DialoGPT for a conversational response
    bot_response, chat_history_ids = generate_response(user_message, chat_history_ids)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
