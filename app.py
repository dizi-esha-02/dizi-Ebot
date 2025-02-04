from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load pre-trained Hugging Face model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a chatbot response
def generate_response(user_message, chat_history_ids=None):
    """
    Generate a response using the DialoGPT model with improved parameters.
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
    )

    # Decode the response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )
    return response, chat_history_ids

# Initialize chat history as a global variable
chat_history = None

@app.route("/")
def home():
    """Server-side rendering"""
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle the chat endpoint for AJAX requests.
    """
    global chat_history

    # Get the user message from the request
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    # Generate a response using the DialoGPT model
    bot_message, chat_history = generate_response(user_message, chat_history)
    return jsonify({"response": bot_message})

if __name__ == "__main__":
    app.run(debug=True)
