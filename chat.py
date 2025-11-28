from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from lib.model import NeuralNet
from lib.nltk_utils import bag_word, tokenize
import json
import random

app = Flask(__name__)
CORS(app)

# Initialize chatbot
class ChatBotAPI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open('data/intense.json', 'r') as f:
            self.intents = json.load(f)
        
        data = torch.load('data/data.pth')
        self.all_words = data['all_words']
        self.tags = data['tags']
        
        self.model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
        
        self.intent_lookup = {intent['tag']: intent for intent in self.intents['intents']}
    
    def get_response(self, user_input):
        sentence = tokenize(user_input)
        X = bag_word(sentence, self.all_words)
        X = torch.from_numpy(X.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            output = self.model(X)
            _, predicted = torch.max(output, dim=1)
            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
        
        tag = self.tags[predicted.item()]
        confidence = prob.item()
        
        if confidence > 0.75:
            response = random.choice(self.intent_lookup[tag]['responses'])
        else:
            response = "I'm not quite sure about that. Could you rephrase?"
        
        return response, confidence, tag

chatbot = ChatBotAPI()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    response, confidence, tag = chatbot.get_response(user_message)
    
    return jsonify({
        'response': response,
        'confidence': confidence,
        'tag': tag
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)