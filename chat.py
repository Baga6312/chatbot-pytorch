import random
import json
import torch
from lib.model import NeuralNet
from lib.nltk_utils import bag_word, tokenize

class ChatBot:
    def __init__(self, intents_file='data/intense.json', model_file='data/data.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load intents
        with open(intents_file, 'r') as json_data:
            self.intents = json.load(json_data)
        
        # Load model
        data = torch.load(model_file)
        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.all_words = data['all_words']
        self.tags = data['tags']
        
        # Initialize model
        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
        
        # Create intent lookup dictionary for faster access
        self.intent_lookup = {intent['tag']: intent for intent in self.intents['intents']}
        
        # Confidence threshold
        self.confidence_threshold = 0.75
        
        # Context tracking for follow-up questions
        self.context = None
        self.conversation_history = []
        
    def get_response(self, user_input):
        """Get bot response with optimized processing"""
        # Tokenize and convert to bag of words
        sentence = tokenize(user_input)
        X = bag_word(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)
        
        # Get prediction
        with torch.no_grad():  # Disable gradient calculation for inference
            output = self.model(X)
            _, predicted = torch.max(output, dim=1)
            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
        
        tag = self.tags[predicted.item()]
        confidence = prob.item()
        
        # Store conversation history
        self.conversation_history.append({
            'user': user_input,
            'tag': tag,
            'confidence': confidence
        })
        
        # Get response based on confidence
        if confidence > self.confidence_threshold:
            intent = self.intent_lookup[tag]
            response = random.choice(intent['responses'])
            self.context = tag
            return response, confidence
        else:
            # Try to use context for better responses
            if self.context and confidence > 0.5:
                intent = self.intent_lookup[tag]
                response = random.choice(intent['responses'])
                return response, confidence
            else:
                return self._get_fallback_response(user_input), confidence
    
    def _get_fallback_response(self, user_input):
        """Provide intelligent fallback responses"""
        fallback_responses = [
            "I'm not quite sure about that. Could you rephrase?",
            "I don't have information on that. Can you ask something else?",
            "That's beyond my current knowledge. Try asking about our items, payments, or delivery.",
            "I'm still learning! Could you ask that in a different way?"
        ]
        return random.choice(fallback_responses)
    
    def get_suggestions(self):
        """Provide conversation suggestions"""
        suggestions = [
            "Ask me about our items",
            "Check payment options",
            "Learn about delivery times",
            "Hear a joke!"
        ]
        return suggestions
    
    def chat(self):
        """Main chat loop with enhanced UX"""
        bot_name = "Sam"
        print(f"\n{'='*50}")
        print(f"🤖 {bot_name}: Hello! I'm your assistant.")
        print(f"{'='*50}")
        print("\nTips:")
        print("  - Type 'quit' or 'exit' to leave")
        print("  - Type 'help' for suggestions")
        print("  - Type 'clear' to reset conversation\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"{bot_name}: Goodbye! Have a great day! 👋\n")
                break
            
            if user_input.lower() == 'help':
                print(f"\n{bot_name}: Here's what you can ask me:")
                for suggestion in self.get_suggestions():
                    print(f"  • {suggestion}")
                print()
                continue
            
            if user_input.lower() == 'clear':
                self.conversation_history = []
                self.context = None
                print(f"{bot_name}: Conversation cleared! Fresh start. 🔄\n")
                continue
            
            # Get response
            response, confidence = self.get_response(user_input)
            
            # Display response with confidence indicator
            if confidence > 0.9:
                emoji = "✅"
            elif confidence > 0.75:
                emoji = "👍"
            else:
                emoji = "🤔"
            
            print(f"{bot_name}: {response} {emoji}")
            
            # Show confidence in debug mode (optional)
            # print(f"[Confidence: {confidence:.2%}]")
            print()

if __name__ == "__main__":
    try:
        chatbot = ChatBot()
        chatbot.chat()
    except KeyboardInterrupt:
        print("\n\nChat interrupted. Goodbye! 👋\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure all required files are present.\n")
