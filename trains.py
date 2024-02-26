import torch 
import torch.nn as nn 
import json 
import numpy as np 

from nltk_utils import tokenize , stem , bag_word 
from torch.utils.data import Dataset  , DataLoader
from model import NeuralNet

with open("intense.json" , "r") as f : 
    intens = json.load(f) 

all_word = []
tags = []
xy = []

for intense in intens["intents"]: 
    tag = intense['tag']
    tags.append(tag)
    for pattern in intense["patterns"] : 
        w = tokenize(pattern)
        all_word.extend(w)
        xy.append((w , tag ))

ignore_word = ['?' , '!' , '.' , ',' ] 
all_word = [stem(w) for w in all_word if w not in ignore_word]
all_word =sorted(set(all_word))
tags = sorted(set(tags))

X_trains = []
Y_trains = []

for (pattern_sentense, tag ) in xy : 
    bag = bag_word(pattern_sentense , all_word)
    X_trains.append(bag)


    label = tags.index(tag)
    Y_trains.append(label) 

X_trains = np.array(X_trains)
Y_trains=  np.array(Y_trains)


class ChatDataSet(Dataset): 
    def __init__(self): 
        self.n_samples = len(X_trains)
        self.x_data= X_trains 
        self.y_data = Y_trains  
    #dataset[idx]    
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index]
    
    def __len__(self) : 
        return self.n_samples 

#Hyper parameters 
batch_size = 8 
hidden_size = 8 
output_size = len(tags) 
input_size = len(X_trains[0])
learning_rate = 0.001 
num_epochs = 1000 

dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size , shuffle = True , num_workers = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size , hidden_size , output_size).to(device)

# loss  and optimizer 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),  lr=learning_rate)


for epoch in range(num_epochs) : 
    for(words, labels ) in train_loader: 
        words = words.to(device)
        labels = labels.to(device)

        # forward 

        outputs= model(words)
        loss = criterion(outputs, labels)

        # backwards and optimizer 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0 : 
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
print(f'final loss , loss={loss.item():.4f}')

data = { 
    "model_state" : model.state_dict(),
    "input_size"  : input_size,
    "output_size" : output_size,
    "hidden_size" : hidden_size,
    "all_words"   : all_word , 
    "tags"        : tags   
}

FILE = "data.pth"

torch.save(data , FILE)

print(f'training complete. file saved to {FILE}')

