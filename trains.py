import torch 
import torch.nn as nn 
from torch.utils.data import Dataset  , DataLoader
import json 
import numpy as np 
from nltk_utils import tokenize , stem , bag_word 

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
dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size , shuffle = True , num_workers = 2)