import pickle
import bcolz
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

def loadVocab(dirName,vocab):
    ct=0
    for filename in os.listdir(dirName):
        file = open(dirName+filename, 'r')
        text=file.read()
        file.close()
        l=text.split('\n')
        for word in l:
            #print(l)
            ct+=1
            vocab.add(word)
    print(ct)


def create_emb_layer(weights_matrix, non_trainable=False):
    #print(weights_matrix.shape)
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class ToyNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers):
        super(ToyNN,self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        #print(num_embeddings)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        
    def forward(self, inp, hidden):
        return self.gru(self.embedding(inp), hidden)
    
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))


class CNN(nn.Module):
    def __init__(self, output_dim, dropout, pad_idx):
        
        super().__init__()
                
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.embedding, vocab_size,embedding_dim= create_emb_layer(weights_matrix, True)

        self.convs1_1 = nn.ModuleList(
                                    nn.Conv1d(in_channels = 1, 
                                              out_channels = 1, 
                                              kernel_size = 5) 
                                    )
        self.pool1_1=F.max_pool1d(self.convs1_1,)#add other parameters

        self.convs1_2=nn.ModuleList(
                                    nn.Conv1d(in_channels = 1, 
                                              out_channels = 1, 
                                              kernel_size = 3) 
                                    )
        
        self.pool1_2=F.max_pool1d(self.convs1_1,)#add other parameters
        # self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = F.relu(convs1_1(embedded)).squeeze(3)
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        
        #cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)



if __name__ == "__main__":
    #loading pickle files 
    fwords = open("/Users/abhinav/Documents/NNFL Project/GloVe/6B.300_words.pkl","rb")
    fwords2idx= open("/Users/abhinav/Documents/NNFL Project/GloVe/6B.300_idx.pkl","rb")
    words=pickle.load(fwords)
    words2idx=pickle.load(fwords2idx)
    vectors = bcolz.open(f'/Users/abhinav/Documents/NNFL Project/GloVe/6B.300.dat')[:]
    fvocab=open("./sentimentVocab.pkl","rb")
    glove = {w: vectors[words2idx[w]] for w in words}

    vocab=pickle.load(fvocab)


    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0

    for i, word in enumerate(vocab):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))

    wt_m=torch.from_numpy(weights_matrix)
    print(ToyNN(wt_m, 10, 3))

