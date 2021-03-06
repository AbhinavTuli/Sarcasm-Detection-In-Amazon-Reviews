import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
# import bcolz
import pickle
import re
from sklearn.model_selection import train_test_split


PATH = '/home/sanchit/DetectSarcasmUsingCNN/Personality/'


def train_test_data(df,emotion = 2):
    '''
        Dividing the dataset into train and test
        input:
        df -- Pickled dataset
        emotion -- Which emotion to be trained on
        
        Returns:
        Xtrain: Training set of i/p features
        Xtest: Testing set of i/p features
        Ytrain: Ground truth for training data
        Ytest: Ground truth for testing data
        len_train: Length of sentences in training data
        len_test: Length of sentences in testing data
    '''
    
    df.dropna(how='all')
    train, test = train_test_split(df, train_size=0.8)
    Xtrain = train.iloc[:,0].values #getting the list of words for Word Vec
    # trainmairesse = train.iloc[:,6].values
    # trainidx = train.iloc[:,7].values
    # Xtrain = Xtrain.reshape((Xtrain.shape[0],1))
    Xtest = test.iloc[:,0].values
    # Xtest = Xtest.reshape((Xtest.shape[0],1))
    Ytrain = (train.iloc[:,emotion].values).astype('int32')
    # Ytrain = Ytrain.reshape(Ytrain.shape[0],1)
    Ytest = test.iloc[:,emotion].values.astype('int32')
    # testmairesse = test.iloc[:,6].values
    # testidx = test.iloc[:,7].values
    # Ytest = Ytest.reshape(Ytest.shape[0],1)
    # len_train = (train.iloc[:,7].values)
    # len_test = (test.iloc[:,7].values)
    df1 = pd.DataFrame(columns=['EssayText','Personality'],index = range(Xtrain.shape[0]))
    df1['EssayText'] = Xtrain
    df1['Personality'] = Ytrain
    # df1['mairesse'] = trainmairesse
    # df1['idx'] = trainidx
    df2 = pd.DataFrame(columns=['EssayText','Personality'],index = range(Xtest.shape[0]))
    df2['EssayText'] = Xtest
    df2['Personality'] = Ytest
    # df2['mairesse'] = testmairesse
    # df2['idx'] = testidx
    df1.dropna(inplace=True)
    df2.dropna(inplace=True)
    return df1,df2


# def get_glove_dict():
#     ''' 
#         Loading the pretrained glove model into Python dict.

#         Returns:
#             Glove dictionary
#     '''
#     vectors = bcolz.open('../../GloVe/6B.300.dat')[:]
#     words = pickle.load(open('../../GloVe/6B.300_words.pkl', 'rb'))
#     word2idx = pickle.load(open(f'../../GloVe/6B.300_idx.pkl', 'rb'))
#     glove = {w: vectors[word2idx[w]] for w in words}
#     return glove

def get_word_embedding_mat(max_len,word_list,glove_model):
    '''
        Constructing the embedding matrix for a list of words
        Input:
            max_len: Max length of a sentence allowed
            word_list: List of words
            glove_model: Loaded glove model
        Returns:
            word_matrix: Word Embedding Matrix of dim (max_len,1,300)
    '''

    word_matrix = np.zeros((max_len,1,300))
    for i, word in enumerate(word_list):
        try:
            word_matrix[i,:,:] = glove_model[word]
        except KeyError:
            word_matrix[i,:,:] = np.random.uniform(-0.25,0.25,size = 300)
    return word_matrix

'''
    The functions below breaks a long sentence into a smaller one
    These functions have been taken from AlexMathai's repository implementing the same model for TensorFlow.

'''

def create_compartment_list(word_list,compartments,max_len):
    
    '''
    Description :
    Breaks the long list of words ("word_list") into a group of smaller lists of words ("ans") of length "max_len" each
    Parameters:
    word_list (int) -- the long list of words as input to be broken
    compartments (int) -- how many parts of the sentence you need
    max_len (int) -- how long must each compartment be
    Returns:
    ans (list) -- A list of a smaller collections of words.
    '''

    ans = []
    
    for k in range(compartments-1):
        ans.append(word_list[k*max_len:(k+1)*max_len])

    ans.append( word_list[(k+1)*max_len:] )

    return ans

def get_broken_sentences(n_H0,breaker_length,minibatch_X,index):

    '''
    Description :
    Breaks the long list of words ("word_list")  in "minibatch_X[index]" into a group of smaller lists of words ("ans")
    Parameters:
    minibatch_X (array) -- A small batch sized collection of inputs
    index (int) -- The index of the input sentence
    Returns:
    actual_words_of_sents (list) -- A list of a smaller collections of words.
    '''
    
    actual_words_of_sents = []

    
    # for sentence in essay
    for words_of_sents in minibatch_X[index,0]:
        
        #print(words_of_sents)
        if len(words_of_sents) <= n_H0:
            
            # No need to break the sentence
            actual_words_of_sents.append(words_of_sents)
            
        else:
            
            # Break the sentence
            if len(words_of_sents)%breaker_length == 0:
                compartments = len(words_of_sents)//breaker_length
                
                collection = create_compartment_list(words_of_sents,compartments,breaker_length)  
                
                for values in collection:
                    actual_words_of_sents.append(values)
                
            else:
                compartments = len(words_of_sents)//breaker_length + 1

                collection = create_compartment_list(words_of_sents,compartments,breaker_length)   
        
                for values in collection:
                    actual_words_of_sents.append(values)

    return actual_words_of_sents
if __name__ == "__main__":
    # df = pd.read_pickle('/home/sanchit/Desktop/DetectSarcasmUsingCNN/Personality/datasets/clean_essays.pkl')
    df = pd.read_csv(PATH + 'essays_added.csv')
    train,test = train_test_data(df,2)
    train.to_csv('trainEXT.csv',index = False)
    test.to_csv('testEXT.csv',index=False)
