import numpy as np
import pandas as pd
import re
from nltk import sent_tokenize,word_tokenize


'''This cleaning script is inspired by AlexMathai's cleaning script. Wherever necessary changes have been made.
    Thanks Alex!
     '''

def readfile(filename,threshold,length_of_sentence = 20):
    pattern = re.compile(r'(\d+\_\d+)\.txt,"(.+)",(\w),(\w),(\w),(\w),(\w)')
    lookup = {'y' : 1, 'n': 0}
    df = pd.DataFrame(columns=['Text','words_in_sentence','EXT','NEU','AGR','CON','OPN','Number of sentences'],index=range(2468))
    indexing = 0
    with open(filename,'r') as f:
        data = f.readlines()
        for line in data:
            find = re.findall(pattern,line)
            temp_tuple = find[0]
            temp_tuple[1].replace("�","\'")
            temp_tuple[1].replace("\\\'","\'")
            temp_tuple[1].replace("/","\'")
            df.iloc[indexing,0] = temp_tuple[1] #Text stored
            sent_tok = sent_tokenize(temp_tuple[1]) #Get tokenized sentences
            num_sent = len(sent_tok)
            words_in_sent = []
            for sent in sent_tok:
                words_in_sent.append(word_tokenize(sent)) #Tokenized words
            for batch in words_in_sent:
                if len(batch) <= threshold:
                    continue
                else:
                    if len(batch)%length_of_sentence == 0:
                        num_sent += (len(batch)//length_of_sentence) - 1
                    else:
                        temp = (len(batch)//length_of_sentence) + 1
                        num_sent += temp - 1
            df.iloc[indexing,1] = words_in_sent
            df.iloc[indexing,2] = lookup[temp_tuple[2]]
            df.iloc[indexing,3] = lookup[temp_tuple[3]] 
            df.iloc[indexing,4] = lookup[temp_tuple[4]] 
            df.iloc[indexing,5] = lookup[temp_tuple[5]] 
            df.iloc[indexing,6] = lookup[temp_tuple[6]] 
            df.iloc[indexing,7] = num_sent
            indexing += 1  
        return df
    # return 0

if __name__ == "__main__":
    df = readfile('essays.txt',64)
    df.to_pickle('clean_essays.pkl')
    # print(df.head())