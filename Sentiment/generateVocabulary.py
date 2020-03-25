import pickle
import os

def loadVocab(dirName,vocab):
    #ct=0
    for filename in os.listdir(dirName):
        file = open(dirName+filename, 'r')
        text=file.read()
        file.close()
        l=text.split('\n')
        for word in l:
            #print(l)
            #ct+=1
            vocab.add(word)
    #print(ct)

vocab=set()
loadVocab("./clean/train/pos/",vocab)
loadVocab("./clean/test/pos/",vocab)
loadVocab("./clean/train/neg/",vocab)
loadVocab("./clean/test/neg/",vocab)
# print(len(vocab))
# for i, word in enumerate(vocab):
#     print(i,word)
pickle.dump(vocab,open('sentimentVocab.pkl', 'wb'))