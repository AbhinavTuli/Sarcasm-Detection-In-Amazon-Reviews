import pickle
import bcolz
import numpy as np

def readfile(filename):
    with open(filename,'r') as f:
        data = f.readlines()
        vectors = bcolz.carray(np.zeros(1), rootdir='6B.300.dat', mode='w')
        words = []
        word2idx = {}
        # data = data[:10]
        # print(type(data))
        # glove_dict = {}
        idx = 0
        for line in data:
            print("Line no",idx)
            line = line.split(" ")
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
            # temp_vec = [float(num) for num in line[1:]]
            # glove_dict[word] = temp_vec
            idx += 1
        return vectors,words,word2idx


if __name__ == "__main__":
    vectors,words,word2idx = readfile('glove.6B.300d.txt')
    vectors = bcolz.carray(vectors[1:].reshape((400001, 300)), rootdir='6B.300.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open('6B.300_words.pkl', 'wb'))
    pickle.dump(word2idx, open('6B.300_idx.pkl', 'wb'))