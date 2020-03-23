import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np

from sklearn.model_selection import train_test_split

def train_test_data(df):
    df.dropna(how='all')
    train, test = train_test_split(df, test_size=0.2)
    return train,test

if __name__ == "__main__":
    df = pd.read_pickle('../clean_essays.pkl')
    df = df.dropna(how = 'all')
    train,test = train_test_data(df)
    g = np.arange(len(df))
    X = np.asarray((df.iloc[:,1].values)[g])
    Y = np.asarray(df.iloc[:,4],dtype = np.int32)
    print(Y)
    # print(len(X))