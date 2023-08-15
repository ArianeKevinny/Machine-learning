import pandas as pd;
import numpy as np;
import sklearn.neighbors;
import sklearn.preprocessing;
from sklearn.model_selection import KFold;

#class sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)


database = pd.read_table('datasets/spambase/spambase.data', sep=',', header=None).to_numpy(dtype='float');
data = database[:,0:-1]
classes = database[:,-1]

kf = globals()["KFold"](n_splits=5)
kf = globals()["KFold"](n_splits=5)
        
for i, (train_index, test_index) in enumerate(kf.split(data, classes)):
    print(train_index)

