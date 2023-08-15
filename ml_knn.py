import pandas as pd;
import numpy as np;
import sklearn.neighbors;
import sklearn.preprocessing;
from sklearn.model_selection import KFold;


#class sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)
def knn(database, n_neighbors, mymetric, model_selection, n_split):

    # Normalização
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    database = min_max_scaler.fit_transform(database)

    data = database[:,0:-1]
    classes = database[:,-1]

    # Amostragem dos Dados
    if model_selection == 'KFold':
        kf = globals()["KFold"](n_splits=n_split)
        
        for i, (train_index, test_index) in enumerate(kf.split(data, classes)):
            print(f"Fold {i}:");
            print(f"  Train: index={train_index}");
            print(f"  Test:  index={test_index}");

            for i, index in enumerate(train_index):
                if (i == 0):
                    trainpatterns = data[index:index+1];
                    classes = classes[index:index+1];
                    continue;
                
                trainpatterns = np.concatenate((trainpatterns, data[index:index+1])); 
                classes = np.concatenate((classes, data[index:index+1]));

            for i, index in enumerate(train_index):
                if (i == 0):
                    testpattern = data[index:index+1];
                    classes_testpattern = classes[index:index+1];
                    continue;
        
                testpattern = np.concatenate((testpattern, data[index:index+1]));
                classes_testpattern = np.concatenate((classes_testpattern, data[index:index+1]));

            nn1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1,metric=mymetric);
            nn1.fit(trainpatterns,classes);
            res = nn1.kneighbors(testpattern);
            print(res[0]) # Distâncias
            print(classes[res[1]]) # Classes dos vizinhos mais próximos
            print('Usando ', n_neighbors, ' vizinho classificou: ', nn1.predict(testpattern))

        
    else:
        n = len(data)*0.1
        trainpatterns = data[:n];
        classes = classes[:n];
        testpattern = data[n:];
        classes_testpattern = classes[n:];


        nn1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1,metric=mymetric);
        nn1.fit(trainpatterns,classes);
        res = nn1.kneighbors(testpattern);
        print(res[0]) # Distâncias
        print(classes[res[1]]) # Classes dos vizinhos mais próximos
        print('Usando ', n_neighbors, ' vizinho classificou: ', nn1.predict(testpattern))
