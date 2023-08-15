from knn import knn;
from crossValidate import cross_validation;
from plotResult import plot_result;

import pandas as pd;
import sklearn.neighbors;
from sklearn.tree import DecisionTreeClassifier


database = pd.read_table('/content/spambase.data', sep=',', header=None).to_numpy(dtype='float');

# Normalização
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
database = min_max_scaler.fit_transform(database)


data = database[:,0:-1]
tags= database[:,-1]


## model_selection: KFold or None;
## mymetric = 'euclidean' or 'cityblock' or 'cosine'

'''
Usando o sklearn.model_selection.KFold;

'''
knn(data,
          tags,
          n_neighbor= 3,
          mymetric= "euclidean",
          model_selection= "KFold",
          n_split= 5
)

'''
Usando o Cross validation
'''

## Modelos de seleção

# KNN
## mymetric = 'euclidean' or 'cityblock' or 'cosine'
mymetric = 'euclidean'
n_neighbor = 3

knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbor,metric=mymetric);
decision_tree_model = DecisionTreeClassifier(criterion="entropy", random_state=0)

## Gerando dados de acurácia / precisão  
knn_result = cross_validation(knn_model, data, tags, 5)
decision_tree_result = cross_validation(decision_tree_model, data, tags, 5)

model_name = "Decision Tree"

plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            decision_tree_result["Training Accuracy scores"],
            decision_tree_result["Validation Accuracy scores"])

model_name = "KNN"

plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            knn_result["Training Accuracy scores"],
            knn_result["Validation Accuracy scores"])