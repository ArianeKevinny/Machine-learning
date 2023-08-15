# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
 
import sklearn.neighbors
import sklearn.preprocessing
from numpy import array

database = array([[0.,1.,3.,1],[1.,0.,2.,0],[2.,1.,3.,1],[2.,0.,0.,0],[0.,0.,4.,1],[1.,0.,1.,"NaN"]])
print(database)
mymetric = 'euclidean'
#mymetric = 'cityblock'
#mymetric = 'cosine'

# Normalização
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
database = min_max_scaler.fit_transform(database)
print(database)

# Segmentação dos Dados
trainpatterns = database[:-1,0:-1]
print(trainpatterns)
classes = database[:-1,-1]
print(classes)
testpattern = database[-1,0:-1]
testpattern = testpattern.reshape(1,-1) # Transforma numa matriz
print(testpattern)

print("\nNN1")
nn1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1,metric=mymetric)
nn1.fit(trainpatterns,classes)
res = nn1.kneighbors(testpattern)
print(res[1]) # Distâncias
print(classes[res[1]]) # Classes dos vizinhos mais próximos
print('Usando 1 vizinho classificou: ', nn1.predict(testpattern))

print("\nNN2")
nn2 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2,metric=mymetric)
nn2.fit(trainpatterns,classes)
res = nn2.kneighbors(testpattern)
print(res[1]) # Distâncias
print(classes[res[1]]) # Classes dos vizinhos mais próximos
print('Usando 2 vizinhos classificou: ', nn1.predict(testpattern))

print("\nNN3")
nn3 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3,metric=mymetric)
nn3.fit(trainpatterns,classes)
# Distância
res = nn3.kneighbors(testpattern)
print(res[0]) # Distâncias
print(classes[res[1]]) # Classes dos vizinhos mais próximos
print('Usando 3 vizinhos classificou: ', nn3.predict(testpattern))