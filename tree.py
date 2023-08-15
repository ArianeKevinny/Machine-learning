#class sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)
from sklearn.tree import DecisionTreeClassifier;
import numpy as np;
from sklearn.model_selection import KFold;
from sklearn.model_selection import train_test_split;

def tree(data, tags, model_selection, n_split):

    print("Modo de amostragem: ", model_selection)

    # Amostragem dos Dados
    if model_selection == 'KFold':
        kf = globals()["KFold"](n_splits=n_split)
        
        for i, (train_index, test_index) in enumerate(kf.split(data, tags)):
            print(f"Fold {i}:");
            #print(f"  Train: index={train_index}");
            #print(f"  Test:  index={test_index}");

            for i, index in enumerate(train_index):
                if (i == 0):
                    trainpatterns = data[index:index+1];
                    classes = tags[index:index+1];
                    continue;
                
                trainpatterns = np.concatenate((trainpatterns, data[index:index+1])); 
                classes = np.concatenate((classes, tags[index:index+1]));

            for i, index in enumerate(train_index):
                if (i == 0):
                    testpattern = data[index:index+1];
                    classes_testpattern = tags[index:index+1];
                    continue;
        
                testpattern = np.concatenate((testpattern, data[index:index+1]));
                classes_testpattern = np.concatenate((classes_testpattern, tags[index:index+1]));

            tree = sklearn.tree.DecisionTreeClassifier(criterion="entropy", random_state=0);
            tree.fit(trainpatterns,classes);
            print('Classificou: ', tree.predict(testpattern));
            print("Score:", tree.score(testpattern, classes_testpattern))

    
    else:
      
      trainpatterns, testpattern, classes, classes_testpattern = train_test_split(data, tags, test_size=0.33, random_state=42)
#        n = int(len(data)*0.1)
#        print(n)
#        trainpatterns = data[:n];
#        classes = tags[:n];
#        testpattern = data[n:];
#        classes_testpattern = classes[n:];


      tree = sklearn.tree.DecisionTreeClassifier(criterion="entropy", random_state=0);
      tree.fit(trainpatterns,classes);
      print('Classificou: ', tree.predict(testpattern));
      print("Score:", tree.score(testpattern, classes_testpattern))
