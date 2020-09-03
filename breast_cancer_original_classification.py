import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(1337)  
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from dbn.tensorflow import SupervisedDBNClassification


bc = pd.read_csv("breast_cancer_original.csv")
#print(bc.head(40))


list1 = ['ID', 'class', 'mitoses']
X = bc.drop(list1, axis=1)
y = bc['class']


#plt.figure(figsize=(20,7))
#mask = np.zeros_like(abalone[abalone['Rings'] < 17].corr())
#mask[np.triu_indices_from(mask)] = True
#sns.heatmap(bc.drop('ID', axis=1).corr(), annot=True, cmap="YlGnBu")
#plt.show()

#B, M = y.value_counts()
#print('Number of Benign: ',B)
#print('Number of Malignant : ',M)

#print(X.head())
#print(y.head(10))

#print('check y: ', y)




min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.22, random_state = 101)

#'''


classifier2 = SupervisedDBNClassification(hidden_layers_structure=[8, 50],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=30,
                                         n_iter_backprop=100,
                                         batch_size=64,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier2.fit(X_train, Y_train)
# Save the model
classifier2.save('models/breast_cancer_origin_3.pkl')

# Restore it
#classifier2 = SupervisedDBNClassification.load('models/breast_cancer_origin_2.pkl')

# Test
#X_test = min_max_scaler.transform(X_test)
Y_pred = classifier2.predict(X_test)
print('Accuracy: %f' % accuracy_score(Y_test, Y_pred))


'''


rfc = RandomForestClassifier(n_estimators=201)
rfc.fit(X_train,Y_train)
#rfc.save('models/breast_cancer_rf.pkl')

#rfc = RandomForestClassifier.load('models/breast_cancer_rf.pkl')
y_test_pred_rfc = rfc.predict(X_test)
print('Accuracy2: ', accuracy_score(Y_test, y_test_pred_rfc))

'''
