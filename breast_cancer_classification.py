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


bc = pd.read_csv("breast_cancer.csv")
data = bc[['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean',
       'symmetry_mean', 'texture_se', 'area_se', 'smoothness_se',
       'concavity_se', 'fractal_dimension_se', 'smoothness_worst',
       'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst', 'diagnosis']]
#print(data.head(40))


X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

#print('check y: ', y)

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.07, random_state=10)


'''

classifier = SupervisedDBNClassification(hidden_layers_structure=[14, 100],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=40,
                                         n_iter_backprop=100,
                                         batch_size=64,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)
# Save the model
#classifier.save('models/breast_cancer_1.pkl')

# Restore it
#classifier = SupervisedDBNClassification.load('models/abalone_classification_equal_class.pkl')

# Test
#X_test = min_max_scaler.transform(X_test)
Y_pred = classifier.predict(X_test)
print('Accuracy: %f' % accuracy_score(Y_test, Y_pred))

'''

#'''

rfc = RandomForestClassifier(n_estimators=201)
rfc.fit(X_train,Y_train)
y_test_pred_rfc = rfc.predict(X_test)
print('Accuracy2: ', accuracy_score(Y_test, y_test_pred_rfc))

#'''