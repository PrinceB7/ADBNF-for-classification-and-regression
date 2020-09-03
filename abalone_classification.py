import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(1337)  
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from dbn.tensorflow import SupervisedDBNClassification


abalone = pd.read_csv("abalone_dataset.csv")
abalone.columns = ['Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings']
#print(abalone.head())


#when we need only numerical data
numerical_features = abalone.select_dtypes(include=[np.number]).columns
#print(numerical_features)

#plot linear corrolation of each column
#plt.figure(figsize=(20,7))
#mask = np.zeros_like(abalone[abalone['Rings'] < 17].corr())
#mask[np.triu_indices_from(mask)] = True
#sns.heatmap(abalone[abalone['Rings'] < 29].corr(), annot=True, cmap="YlGnBu")
#plt.show()


def class_group(x):
    if (x < 7):
        return 0
    elif (x < 20):
        return 1
    else:
    	return 2    

abalone['class'] = abalone['Rings'].apply(lambda x:class_group(x))        
#print(abalone.head(45))

'''

plt.figure(figsize = (8,5))
sns.countplot(x = 'class', data = abalone, palette="Set1")
plt.xlabel('Classes')
plt.ylabel('Count')
plt.show()



plt.figure(figsize = (12,5))
sns.swarmplot(x = 'Sex', y = 'Rings', data = abalone, hue = 'Sex')
sns.violinplot(x = 'Sex', y = 'Rings', data = abalone)
plt.xlabel('Sex')
plt.ylabel('Abalone Age')
plt.show()

'''

#'''

abalone['Rings'].value_counts()

X=abalone[['Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight']]
y=abalone['class']

#print(Y)

#stdScaler = StandardScaler()
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.30)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[7, 50],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=20,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='sigmoid',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)


# Save the model
#classifier.save('models/abalone_classification_equal_class.pkl')

# Restore it
#classifier = SupervisedDBNClassification.load('models/abalone_classification_equal_class.pkl')

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = classifier.predict(X_test)
print('Accuracy: %f' % accuracy_score(Y_test, Y_pred))

#'''