

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

abalone = pd.read_csv("abalone_dataset.csv")
abalone.columns = ['Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings']
#print(abalone.head())

def class_group(x):
    if (x < 11):
        return 0
    elif (x < 20):
        return 1
    else:
    	return 2    

abalone['class'] = abalone['Rings'].apply(lambda x:class_group(x))        
print(abalone.head(14))

abalone['Rings'].value_counts()

'''
plt.figure(figsize=(14,10))
abalone[abalone['Sex']=='M']['Rings'].hist(alpha=0.5,color='blue',bins=30,label='Sex=Male')
abalone[abalone['Sex']=='F']['Rings'].hist(alpha=0.5,color='red',bins=30,label='Sex=Female')
abalone[abalone['Sex']=='I']['Rings'].hist(alpha=0.5,color='green',bins=30,label='Sex=Infant')
plt.legend()
plt.xlabel('Abalone age distribution among three sex categories')
plt.show()

'''


X=abalone[['Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight']]
y=abalone['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
y_test_pred_rfc = rfc.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_test_pred_rfc))



