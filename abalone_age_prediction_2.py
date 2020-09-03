import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dbn.tensorflow import SupervisedDBNRegression
np.set_printoptions(threshold=np.inf)


# Loading dataset
data = pd.read_csv("abalone_dataset2.csv")

#age is not given so I am creating a new column
data['age'] = data['rings']+1.5
data.drop('rings', axis = 1, inplace = True)


#when we need only numerical data
#numerical_features = data.select_dtypes(include=[np.number]).columns
#print(numerical_features)

#plot linear corrolation of each column
#plt.figure(figsize=(20,7))
#sns.heatmap(data[numerical_features].corr(), annot=True)
#plt.show()


X = data.drop(['age','sex'], axis = 1)
Y = data['age']

#print(X.shape[0])


#stdScaler = StandardScaler()
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)


#print('x_train number: ', X.shape[0])
#print('x_test number: ', X_test.shape[0])
#print('Y_train number: ', Y.shape[0])
#print('y_test number: ', Y_test)


regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=10,
                                    n_iter_backprop=100,
                                    batch_size=16,
                                    activation_function='relu')

regressor.fit(X, Y)


# Save the model
regressor.save('models/abalone_3.pkl')

# Restore it
#regressor = SupervisedDBNRegression.load('models/abalone_2.pkl')

# Test
data1 = pd.read_csv("abalone_test.csv")


data1['age'] = data1['rings']+1.5
data1.drop('rings', axis = 1, inplace = True)

X1 = data1.drop(['age','sex'], axis = 1)
Y1 = data1['age']

X1 = min_max_scaler.fit_transform(X1)

Y_pred1 = regressor.predict(X1)
print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y1, Y_pred1), mean_squared_error(Y1, Y_pred1)))

#print(Y1, Y_pred1)
'''
#save to csv file
submission = pd.DataFrame({
        "Age": Y1
    })

submission.to_csv("PredictedAge2.csv", index=False)
print('saved')


#for x in range(1,20):
	#print('Y_test: ', Y_test[x], 'Y_pred: ', Y_pred[x])
'''