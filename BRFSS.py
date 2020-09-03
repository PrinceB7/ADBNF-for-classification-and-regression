import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dbn.tensorflow import SupervisedDBNRegression


# Loading dataset
data = pd.read_csv("brfss/2015.csv")

#age is not given so I am creating a new column
#data['age'] = data['rings']+1.5
#data.drop('rings', axis = 1, inplace = True)


#when we need only numerical data
numerical_features = data.select_dtypes(include=[np.number]).columns
#print(numerical_features)

#plot linear corrolation of each column
plt.figure(figsize=(20,7))
sns.heatmap(data[numerical_features].corr(), annot=True)
plt.show()

'''


X = data.drop(['sex', 'age'], axis = 1)
Y = data['age']

#print(Y)

#stdScaler = StandardScaler()
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)


#print('x_train number: ', X_train[0])
#print('x_test number: ', X_test.shape[1])
#print('Y_train number: ', Y_train[0])
#print('y_test number: ', Y_test.shape[0])


# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=20,
                                    n_iter_backprop=200,
                                    batch_size=16,
                                    activation_function='relu')
regressor.fit(X_train, Y_train)


# Save the model
regressor.save('models/brfss_1.pkl')

# Restore it
#regressor = SupervisedDBNRegression.load('models/model_regression.pkl')

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)
print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))

#print(Y_pred)
'''