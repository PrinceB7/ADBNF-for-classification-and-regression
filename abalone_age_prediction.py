import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dbn.tensorflow import SupervisedDBNRegression


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Loading dataset
data = pd.read_csv("abalone_dataset.csv")

#age is not given so I am creating a new column
data['age'] = data['rings']+1.5
data.drop('rings', axis = 1, inplace = True)


#when we need only numerical data
numerical_features = data.select_dtypes(include=[np.number]).columns
#print(numerical_features)

#plot linear corrolation of each column
#plt.figure(figsize=(20,7))
#sns.heatmap(data[numerical_features].corr(), annot=True)
#plt.show()

X = data.drop(['sex', 'age'], axis = 1)
Y = data['age']

#print(Y)

#stdScaler = StandardScaler()
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

#print('x_train number: ', X_train[0])
#print('x_test number: ', X_test.shape[0])
#print('Y_train number: ', Y_train[0])
#print('y_test number: ', Y_test)



#'''

regressor = SupervisedDBNRegression(hidden_layers_structure=[100,100],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=10,
                                    n_iter_backprop=200,
                                    batch_size=16,
                                    activation_function='relu')

#regressor.fit(X_train, Y_train)


# Save the model
#regressor.save('models/abalone_4.pkl')

# Restore it
#regressor = SupervisedDBNRegression.load('models/abalone_3.pkl')

# Test
Y_pred = regressor.predict(X_test)
print('Done.\nR-squared: %.4g\nMSE: %.4g\nMAPE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred), mean_absolute_percentage_error(Y_train, Y_pred)))



#'''


'''
#save to csv file

submission = pd.DataFrame({
        "Pred": [Y_pred]
    })

submission.to_csv("PredictedAge2.csv", index=False)
'''


#for x in range(1,20):
	#print('Y_test: ', Y_test[x], 'Y_pred: ', Y_pred[x])
