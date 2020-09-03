import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from dbn.tensorflow import SupervisedDBNRegression

from keras.utils.vis_utils import plot_model


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Loading dataset
boston = load_boston()
X, Y = boston.data, boston.target

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)


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
#regressor.fit(X_train, Y_train)


# Save the model
#regressor.save('model_regression_128.pkl')

# Restore it
regressor = SupervisedDBNRegression.load('models/model_regression.pkl')

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_train)
print('Done.\nR-squared: %f\nMSE: %f\nMAPE: %f' % (r2_score(Y_train, Y_pred), mean_squared_error(Y_train, Y_pred), mean_absolute_percentage_error(Y_train, Y_pred)))
plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#print(Y_pred)
