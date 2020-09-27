import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('housing.csv')

x_matrix = dataset['total_rooms']
y_matrix = dataset['median_house_value']

x_matrix = np.hstack((np.ones_like(x_matrix), x_matrix))
x_matrix = x_matrix.reshape(int(x_matrix.shape[0]/2), 2)

x_matrix /= np.max(x_matrix)

order = np.random.permutation(len(x_matrix))
portion = 20
test_x = np.array(x_matrix[order[:portion]])
test_y = np.array(y_matrix[order[:portion]])
train_x = np.array(x_matrix[order[portion:]])
train_y = np.array(y_matrix[order[portion:]])

def get_gradient(w, x, y):
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    gradient = -(1.0/len(x)) * error.dot(x)
    return y_estimate, gradient, np.power(error, 2)

w = np.random.randn(2)
alpha = 0.5 
tolerance = 1e-5

iterations = 1
while True:
    _, gradient, error = get_gradient(w, train_x, train_y)
    new_w = w - alpha * gradient
    
    if np.sum(abs(new_w - w)) < tolerance:
        print(f"Converged --- {new_w} --- {w}")
        break
        
    if iterations % 100 == 0:
        print(f'Iteration: {iterations} --- Error: {error}')
    
    iterations += 1
    w = new_w


pred, _, _ = get_gradient(w, test_x, test_y)
print(test_x, test_y)
print(pred)

y_matrix = np.hstack((np.ones_like(y_matrix), y_matrix))
y_matrix = y_matrix.reshape(int(y_matrix.shape[0]/2), 2)

f, (ax1, ax2) = plt.subplots(1, 2)

train_x.resize(train_y.shape)
print(x_matrix.shape, y_matrix.shape)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape, w.shape)

ax1.scatter(train_x, train_y)
ax1.plot(pred, color='g')
ax1.plot(test_y, color='r')
ax1.set_xlabel  = 'Total bedrooms'

ax2.scatter(train_x, train_y)
ax2.plot(pred, color='g')
ax2.plot(test_y, color='r')
ax2.set_xlabel  = 'Total bedrooms'
plt.show()


