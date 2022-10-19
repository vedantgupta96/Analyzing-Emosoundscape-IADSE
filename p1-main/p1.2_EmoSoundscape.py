import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


ds = pd.read_csv('EmoSounds.csv')
ds.drop(ds.loc[:, 'dataset':'vocals'].columns, axis=1, inplace=True)
print(ds.head())


ds2 = pd.read_csv('EmoSounds.csv')
ds2.drop(ds2.loc[:, 'dynamics_rms_mean':'tonal_mode_std'].columns, axis=1, inplace=True)
ds2.drop(['dataset', 'fnames', 'splits', 'vocals'], axis=1, inplace=True)

# display(ds2.head())

# Visualizing dataset as a pairplot
sns.pairplot(ds2, hue="genre")
plt.show()


# Min-Max Scaling
ds_scaler = MinMaxScaler()
ds_scaled = pd.DataFrame(ds_scaler.fit_transform(ds)) 
# print(ds_scaled)

X = ds_scaled.iloc[:,1].values.reshape(-1,1)
y = ds_scaled.iloc[:,0].values.reshape(-1,1)


# Splitting the preprocess data with 80:20 ratio
train, test = train_test_split(ds_scaled, test_size=0.2)

x_train = train.iloc[:,1] # valence
y_train = train.iloc[:,0] # arousal

x_test = test.iloc[:,1] # valence
y_test = test.iloc[:,0] # arousal

eval = ds_scaled.iloc[:,1]


#Applying Linear Regression
model_lin = LinearRegression()
model_lin.fit(x_train.values.reshape(-1,1), y_train.values.reshape(-1,1))

y_train_pred = model_lin.predict(x_train.values.reshape(-1,1))
y_test_pred = model_lin.predict(x_test.values.reshape(-1,1))
eval_pred = model_lin.predict(eval.values.reshape(-1,1))

plt.xlabel("valence")
plt.ylabel("arousal")
plt.scatter(x_train, y_train, s=10)
plt.plot(x_train, y_train_pred, color='red')
plt.show()


# Linear Regression RMSE
rmse_train = math.sqrt(mean_squared_error(y_train, y_train_pred))
print("Training RMSE:", rmse_train)

rmse_eval = math.sqrt(mean_squared_error(eval, eval_pred))
print("Evaluation RMSE:", rmse_eval)

rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred))
print("Testing RMSE:", rmse_test)


# Using 5-fold cross validation
scores = cross_val_score(LinearRegression(), X, y, cv=5)

print('\nAccuracy: %.3f with %.3f standard deviation' % (scores.mean(), scores.std()))
print('Overall RSME: %.3f' % (math.sqrt(np.mean(np.abs(scores)))))


polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(X)

# Splitting the preprocess data with 80:20 ratio
x_poly_train = x_poly[:-20]
y_train2 = y[:-20]

x_poly_test = x_poly[:-20]
y_test2 = y[:-20]


#Applying Non-Linear Regression
model_nonlin = LinearRegression()
model_nonlin.fit(x_poly_train, y_train2)
y_train_pred2 = model_nonlin.predict(x_poly_train)
y_test_pred2 = model_nonlin.predict(x_poly_test)
eval_pred2 = model_nonlin.predict(x_poly)

plt.xlabel("valence")
plt.ylabel("arousal")
plt.scatter(X, y, s=10)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_train_pred2), key=sort_axis)
x_poly_train, y_train_pred2 = zip(*sorted_zip)
plt.plot(x_poly_train, y_train_pred2, color='red')
plt.show()


# Non-Linear Regression RMSE
rmse_train = math.sqrt(mean_squared_error(y_train2, y_train_pred2))
print("Training RMSE:", rmse_train)

rmse_eval = math.sqrt(mean_squared_error(X, eval_pred2))
print("Evaluation RMSE:", rmse_eval)

rmse_test = math.sqrt(mean_squared_error(y_test2, y_test_pred2))
print("Testing RMSE:", rmse_test)


# Using 5-fold cross validation
clf = SVR(kernel="poly", C = 1.0, gamma="auto", degree=2, epsilon=0.1, coef0=1)
scores = cross_val_score(clf, X, y.reshape(-1), cv=5)

print('\nAccuracy: %.3f with %.3f standard deviation' % (scores.mean(), scores.std()))
print('Overall RSME: %.3f' % (math.sqrt(np.mean(np.abs(scores)))))


