# %%
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import math
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# %%
ds = pd.read_csv('EmoSounds.csv')
ds.drop(ds.loc[:, 'dataset':'vocals'].columns, axis=1, inplace=True)
print(ds.head())

# %%
ds2 = pd.read_csv('EmoSounds.csv')
ds2.drop(ds2.loc[:, 'dynamics_rms_mean':'tonal_mode_std'].columns, axis=1, inplace=True)
ds2.drop(['dataset', 'fnames', 'splits', 'vocals'], axis=1, inplace=True)

# display(ds2.head())

# Visualizing dataset as a pairplot
sns.pairplot(ds2, hue="genre")
plt.show()

# %%
# Min-Max Scaling
ds_scaler = MinMaxScaler()
ds_scaled = pd.DataFrame(ds_scaler.fit_transform(ds)) 

X = ds_scaled.iloc[:,1].values.reshape(-1,1)
Y = ds_scaled.iloc[:,0].values.reshape(-1,1)

# %%
# Splitting the preprocess data with 60:20:20 ratio
x, x_test, y, y_test = train_test_split (X, Y, test_size = 0.2, train_size = 0.8, random_state = 1)
x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size = 0.25, train_size = 0.75, random_state = 1)

# %%
#Applying Linear Regression
model_lin = LinearRegression()
model_lin.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))

y_train_pred = model_lin.predict(x_train.reshape(-1,1))
y_test_pred = model_lin.predict(x_test.reshape(-1,1))
y_eval_pred = model_lin.predict(x_eval.reshape(-1,1))

plt.xlabel("valence")
plt.ylabel("arousal")
plt.scatter(x_train, y_train, s=10)
plt.plot(x_train, y_train_pred, color='red')
plt.show()

# %%
# Linear Regression RMSE
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
print("Training RMSE:", rmse_train)

rmse_eval = mean_squared_error(y_eval, y_eval_pred, squared=False)
print("Evaluation RMSE:", rmse_eval)

rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
print("Testing RMSE:", rmse_test)

# %%
# Using 5-fold cross validation
scores = cross_val_score(LinearRegression(), X, Y, cv=5)

print('\nAccuracy: %.3f with %.3f standard deviation' % (scores.mean(), scores.std()))
print('Overall RSME: %.3f' % (math.sqrt(np.mean(np.abs(scores)))))

# %%
polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(X)

# Splitting the preprocess data with 80:20 ratio
x_poly_train = x_poly[:-20]
y_train = Y[:-20]

x_poly_test = x_poly[:-20]
y_test = Y[:-20]

# %%
#Applying Non-Linear Regression
model_nonlin = LinearRegression()
model_nonlin.fit(x_poly_train, y_train)

y_train_pred = model_nonlin.predict(x_poly_train)
y_test_pred = model_nonlin.predict(x_poly_test)
y_eval_pred = model_nonlin.predict(x_poly)

plt.xlabel("valence")
plt.ylabel("arousal")
plt.scatter(X, Y, s=10)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_train_pred), key=sort_axis)
x_poly_train, y_train_pred = zip(*sorted_zip)
plt.plot(x_poly_train, y_train_pred, color='red')
plt.show()

# %%
# Non-Linear Regression RMSE
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
print("Training RMSE:", rmse_train)

rmse_eval = mean_squared_error(X, y_eval_pred, squared=False)
print("Evaluation RMSE:", rmse_eval)

rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
print("Testing RMSE:", rmse_test)

# %%
# Using 5-fold cross validation
clf = SVR(kernel="poly", C = 1.0, gamma="auto", degree=2, epsilon=0.1, coef0=1)
scores = cross_val_score(clf, X, Y.reshape(-1), cv=5)

print('\nAccuracy: %.3f with %.3f standard deviation' % (scores.mean(), scores.std()))
print('Overall RSME: %.3f' % (math.sqrt(np.mean(np.abs(scores)))))

# %%
# Dimensionality Reduction: Linear PCA
scaler = preprocessing.StandardScaler()
pca = PCA(n_components=2)

x_ds = scaler.fit_transform(ds)
pca.fit(x_ds)

X_pca = pca.transform(x_ds)

PC1_var, PC2_var = pca.explained_variance_ratio_

cm = plt.cm.get_cmap('plasma')
z = ds['valence']
z_min, z_max = z.min(), z.max()

plt.figure(figsize=(9,7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=z, vmin=z_min, vmax=z_max, alpha=0.5, cmap=cm)
plt.xlabel("PC1({:.2%})".format(PC1_var))
plt.ylabel("PC1({:.2%})".format(PC2_var))
plt.colorbar()
plt.show()

# %%
recon = pca.inverse_transform(pca.fit_transform(X_pca))
rmse_test = mean_squared_error(X_pca, recon, squared=False)
print("Testing RMSE of Linear PCA:", rmse_test)

# %%
# Feature Selection: f_regression
X = ds_scaled.iloc[:,ds.columns != 'arousal'].values

x, x_test, y, y_test = train_test_split (X, Y.ravel(), test_size = 0.2, train_size = 0.8, random_state = 1)
x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size = 0.25, train_size = 0.75, random_state = 1)

# %%
fs = SelectKBest(score_func = f_regression, k = 'all')

fs.fit(x_train, y_train)

X_train_fs = fs.transform(x_train)
X_test_fs = fs.transform(x_test)

y_test_pred = model_lin.predict(X_test_fs.reshape(-1,1))

y_test_list = []
for i in range(len(y_test)):
    y_test_list.append(y_test_pred[i])

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.xlabel("# of Features")
plt.ylabel("Score")
plt.show()

# %%
rmse_test = mean_squared_error(y_test, y_test_list, squared=False)
print("Testing RMSE of f_regression:", rmse_test)


