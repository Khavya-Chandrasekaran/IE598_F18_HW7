#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
df=pd.read_csv("K:\Khavya_UIUC\Fall 2019\MLF\Assignments\HW7\ccdefault.csv")

from sklearn.model_selection import train_test_split
X=df.iloc[:,1:23]
y=df.iloc[:,24]
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=33)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_train_std=StandardScaler().fit_transform(X_train)
X_test_std=StandardScaler().fit_transform(X_test)

#PCA
pca=PCA()
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
print(np.cumsum(pca.explained_variance_ratio_))

pca=PCA(n_components=21)
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=75, random_state=1, n_jobs=2)
start = time.process_time()
forest.fit(X_train_pca, y_train)
elapsed = (time.process_time() - start)
print("Processing time1 = " + str(elapsed))
y_train_pred=forest.predict(X_train_pca)
y_test_pred= forest.predict(X_test_pca)
score=accuracy_score(y_train, y_train_pred)
print("Training Accuracy= %.3f" %(score))
score1=accuracy_score(y_test, y_test_pred)
print("Test Acccuracy= %.3f" %(score1))


# In[41]:


print(forest.get_params())

params_forest = {'n_estimators':[50,100,350,500], 'max_features':['log2', 'auto', 'sqrt'], 'min_samples_leaf':[2,10,30]}
from sklearn.model_selection import GridSearchCV
grid_forest = GridSearchCV(estimator=forest, param_grid=params_forest, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
start = time.process_time()
grid_forest.fit(X_train_pca,y_train)
best_model = grid_forest.best_estimator_
elapsed = (time.process_time() - start)
print("Processing time1 = " + str(elapsed))
print(best_model)
y_pred_traingrid= best_model.predict(X_train_pca)
y_pred_testgrid = best_model.predict(X_test_pca)

from sklearn.model_selection import cross_val_score
score1=cross_val_score(forest, X_train_pca, y_train, cv=10)
print('Training Accuracy= {:.3f}'.format(np.mean(score1)))
score2=cross_val_score(forest, X_test_pca, y_test, cv=10)
print('Test Accuracy= {:.3f}'.format(np.mean(score2)))

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
df_col=df.columns[1:]
for f in range(X_train_pca.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, df_col[f], importances[indices[f]]))
plt.title('Feature Importances')
plt.bar(range(X_train_pca.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train_pca.shape[1]), df_col, rotation=90)
plt.xlim([-1, X_train_pca.shape[1]])
plt.show()

print("My name is Khavya Chandrasekaran")
print("My NetID is: khavyac2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
# In[ ]:




