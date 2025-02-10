# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:34:09 2025

@author: harpr
"""
#%%
import pandas as pd
import numpy as np

#%% Checking Null Values 
dataset = pd.read_csv('depression_data.csv')

#%% Gettting to know Dataset
print(dataset.head())

#%%
print(dataset.describe())

#%%
print(dataset.info())

#%%
dataset.drop(['Name'], axis =1, inplace = True)

#%%
dataset_sample = dataset.sample(n=50000, random_state=1)

#%%
X = dataset_sample.iloc[:, :-1].values
y = dataset_sample.iloc[:, -1].values

#%% Label encoder will convert yes/no to 1/0
from sklearn.preprocessing import LabelEncoder
le_employement = LabelEncoder()
X[:, 6] = le_employement.fit_transform(X[:, 6])

#%%
le_mental_illness = LabelEncoder()
X[:, 11] = le_mental_illness.fit_transform(X[:, 11])

#%%
le_substance_abuse = LabelEncoder()
X[:, 12] = le_substance_abuse.fit_transform(X[:, 12])

#%%
le_fam_history = LabelEncoder()
X[:, 13] = le_fam_history.fit_transform(X[:, 13])

#%%
le = LabelEncoder()
y = le.fit_transform(y)

#%%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 4, 5, 8, 9, 10])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#%% Model Dictionary
model_params = {
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['lbfgs', 'liblinear', 'saga']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }
}

#%%
from sklearn.model_selection import GridSearchCV

scores = []

for model_name, mp in model_params.items():
    print(f"Runing for {model_name}")
    
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    
    clf.fit(X_train, y_train)
    
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
    print(f"Done for {model_name}")

#%%
scores_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
scores_df.head()

#%%
import matplotlib.pyplot as plt
dataset.plot(x ='Age', y='Income', kind='line'),
plt.ylim(ymin=0)
plt.xlim(xmin=0)

plt.show()

#%%


#%%

