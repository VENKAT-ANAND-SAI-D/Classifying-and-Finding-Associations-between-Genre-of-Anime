import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('animeDataSet.csv')
dataset.drop(dataset.iloc[:,1:6], inplace=True,axis=1)
dataset.drop(dataset.iloc[:,4:12], inplace=True,axis=1)
dataset.drop(dataset.iloc[:,6:14], inplace=True,axis=1)
dataset.drop(dataset.iloc[:,[2,6,8,9,11]], inplace=True,axis=1)

dataset.drop(dataset[dataset.episodes<1].index,inplace=True)
dataset.drop(dataset[dataset.duration_min<1].index,inplace=True)
dataset.drop(dataset[dataset.type=='Music'].index,inplace=True)
dataset.drop(dataset[dataset.type=='Special'].index,inplace=True)
dataset.drop(dataset[(dataset['type']=='Movie') & (dataset['duration_min']<=15)].index,inplace=True)
dataset.describe()
dataset.info()

dat = dataset

x = dataset.iloc[:,[2,6]].values
y = dataset.iloc[:,1].values 

LE = LabelEncoder()
y = LE.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=0)

model = DecisionTreeClassifier(criterion='gini', random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

accuracies = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

ada_boost = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', random_state=0))
scores = cross_val_score(estimator=ada_boost, X=x, y=y, cv=10)
scores.mean()

parameters = [{'criterion':['gini']},{'criterion':['entropy']}]
grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy',cv=10)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_
print(best_accuracy)
