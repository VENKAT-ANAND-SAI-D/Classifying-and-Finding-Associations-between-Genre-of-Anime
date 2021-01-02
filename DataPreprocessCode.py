import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('animeDataSet.csv')
dataset.info()

dataset.drop(dataset.iloc[:,1:6], inplace=True,axis=1)
dataset.drop(dataset.iloc[:,4:23], inplace=True,axis=1)
dataset.drop(dataset.iloc[:,[2,5,6,8]], inplace=True,axis=1)

x = dataset['episodes']
plt.hist(x, bins=10)
plt.title('episodes')
plt.xlabel('no._of_episodes')
plt.ylabel('count')
plt.show()

x = dataset['duration_min']
plt.hist(x, bins=10)
plt.title('duration_min')
plt.xlabel('duration_min')
plt.ylabel('count')
plt.show()

x = dataset['popularity']
plt.hist(x, bins=10)
plt.title('popularity')
plt.xlabel('popularity')
plt.ylabel('count')
plt.show()

x = dataset['rank']
plt.hist(x, bins=10)
plt.title('rank')
plt.xlabel('rank')
plt.ylabel('count')
plt.show()

boxplt_dataset = dataset[['rank','popularity','episodes','duration_min']]
boxplt_dataset.boxplot()

dataset.drop(dataset[dataset.episodes<1].index,inplace=True)
dataset.drop(dataset[dataset.duration_min<1].index,inplace=True)
dataset.drop(dataset[dataset.type=='Music'].index,inplace=True)
dataset.drop(dataset[dataset.type=='Special'].index,inplace=True)
dataset.drop(dataset[(dataset['type']=='Movie') & (dataset['duration_min']<=30)].index,inplace=True)
dataset.describe()
dataset.info()




