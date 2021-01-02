#importing modules
import numpy as np
import pandas as pd
from apyori import apriori

#importing the csv dataset
dataset = pd.read_csv('animeDataSet.csv')

#converting the genre column datatype to string
dataset.genre = dataset.genre.astype('str')

#appending the values of genre column in dataset to a list
genre_list=[]
for i in range(0,6668):
    genre_list.append([dataset.values[i,28]])

#declare another list and append the elements of genre_list by splitting them with ','
genre_list_mod=[]
for i in range(0,6668):
    for j in genre_list[i]:
        genre_list_mod.append(j.split(','))

#generating rules with apriori algorithm
rules = list(apriori(genre_list_mod, min_support=0.03, min_confidence=0.6, min_lift=3, min_length=2))

#visualizing the rules
for i in rules:
    print(i,"\n")