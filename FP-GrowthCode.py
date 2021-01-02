import numpy as np
import pandas as pd
import pyfpgrowth

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

#generating patterns and rules
patterns = pyfpgrowth.find_frequent_patterns(genre_list_mod,2)

rules = pyfpgrowth.generate_association_rules(patterns,0.7)

#visualizing the rules
for i in rules:
    print(i,"\n")