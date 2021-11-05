# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 07:18:42 2020

@author: Alec
"""

#Load packages
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np


#Load in the Data
movie_overviews = "C://Users/Alec/Desktop/Syracuse Data Analytics/Data Analytics/Final Project/Overview.csv"

#Removing labels
overview_list= []
revenue_list = []


#Reading in the data from the csv file into the correct list
with open(movie_overviews, 'r', encoding = "utf8") as FILE:
    FILE.readline()
    for row in FILE:
        next_revenue, next_overview = row.split(",", 1)
        overview_list.append(next_overview)
        revenue_list.append(next_revenue)

#View the lists to make sure they contain what is expected
print(overview_list)
print(revenue_list)


#Vectorization and Word Frequency Counts
overview_cv = CountVectorizer(stop_words = 'english')
overview_vec = overview_cv.fit_transform(overview_list)

col_names_overview = overview_cv.get_feature_names()
print(col_names_overview)

DF_overview = pd.DataFrame(overview_vec.toarray(), columns = col_names_overview)
print(DF_overview)

DF_overview.insert(loc = 0, column = 'Revenue', value = revenue_list)
print(DF_overview)

DF_revenue_group = DF_overview.groupby('Revenue').sum()
print(DF_revenue_group)

#Transposing the data frame
DF_revenue_groupT = DF_revenue_group.transpose()


top_rev = DF_revenue_groupT.nlargest(10, ['Highest Revenue'])
bottom_rev = DF_revenue_groupT.nlargest(10, ['Lowest Revenue'])

top_rev = top_rev.drop(columns = 'Average Revenue')
top_rev = top_rev.drop(columns = 'High Revenue')
top_rev = top_rev.drop(columns = 'Very High Revenue')
top_rev = top_rev.drop(columns = 'Lowest Revenue')
top_rev = top_rev.drop(columns = 'Very Low Revenue')
top_rev = top_rev.drop(columns = 'Low Revenue')

print(top_rev)

bottom_rev = bottom_rev.drop(columns = 'Average Revenue')
bottom_rev = bottom_rev.drop(columns = 'High Revenue')
bottom_rev = bottom_rev.drop(columns = 'Very High Revenue')
bottom_rev = bottom_rev.drop(columns = 'Highest Revenue')
bottom_rev = bottom_rev.drop(columns = 'Very Low Revenue')
bottom_rev = bottom_rev.drop(columns = 'Low Revenue')

print(bottom_rev)


#Performing MNB
DF_overview_train, DF_overview_test = train_test_split(DF_overview, test_size = 0.2)       

#Remove the labels
DF_overview_train_revenue = pd.DataFrame()
DF_overview_train_revenue['Revenue'] = DF_overview_train['Revenue'].values
DF_overview_test_revenue = pd.DataFrame()
DF_overview_test_revenue['Revenue'] = DF_overview_test['Revenue'].values
del(DF_overview_train['Revenue'])
del(DF_overview_test['Revenue'])


#MNB
overview_NB = MultinomialNB()
overview_NB.fit(DF_overview_train, DF_overview_train_revenue)
overview_pred = overview_NB.predict(DF_overview_test)
overview_CNF = confusion_matrix(DF_overview_test_revenue, overview_pred)
#Confusion Matrix
print(overview_CNF)
#Confidence levels for predictions
print(np.round(overview_NB.predict_proba(DF_overview_test), 2))
#Classification report
print(classification_report(DF_overview_test_revenue, overview_pred))



#Predicting the New Movies Revenue

#Load in the Data
movie_new_overviews = "C://Users/Alec/Desktop/Syracuse Data Analytics/Data Analytics/Final Project/New Overview.csv"

#Removing labels
new_overview_list= []
new_revenue_list = []


#Reading in the data from the csv file into the correct list
with open(movie_new_overviews, 'r', encoding = "utf8") as FILE:
    FILE.readline()
    for row in FILE:
        next_new_revenue, next_new_overview = row.split(",", 1)
        new_overview_list.append(next_new_overview)
        new_revenue_list.append(next_new_revenue)

#View the lists to make sure they contain what is expected
print(new_overview_list)
print(new_revenue_list)


#Vectorization and Word Frequency Counts
new_overview_cv = CountVectorizer(stop_words = 'english')
new_overview_vec = new_overview_cv.fit_transform(new_overview_list)

col_names_new_overview = new_overview_cv.get_feature_names()
print(col_names_new_overview)

DF_new_overview = pd.DataFrame(new_overview_vec.toarray(), columns = col_names_new_overview)
print(DF_new_overview)

DF_new_overview.insert(loc = 0, column = 'Revenue', value = new_revenue_list)
print(DF_new_overview)


twilight_overview = DF_new_overview.drop(DF_new_overview.index[5:3488])

twilight_overview_revenue = pd.DataFrame()
twilight_overview_revenue['Revenue'] = twilight_overview['Revenue'].values
del(twilight_overview['Revenue'])
del(twilight_overview['00'])



twilight_pred = overview_NB.predict(twilight_overview)
print(np.round(overview_NB.predict_proba(twilight_overview), 2))

twilight_CNF = confusion_matrix(twilight_overview_revenue, twilight_pred)
print(twilight_CNF)


















































