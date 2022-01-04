# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:35:41 2020

@author: Alec
"""

#Loading in packages
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import  LatentDirichletAllocation
from nltk.corpus import stopwords


"""This is the first task"""
"""Building a Word Cloud of Beatles Lyrics"""

#Defining the word cloud dictionary
dict = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

#Opening the files with the Beatles lyrics
with open('Maybe.csv', 'r') as f2:
    data = f2.read()
    print(data)
    
#Counting the number of unique words within the lyrics 
def Convert(string):
    li = list(string.split(" "))
    return li

print(Convert(data))

len(Convert(data))  

#Opening the abbey road image file
ar_mask = np.array(Image.open(path.join(dict, "Abbey Road.png")))

#Building the word cloud saving the word cloud and displaying the word cloud
wc = WordCloud(background_color = "white", max_words = 40000, mask = ar_mask, colormap = 'tab10')

wc.generate(data)

wc.to_file(path.join(dict, "Abbey Road Color.png"))

plt.imshow(ar_mask, cmap = plt.cm.gray, interpolation = 'bilinear')
plt.axis("off")
plt.figure()
plt.imshow(wc, interpolation = 'bilinear')
plt.axis("off")
plt.show()


#Defining the word cloud dictionary
dict = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

#Opening the files with the Beatles lyrics
with open('pm_lyrics.csv', 'r') as f2:
    data = f2.read()
    print(data)
    
#Counting the number of unique words within the lyrics 
def Convert(string):
    li = list(string.split(" "))
    return li

print(Convert(data))

len(Convert(data))  

#Opening the abbey road image file
ar_mask = np.array(Image.open(path.join(dict, "PM.png")))

#Building the word cloud saving the word cloud and displaying the word cloud
wc = WordCloud(background_color = "white", max_words = 90000, mask = ar_mask, color_func = lambda *args, **kwargs: "black")

wc.generate(data)

wc.to_file(path.join(dict, "PM WC.png"))

plt.imshow(ar_mask, cmap = plt.cm.gray, interpolation = 'bilinear')
plt.axis("off")
plt.figure()
plt.imshow(wc, interpolation = 'bilinear')
plt.axis("off")
plt.show()


#Defining the word cloud dictionary
dict = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

#Opening the files with the Beatles lyrics
with open('jl_lyrics.csv', 'r') as f2:
    data = f2.read()
    print(data)
    
#Counting the number of unique words within the lyrics 
def Convert(string):
    li = list(string.split(" "))
    return li

print(Convert(data))

len(Convert(data))  

#Opening the abbey road image file
ar_mask = np.array(Image.open(path.join(dict, "JL.png")))

#Building the word cloud saving the word cloud and displaying the word cloud
wc = WordCloud(background_color = "white", max_words = 35000, mask = ar_mask, color_func = lambda *args, **kwargs: "black")

wc.generate(data)

wc.to_file(path.join(dict, "JL WC.png"))

plt.imshow(ar_mask, cmap = plt.cm.gray, interpolation = 'bilinear')
plt.axis("off")
plt.figure()
plt.imshow(wc, interpolation = 'bilinear')
plt.axis("off")
plt.show()



"""This is now a different task"""
"""This is the sentiment bar plot task"""


#Creating a Bar Plot of sentiment for Beatles albums
    #Reading in the beatles songs data
songs = pd.read_csv('beatles_songs.txt', delimiter = "\t")
print(songs)

songs['compound'].mean()

testing = []
testing = songs['Lyrics'].tolist()

    #Extracting only the columns I need from the data
album = pd.DataFrame()
album = songs[['Album', 'Year', 'compound']].copy()

    #Placing the songs not in an album into their own data frame
unreleased = album
unreleased = unreleased.drop(unreleased.index[0:184])

    #Removing the songs not in an album from the album data frame
album = album.drop(album.index[184:209])

    #Grouping the album data frame by album and calculating the mean sentiment of each album
album_grouped = album.groupby('Album').mean()
album_grouped = album_grouped.sort_values(by = ['Year'])
print(album_grouped)

    #Grabbing the album title names from the index
Album_Titles = list(album_grouped.index)

    #Chaning the index to be numbers instead of album titles names
album_grouped = album_grouped.set_index(pd.Index([0,1,2,3,4,5,6,7,8,9,10,11,12,13]))

    #Adding the album title names back in as a column in the data frame
album_grouped.insert(loc = 0, column = 'Album', value = Album_Titles)

    #Creating a barplot of the albums sentiment
plt.bar(album_grouped['Album'], album_grouped['compound'])
plt.xticks(rotation = 90)


#The most positive songs of each album
album_pos = album.groupby('Album').max()
album_pos = album_pos.sort_values(by = ['Year'])

album_pos = album_pos.set_index(pd.Index([0,1,2,3,4,5,6,7,8,9,10,11,12,13]))

album_pos.insert(loc = 0, column = 'Album', value = Album_Titles)

plt.bar(album_pos['Album'], album_pos['compound'], album_grouped['compound'])
plt.xticks(rotation = 90)


#The most negative songs of each album
album_neg = album.groupby('Album').min()
album_neg = album_neg.sort_values(by = ['Year'])

album_neg = album_neg.set_index(pd.Index([0,1,2,3,4,5,6,7,8,9,10,11,12,13]))

album_neg.insert(loc = 0, column = 'Album', value = Album_Titles)

plt.bar(album_neg['Album'], album_neg['compound'])
plt.xticks(rotation = 90)


#Building a grouped bar chart
test = pd.DataFrame()
test = album_pos
test = test.append(album_grouped)
test = test.append(album_neg)

test = test.sort_values(by = ['Year', 'Album'])

sents = ['Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg']

test.insert(loc = 1, column = 'Sentiment', value = sents)

chart = sns.catplot(x = 'Album', y = 'compound', hue = 'Sentiment', data = test, kind = "bar", height = 6)
chart.set_xticklabels(rotation = 90)



"""This is now a different task"""
"""This is a sentiment Bar plot of John Lennon's Solo Career"""



#Creating a Bar Plot of sentiment for Beatles albums
    #Reading in the beatles songs data
jl_songs = pd.read_csv('jl_songs2.txt', delimiter = "\t")  
print(jl_songs)  

jl_songs['compound'].mean()

    #Extracting only the columns I need from the data
jl_albums = pd.DataFrame()
jl_album = jl_songs[['Album', 'Year', 'compound']].copy()

special = jl_album
special = special.drop(special.index[142])
special = special.drop(special.index[128:130])
special = special.drop(special.index[105:111])
special = special.drop(special.index[91])
special = special.drop(special.index[19:54])
special = special.drop(special.index[8])    
special = special.drop(special.index[0]) 

    #Grouping the album data frame by album and calculating the mean sentiment of each album
jl_album_grouped = special.groupby('Album').mean()
jl_album_grouped = jl_album_grouped.sort_values(by = ['Year'])
print(jl_album_grouped)

    #Grabbing the album title names from the index
jl_Album_Titles = list(jl_album_grouped.index)

    #Chaning the index to be numbers instead of album titles names
jl_album_grouped = jl_album_grouped.set_index(pd.Index([0,1,2,3,4,5,6,7,8]))

    #Adding the album title names back in as a column in the data frame
jl_album_grouped.insert(loc = 0, column = 'Album', value = jl_Album_Titles)

    #Creating a barplot of the albums sentiment
plt.bar(jl_album_grouped['Album'], jl_album_grouped['compound'])
plt.xticks(rotation = 90)


#The most positive songs of each album
jl_album_pos = special.groupby('Album').max()
jl_album_pos = jl_album_pos.sort_values(by = ['Year'])

jl_album_pos = jl_album_pos.set_index(pd.Index([0,1,2,3,4,5,6,7,8]))

jl_album_pos.insert(loc = 0, column = 'Album', value = jl_Album_Titles)

plt.bar(jl_album_pos['Album'], jl_album_pos['compound'], jl_album_grouped['compound'])
plt.xticks(rotation = 90)


#The most negative songs of each album
jl_album_neg = special.groupby('Album').min()
jl_album_neg = jl_album_neg.sort_values(by = ['Year'])

jl_album_neg = jl_album_neg.set_index(pd.Index([0,1,2,3,4,5,6,7,8]))

jl_album_neg.insert(loc = 0, column = 'Album', value = jl_Album_Titles)

plt.bar(jl_album_neg['Album'], jl_album_neg['compound'])
plt.xticks(rotation = 90)


#Building a grouped bar chart
jl_test = pd.DataFrame()
jl_test = jl_album_pos
jl_test = jl_test.append(jl_album_grouped)
jl_test = jl_test.append(jl_album_neg)

jl_test = jl_test.sort_values(by = ['Year', 'Album'])

jl_sents = ['Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg']

jl_test.insert(loc = 1, column = 'Sentiment', value = jl_sents)

jl_chart = sns.catplot(x = 'Album', y = 'compound', hue = 'Sentiment', data = jl_test, kind = "bar", height = 6)
jl_chart.set_xticklabels(rotation = 90)



"""This is now a different task"""
"""This is a sentiment Bar plot of Paul McCartney's Solo Career"""


#Creating a Bar Plot of sentiment for Beatles albums
    #Reading in the beatles songs data
pm_songs = pd.read_csv('pm_songs2.txt', delimiter = "\t")  
print(pm_songs)  

pm_songs['compound'].mean()

    #Extracting only the columns I need from the data
pm_albums = pd.DataFrame()
pm_ablum = pm_songs[['Album', 'Year', 'compound']].copy()     
    
pm_album = pd.DataFrame()
pm_album = pm_songs[['Album', 'Year', 'compound']].copy()

    #Removing Albums that are too small
pm_career = pm_album
pm_career = pm_career.drop(pm_career.index[348:353])
pm_career = pm_career.drop(pm_career.index[340])
pm_career = pm_career.drop(pm_career.index[299:313])
pm_career = pm_career.drop(pm_career.index[280:284])
pm_career = pm_career.drop(pm_career.index[279])
pm_career = pm_career.drop(pm_career.index[231:234])
pm_career = pm_career.drop(pm_career.index[154:156])
pm_career = pm_career.drop(pm_career.index[145])
pm_career = pm_career.drop(pm_career.index[123:127])
pm_career = pm_career.drop(pm_career.index[116])
pm_career = pm_career.drop(pm_career.index[82])
pm_career = pm_career.drop(pm_career.index[32:34])
pm_career = pm_career.drop(pm_career.index[0:8])

    #Grouping the album data frame by album and calculating the mean sentiment of each album
pm_album_grouped = pm_career.groupby('Album').mean()
pm_album_grouped = pm_album_grouped.sort_values(by = ['Year'])
print(pm_album_grouped)

    #Grabbing the album title names from the index
pm_Album_Titles = list(pm_album_grouped.index)

    #Chaning the index to be numbers instead of album titles names
pm_album_grouped = pm_album_grouped.set_index(pd.Index([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]))

    #Adding the album title names back in as a column in the data frame
pm_album_grouped.insert(loc = 0, column = 'Album', value = pm_Album_Titles)

    #Creating a barplot of the albums sentiment
plt.bar(pm_album_grouped['Album'], pm_album_grouped['compound'])
plt.xticks(rotation = 90)


#The most positive songs of each album
pm_album_pos = pm_career.groupby('Album').max()
pm_album_pos = pm_album_pos.sort_values(by = ['Year'])

pm_album_pos = pm_album_pos.set_index(pd.Index([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]))

pm_album_pos.insert(loc = 0, column = 'Album', value = pm_Album_Titles)

plt.bar(pm_album_pos['Album'], pm_album_pos['compound'], pm_album_grouped['compound'])
plt.xticks(rotation = 90)


#The most negative songs of each album
pm_album_neg = pm_career.groupby('Album').min()
pm_album_neg = pm_album_neg.sort_values(by = ['Year'])

pm_album_neg = pm_album_neg.set_index(pd.Index([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]))

pm_album_neg.insert(loc = 0, column = 'Album', value = pm_Album_Titles)

plt.bar(pm_album_neg['Album'], pm_album_neg['compound'])
plt.xticks(rotation = 90)


#Building a grouped bar chart
pm_plot = pd.DataFrame()
pm_plot = pm_album_pos
pm_plot = pm_plot.append(pm_album_grouped)
pm_plot = pm_plot.append(pm_album_neg)

pm_plot = pm_plot.sort_values(by = ['Year', 'Album'])

pm_sents = ['Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg', 'Pos', 'Mean', 'Neg']

pm_plot.insert(loc = 1, column = 'Sentiment', value = pm_sents)

pm_chart = sns.catplot(x = 'Album', y = 'compound', hue = 'Sentiment', data = pm_plot, kind = "bar", height = 6)
pm_chart.set_xticklabels(rotation = 90)



"""This is now a different task"""
"""This is the topic modeling part of the project"""
"""This is topic modeling for all of the Beatles songs"""



lyrics = songs['Lyrics'].tolist()

b_stopwords = stopwords.words('english')
b_add_list = ['na', 'la', 'oh', 'da', 'ah', 'yeah']
b_stopwords.extend(b_add_list)

#Vectorization of the Lyrics Data
lyrics_CV = CountVectorizer(stop_words = b_stopwords)
lyrics_vec = lyrics_CV.fit_transform(lyrics)

lyrics_colnames = lyrics_CV.get_feature_names()
#print(lyrics_colnames)

lyrics_DF = pd.DataFrame(lyrics_vec.toarray(), columns = lyrics_colnames)
#print(lyrics_DF)

#Time for the LDA
num_topics = 7

lyrics_lda = LatentDirichletAllocation(n_components = num_topics, max_iter = 100)
lyrics_lda_model = lyrics_lda.fit_transform(lyrics_DF)

word_topic = np.array(lyrics_lda.components_)
word_topic = word_topic.transpose()

num_top_words = 7 
vocab_array = np.asarray(lyrics_colnames)

fontsize_base = 10

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('BTopic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()



"""This is now a different task"""
"""This is topic modeling for Paul McCartney's career"""



#Loading in Paul's Lyrics data
pm_lyrics = pm_songs['Lyrics'].tolist()

#Vectorization of the Lyrics Data
pm_stopwords = stopwords.words('english')
pm_add_list = ['oh', 'ooh', 'la', 'yeah', 'ya']
pm_stopwords.extend(pm_add_list)


pm_lyrics_CV = CountVectorizer(stop_words = pm_stopwords)
pm_lyrics_vec = pm_lyrics_CV.fit_transform(pm_lyrics)

pm_lyrics_colnames = pm_lyrics_CV.get_feature_names()
#print(pm_lyrics_colnames)

pm_lyrics_DF = pd.DataFrame(pm_lyrics_vec.toarray(), columns = pm_lyrics_colnames)
#print(pm_lyrics_DF)

#Time for the LDA
pm_num_topics = 7

pm_lyrics_lda = LatentDirichletAllocation(n_components = pm_num_topics, max_iter = 100)
pm_lyrics_lda_model = pm_lyrics_lda.fit_transform(pm_lyrics_DF)

pm_word_topic = np.array(pm_lyrics_lda.components_)
pm_word_topic = pm_word_topic.transpose()

pm_num_top_words = 7
pm_vocab_array = np.asarray(pm_lyrics_colnames)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 10

for t in range(pm_num_topics):
    plt.subplot(1, pm_num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, pm_num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('PTopic #{}'.format(t))
    top_words_idx = np.argsort(pm_word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:pm_num_top_words]
    top_words = pm_vocab_array[top_words_idx]
    top_words_shares = pm_word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, pm_num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()



"""This is now a different task"""
"""This is the topic modeling part of the project"""
"""This is Topics modeling of John Lennon's Solo career"""



jl_lyrics = jl_songs['Lyrics'].tolist()

#Vectorization of the Lyrics Data
jl_stopwords = stopwords.words('english')
jl_add_list = ['uh', 'oh', 'ah', 'ooh', 'da', 'la', 'yeah']
jl_stopwords.extend(jl_add_list)


jl_lyrics_CV = CountVectorizer(stop_words = jl_stopwords)
jl_lyrics_vec = jl_lyrics_CV.fit_transform(jl_lyrics)

jl_lyrics_colnames = jl_lyrics_CV.get_feature_names()
#print(jl_lyrics_colnames)

jl_lyrics_DF = pd.DataFrame(jl_lyrics_vec.toarray(), columns = jl_lyrics_colnames)
#print(jl_lyrics_DF)

#Time for the LDA
num_topics = 7

jl_lyrics_lda = LatentDirichletAllocation(n_components = num_topics, max_iter = 100)
jl_lyrics_lda_model = jl_lyrics_lda.fit_transform(jl_lyrics_DF)

jl_word_topic = np.array(jl_lyrics_lda.components_)
jl_word_topic = jl_word_topic.transpose()

num_top_words = 7
jl_vocab_array = np.asarray(jl_lyrics_colnames)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 10

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('JTopic #{}'.format(t))
    top_words_idx = np.argsort(jl_word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = jl_vocab_array[top_words_idx]
    top_words_shares = jl_word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()