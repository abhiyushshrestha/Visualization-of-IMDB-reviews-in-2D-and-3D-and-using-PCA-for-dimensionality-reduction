#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:44:47 2018

@author: abhiyush
"""

import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


#Importing the datasets using pandas
datasets = pd.read_csv("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/imdb_reviews.csv", delimiter = '\t', quoting = 3, header = None)

# Naming the columns
datasets.columns = ['reviews', 'likes']

#Initializing the object for lemmatizer
lemmatizer = WordNetLemmatizer()

#Preprocessing the datasets 
corpus = []
for i in range(0, len(datasets)):
    review = re.sub('[^a-zA-Z]', ' ', datasets['reviews'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)

#Creating the dataframe  
clean_data = pd.DataFrame({'Reviews' : corpus, 'Likes' : datasets['likes']})

corpus
#Combining all the sentences and form a collection of words
corpus_combine = ' '.join(corpus)

#Splitting the words with the help of space
words = corpus_combine.split()

#Count the words of each words
frequency_map = Counter(words)
len(frequency_map)

# making the list of number of count of words and store it into variable y
max_indices = 50
y = list(frequency_map.values())[:max_indices]
x = list(range(len(y)))

# Making a list of words and store it into variable words
words = list(frequency_map.keys())[:max_indices]

# Plotting the scatter plot
plt.scatter(x,y, c = 'r')

# Plotting the scatter plot with the labels for each points
fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(x,y)
for i,txt in enumerate(words):
    ax.annotate(txt, (x[i],y[i]))

# converting to list 
list(enumerate(words))

# Calculating the frequency distribution
tokens = nltk.word_tokenize(corpus_combine)
freq = FreqDist(tokens)
print(freq['movie'])

# Plotting the histogram using frequency distribution
freq.plot(40, cumulative = False)


# Creating a list of dictionary
training_data = []
for i in range(0,len(clean_data)):
    training_data.append({
            'Likes' : clean_data.loc[i,'Likes'],
            'Reviews' : clean_data.loc[i,'Reviews']
            })
    
# Initializing the dictionary 
likes_class = {}

# Taking the only the unique value and converting it into the list
likes = list(set([likes for likes in clean_data['Likes']]))

# creating a empty list for each 'likes'
for c in likes:
    likes_class[c] = []

# Storing the words into the dictionary according to the likes
for data in training_data:
    for word in nltk.word_tokenize(data['Reviews']):
        likes_class[data['Likes']].extend([word])


# Preprocessing to visualize the data using scatter plot by assigning different color for each positive and negative reviews
likes0 = ' '.join(likes_class[0])
likes1 = ' '.join(likes_class[1])

words0 = likes0.split()
words1 = likes1.split()

frequency_map0 = Counter(likes_class[0])
frequency_map1 = Counter(likes_class[1])

len(frequency_map)

max_indices = 50
y0 = list(frequency_map0.values())[:max_indices]
y1 = list(frequency_map1.values())[:max_indices]

x0 = list(range(len(y0)))
x1 = list(range(len(y1)))

words = list(frequency_map.keys())[:max_indices]


fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(x0,y0, c = 'r')
ax.scatter(x1,y1, c = 'b')


#Visualizing a sentences (i.e document)
corpus
type(corpus)
y = datasets.iloc[:,1].values

df = pd.DataFrame({"Reviews" : corpus, "Likes" : y})
df
df.shape

#Splitting the data sets to train and test sets 
x_train, x_test, y_train, y_test = train_test_split(corpus, y, test_size = 0.2, random_state = 0)

# Using tf-idf approach to create a matrix for the reviews
tfidf_vectorizer = TfidfVectorizer()
fit_tfidf_vectorizer = tfidf_vectorizer.fit(x_train)
print(fit_tfidf_vectorizer.get_feature_names())

transform_tfidf_vectorizer = tfidf_vectorizer.transform(x_train)
transform_tfidf_vectorizer_toarray = transform_tfidf_vectorizer.toarray()
transform_tfidf_vectorizer_toarray_test = tfidf_vectorizer.transform(x_test).toarray()

type(transform_tfidf_vectorizer_toarray)

# transforming the numpy array to list 
train_data = list(transform_tfidf_vectorizer_toarray)
type(train_data)

# Creating a dataframe of the train sets 
corpus_split  = pd.DataFrame({"Reviews": x_train, "Likes" : y_train})

# creating a dataframe of train sets after using tf-idf approach
train_data_df = pd.DataFrame(train_data, columns = fit_tfidf_vectorizer.get_feature_names())
train_data_df['Likes'] = y_train

train_data_df.shape

# Training the models using Naive Bayes MultinomialNB

# =============================================================================
# x_train = train_data_df.iloc[:, 0:2510].values
# y_train = train_data_df.iloc[:, 2510].values
# 
# =============================================================================

x_train = transform_tfidf_vectorizer_toarray
x_test = tfidf_vectorizer.transform(x_test).toarray()
x_train.shape
x_test.shape

# Creating a classifier to predict the model

classifier = MultinomialNB(alpha = 0.1)
classifier.fit(x_train, y_train
               
#Predicting using the Naive Bayes classifier
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0]+cm[1,1])/cm.sum() *100

print("The accuracy of the model is: ", accuracy)

# Applying PCA for dimensionality reduction
# creating a PCA with n_components 2 for 2D array
pca = PCA(n_components = 2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_


datasets_pca = pd.DataFrame(x_train_pca)
datasets_pca['Likes'] = y_train
x_train_pca[:,0]

# Creating a separate list for positive and negative reviews 

likes_class_pca_neg = []
likes_class_pca_pos = []

for i in range(0, len(datasets_pca)):
    if (datasets_pca.iloc[i, 2] == 0):
        likes_class_pca_neg.append(datasets_pca.iloc[i, 0:2].values)
    else:
        likes_class_pca_pos.append(datasets_pca.iloc[i, 0:2].values)

likes_class_pca_neg_df = pd.DataFrame(likes_class_pca_neg)
likes_class_pca_pos_df = pd.DataFrame(likes_class_pca_pos)

Xp_2d = likes_class_pca_pos_df.iloc[:,0]
Yp_2d = likes_class_pca_pos_df.iloc[:,1]

Xn_2d = likes_class_pca_neg_df.iloc[:,0]
Yn_2d = likes_class_pca_neg_df.iloc[:,1]

# Visualizing the datasets in 2D
fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(Xn_2d, Yn_2d, c = 'r')
ax.scatter(Xp_2d, Yp_2d, c = 'b')



#For visualizaing the datasets in 3D 

pca_3d = PCA(n_components = 3)
x_train_pca_3d = pca_3d.fit_transform(x_train)
x_test_pca_3d = pca_3d.transform(x_test)
explained_variance_3d = pca_3d.explained_variance_ratio_


datasets_pca_3d = pd.DataFrame(x_train_pca_3d)
datasets_pca_3d['Likes'] = y_train

likes_class_pca_neg_3d = []
likes_class_pca_pos_3d = []

for i in range(0, len(datasets_pca_3d)):
    if (datasets_pca_3d.iloc[i, 3] == 0):
        likes_class_pca_neg_3d.append(datasets_pca_3d.iloc[i, 0:3].values)
    else:
        likes_class_pca_pos_3d.append(datasets_pca_3d.iloc[i, 0:3].values)

likes_class_pca_neg_df_3d = pd.DataFrame(likes_class_pca_neg_3d)
likes_class_pca_pos_df_3d = pd.DataFrame(likes_class_pca_pos_3d)

Xp = likes_class_pca_pos_df_3d.iloc[:,0]
Yp = likes_class_pca_pos_df_3d.iloc[:,1]
Zp = likes_class_pca_pos_df_3d.iloc[:,2]

Xn = likes_class_pca_neg_df_3d.iloc[:,0]
Yn = likes_class_pca_neg_df_3d.iloc[:,1]
Zn = likes_class_pca_neg_df_3d.iloc[:,2]

fig, ax = plt.subplots(figsize = (10,10))
ax = Axes3D(fig)

ax.scatter(Xp, Yp, Zp, c = 'r', s = 150)
ax.scatter(Xn, Yn, Zn, c = 'b', s = 150)

# =============================================================================
# for i, txt in enumerate(filenames):
#     #ax.annotate(txt, (X[i],Y[i], Z[i]))
#     ax.text(Xp[i], Yp[i], Zp[i], txt, color='red')
# plt.show()
# 
# =============================================================================
# %matplotlib qt - to open the figure in separate application

