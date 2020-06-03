#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 21:08:00 2020

@author: swapnillagashe
"""

#Twitter Sentiment Analysis Kaggle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from keras import Sequential
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import keras
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from collections import Counter
import re
from nltk.tokenize import RegexpTokenizer 
import spacy
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
os.chdir('/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction')
train_data= pd.read_csv('train.csv')
test_data= pd.read_csv('test.csv')


stopwords=stopwords.words('English')
def remove_all_characters(string_):
    return(''.join(e for e in string_ if e.isalnum()))
    
def preprocess(string_):
    
    string_ = ('').join(i for i in string_ if not i.isdigit()) #remove any numeric values in string
    string_=str(string_)
    string_= string_.lower()
    
    string_=re.sub('[^A-Za-z0-9 ]+', ' ', string_)
    string_=string_.lstrip().rstrip()

    string_tokens = word_tokenize(string_) #tokenize as words
#   string_tokens=list(map(remove_all_characters,string_tokens))
#   string_tokens = [strings for strings in string_tokens if strings != ""]
    tokens_without_sw= [word for word in string_tokens if not word in stopwords]
#    tokens_without_punct=[word for word in tokens_without_sw if not word in punctuation ] #remove punctuations
    string_=(" ").join(tokens_without_sw)
    return string_

string_= ' 5***#corr4upted.okay.each..#$%^*afkam^...hjgldt4 '
""" Some Observations:
1. 'not' is present in the stopwords list which I think should not be there(May remove this later
2. I have considered one review as one string, May need to apply sentiment analysis individually on sentences within one review as well
3.take care of strings containing ***, this is mostly due to curse words which means a negative sentiment, can replace *** with a word with highly negative sentiment (like 'worst')
4. May remove '...'
5. check if the punctuation removal is working properly, because it has not removed * and .)
6. I have used padding, we can try to use one hot encoding as well or tfidf matrices
7. I havent used any bigrams yet, this may help in increasing accuracy
8. I can remove urls from the text in the cleaning process"""


#sample_train = train_data[0:200] # working on a sample set of 200 rows
#sample_train['cleaned'] = sample_train.apply(lambda x : preprocess(x['text']), axis=1)
train_data.drop(['textID'], axis=1,inplace=True)
test_data.drop(['textID'], axis=1,inplace=True)
train_data.dropna(inplace=True)
train_data['text']=train_data.apply(lambda x: str(x['text'].lower().lstrip()), axis=1)
train_data['selected_text']=train_data.apply(lambda x: str(x['selected_text'].lower().lstrip()), axis=1)
test_data['text']=test_data.apply(lambda x: str(x['text'].lower().lstrip()), axis=1)




train_data=train_data[train_data['sentiment']!='neutral']
test_data=test_data[test_data['sentiment']!='neutral']

train_data['cleaned'] = train_data.apply(lambda x : preprocess(x['text']), axis=1)
train_data['selected_cleaned'] = train_data.apply(lambda x : preprocess(x['selected_text']), axis=1)

test_data['cleaned'] = test_data.apply(lambda x : preprocess(x['text']), axis=1)
#train_data.isna()

# after this, lets join all the reviews to create a word to int mapping for all words
all_text_train = (' ').join(train_data['cleaned'])
all_text_train_selceted = (' ').join(train_data['selected_cleaned'])

#all_text_train = (' ').join(train_data['text'])

all_text_test = (' ').join(test_data['cleaned'])
#all_text_test = (' ').join(test_data['text'])

all_text = all_text_train + ' ' + all_text_test + ' ' + all_text_train_selceted
#all_text = all_text_train + ' ' + all_text_test
all_text=all_text.lower()

all_words = all_text.split()
count_words = Counter(all_words)
len_words = len(count_words)
sorted_words = count_words.most_common(len_words) # tells you how many times a word occurs in the entire corpus
vocab_to_int = {w: i+1 for i, (w,c) in enumerate (sorted_words)} #i+1 because we want to start with 1, 0 is reserved for padding
vocab_size=len_words
#apply encodings to text
dictarr = np.asarray(vocab_to_int.values()).reshape(-1, 1)
enc=OneHotEncoder()
enc.fit(dictarr)






import numpy as np
from sklearn.preprocessing import OneHotEncoder
wdict = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
dictarr1 = np.asarray(wdict.values()).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(d)
p=enc.transform([[0,1]]).toarray()
p
d=[list(wdict.values())]
wdict.keys.to_numpy()




def encoding(string):
    encodings=[vocab_to_int[w] for w in string.split()]
    return encodings

#sample_train['encoded'] = sample_train.apply(lambda x : encoding(x['cleaned']), axis=1 )
train_data['encoded'] = train_data.apply(lambda x : encoding(x['cleaned']), axis=1 )
#train_data['encoded'] = train_data.apply(lambda x : encoding(x['text']), axis=1 )
train_data['selected_cleaned_encoded'] = train_data.apply(lambda x : encoding(x['selected_cleaned']), axis=1 )
#errror may be because some words in selected text are  not in vocabulary i think
#train_data['selected_text_cleaned']=train_data['selected_text']



test_data['encoded'] = test_data.apply(lambda x : encoding(x['cleaned']), axis=1 )

#
#def label_encode(string):
#    if string=='negative':
#        return -1
#    elif string == 'positive':
#        return 1
#    else:
#        return 0 
#encode labels
#sample_train['label'] = sample_train.apply(lambda x:label_encode(x['sentiment']),axis=1)
#train_data['label'] = train_data.apply(lambda x:label_encode(x['sentiment']),axis=1)
#test_data['label'] = test_data.apply(lambda x:label_encode(x['sentiment']),axis=1)

# lets check the lengths of reviews

#sample_train['length'] = [len(x ) for x in sample_train['cleaned']]
#sample_train['length'].hist()
#sample_train['length'].describe()

train_data['length_cleaned'] = [len(x ) for x in train_data['cleaned']]
train_data['length_cleaned'].hist()
train_data['length_cleaned'].describe()

test_data['length_cleaned'] = [len(x ) for x in test_data['cleaned']]
test_data['length_cleaned'].hist()
train_data['no_cleaned_words'].describe()
""" we can remove extremely long reviews"""

# padding short reviews  with 0 and truncating long reviews
#sample_train['no_cleaned_words'] = sample_train.apply(lambda x: len(x['cleaned'].split()),axis=1) #creating a column with the count of number of cleaned words

train_data['no_cleaned_words'] = train_data.apply(lambda x: len(x['cleaned'].split()),axis=1)
test_data['no_cleaned_words'] = test_data.apply(lambda x: len(x['cleaned'].split()),axis=1)
train_data['no_cleaned_words_selected'] = train_data.apply(lambda x: len(x['selected_cleaned'].split()),axis=1)

sequence_length= 25
def pad_features(encoded_list,no_cleaned_words):
    if no_cleaned_words<sequence_length:
        pad=list(np.zeros(sequence_length-no_cleaned_words))
        new_pad=pad+encoded_list
        
    else:
        new_pad=encoded_list[0:sequence_length]
    
    return (new_pad)
        

#sample_train['padded'] = sample_train.apply(lambda x: pad_features(x['encoded'], x['no_cleaned_words']), axis=1)

train_data['padded_clened'] = train_data.apply(lambda x: pad_features(x['encoded'], x['no_cleaned_words']), axis=1)
train_data['padded_selected']= train_data.apply(lambda x: pad_features(x['selected_cleaned_encoded'], x['no_cleaned_words_selected']), axis=1)
test_data['padded_cleaned'] = test_data.apply(lambda x: pad_features(x['encoded'], x['no_cleaned_words']), axis=1)


train_data['padded_array']=train_data.apply(lambda x : np.asarray(x['padded_clened'],dtype=float), axis=1)
train_data['padded_array_selected']=train_data.apply(lambda x : np.asarray(x['padded_selected'],dtype=float), axis=1)

test_data['padded_array']=test_data.apply(lambda x : np.asarray(x['padded_cleaned'],dtype=float), axis=1)

train_data.columns







#convert padded list to padded array
#sample_train['padded_array']=sample_train.apply(lambda x : np.asarray(x['padded'],dtype=float), axis=1)
#train_data['padded_array']=train_data.apply(lambda x : np.asarray(x['padded'],dtype=float), axis=1)
#test_data['padded_array']=test_data.apply(lambda x : np.asarray(x['padded'],dtype=float), axis=1)

#train_array= np.stack(sample_train['padded_array'])
train_arrayX= np.stack(train_data['padded_array'])
train_arrayY=np.stack(train_data['padded_selected'])
to_predict_array= np.stack(test_data['padded_array'])

#note that test_array and to_predict_array are different
 
#test_array=pd.get_dummies(sample_train['label'])
#test_array=pd.get_dummies(train_data['label'])

#test_array= np.asarray(test_array,dtype=float)

trainX, testX, trainY, testY = train_test_split(train_arrayX, train_arrayY, test_size=0.2, random_state=42) #this is in series, we may need to convert it to a numpy array for feeding into the model

#defining the model
num_features = len(trainX[1])
n_hidden =256
num_folds=5
num_labels=3

""" for embedding layer:
    input_dim = size of vocabulary
    output_dim = length of output vector that we want, can be varied to any value
    input_length = length of input vector"""
input_dim=vocab_size
output_dim=50
input_length = 200
kfold=KFold(n_splits = num_folds,shuffle=True)


model = Sequential([
        keras.layers.Embedding(input_dim+1, output_dim), #note that the input dimension size has to be 1 more than the actual input_dim
        keras.layers.SpatialDropout1D(0.4),
        keras.layers.LSTM(n_hidden, dropout=0.2, recurrent_dropout=0.2),
        keras.layers.Dense(25, activation='softmax')
        
        ])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history= model.fit(trainX, trainY, batch_size=50, epochs=5)

plt.plot(history.history['accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

#We can see that the model is highly overfitting

########SVM Model
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(max_iter=1000, tol=1e-3)
trainX1, testX1, trainY1, testY1 = train_test_split(train_arrayX, train_arrayY, test_size=0.2, random_state=42) #this is in series, we may need to convert it to a numpy array for feeding into the model

clf.fit(trainX1, trainY1)

###############################################predicting on new data (later I will create seperate .py files for data cleaning, model and predictions)

#import data
data_to_predict= pd.read_csv('test.csv')
data_to_predict['cleaned'] = data_to_predict.apply(lambda x : preprocess(x['text']), axis=1)
data_to_predict['encoded'] = data_to_predict.apply(lambda x : encoding(x['cleaned']), axis=1 )


predictions = model.predict(to_predict_array)
pred_df= pd.DataFrame(predictions)
pred_df2=pred_df.copy()
pred_df2.columns=['negative', 'neutral', 'positive']
pred_df4=pred_df2.idxmax(axis=1)
test_data['predicted_label']=pred_df4
test_accuracy=(test_data[test_data['sentiment']==test_data['predicted_label']].shape[0])/test_data.shape[0]
test_accuracy

###predictions from SVM
predictions1 = clf.predict(to_predict_array)
pred_df1= pd.DataFrame(predictions1)
test_data['predicted_label_SVM']=pred_df1
test_accuracy_SVM=(test_data[test_data['sentiment']==test_data['predicted_label_SVM']].shape[0])/test_data.shape[0]
test_accuracy_SVM
#test accuracy is less because of overfitting, I can use more data during training to reduce this, add more dropout etc
#Atleast our NN is performing much better than the SVM






#-------------------------lets try a simple model without any feature engg and even cleaning the text
train_data['selected_text_cleaned'] = train_data.apply(lambda x : preprocess(x['selected_text']), axis=1)

train_data['selected_text_encoded'] = train_data.apply(lambda x : encoding(x['cleaned']), axis=1 )
